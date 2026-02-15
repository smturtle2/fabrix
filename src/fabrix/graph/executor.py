from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fabrix.events.models import (
    AgentEvent,
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
    ToolEvent,
)
from fabrix.graph.state import FinishState, NextState, ReasoningState, ResponseState, ToolCallState
from fabrix.graph.transitions import validate_transition
from fabrix.llm import StateProvider
from fabrix.tools.registry import ToolRegistry
from fabrix.tools.runtime import ToolExecutionResult, execute_tool


class GraphExecutor:
    def __init__(
        self,
        *,
        state_provider: StateProvider,
        tool_registry: ToolRegistry,
        max_steps: int,
    ) -> None:
        self._state_provider = state_provider
        self._tool_registry = tool_registry
        self._max_steps = max_steps

    async def run_stream(
        self,
        *,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        payload = context or {}
        history: list[dict[str, Any]] = []
        current_state = NextState.reasoning
        last_response = ""
        tool_schemas = self._tool_registry.schemas()

        for step in range(1, self._max_steps + 1):
            try:
                envelope = await self._state_provider.generate_state(
                    task=task,
                    context=payload,
                    history=history,
                    current_state=current_state,
                    step=step,
                    tool_schemas=tool_schemas,
                )
            except Exception as exc:
                yield TaskFailedEvent(
                    step=step,
                    error_code="llm_error",
                    message=str(exc) or exc.__class__.__name__,
                )
                return

            state = envelope.state
            if state.state_type != current_state.value:
                yield TaskFailedEvent(
                    step=step,
                    error_code="invalid_state_type",
                    message=(
                        f"expected state_type `{current_state.value}`, got `{state.state_type}`"
                    ),
                )
                return

            is_valid, error_code = validate_transition(current_state, state.next_state)
            if not is_valid:
                yield TaskFailedEvent(
                    step=step,
                    error_code=error_code or "invalid_transition",
                    message=f"transition {state.state_type} -> {state.next_state.value} is not allowed",
                )
                return

            history.append({"kind": "state", "step": step, "state": state.model_dump(mode="python")})

            if isinstance(state, ReasoningState):
                yield ReasoningEvent(
                    step=step,
                    reasoning=state.reasoning,
                    focus=state.focus,
                    next_state=state.next_state,
                )

            if isinstance(state, ToolCallState):
                started_calls: dict[int, tuple[str, str, dict[str, Any]]] = {}
                results_by_index: dict[int, ToolExecutionResult] = {}
                pending_tasks: dict[int, asyncio.Task[ToolExecutionResult]] = {}

                for idx, call in enumerate(state.tool_calls, start=1):
                    call_id = call.call_id or f"s{step}_c{idx}"
                    raw_arguments: Any = call.arguments
                    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
                    started_calls[idx] = (call.name, call_id, arguments)

                    yield ToolEvent(
                        step=step,
                        phase="start",
                        tool_name=call.name,
                        call_id=call_id,
                        arguments=arguments,
                    )

                    spec = self._tool_registry.get(call.name)
                    if not isinstance(raw_arguments, dict):
                        results_by_index[idx] = self._tool_failure_result(
                            "tool arguments must be a JSON object"
                        )
                        continue
                    if spec is None:
                        results_by_index[idx] = self._tool_failure_result(
                            f"tool not found: {call.name}"
                        )
                        continue

                    pending_tasks[idx] = asyncio.create_task(execute_tool(spec, raw_arguments))

                if pending_tasks:
                    pending_indexes = tuple(pending_tasks.keys())
                    gathered = await asyncio.gather(
                        *(pending_tasks[idx] for idx in pending_indexes),
                        return_exceptions=True,
                    )
                    for idx, result in zip(pending_indexes, gathered, strict=True):
                        if isinstance(result, Exception):
                            results_by_index[idx] = self._tool_failure_result(
                                str(result) or result.__class__.__name__
                            )
                        else:
                            results_by_index[idx] = result

                for idx in sorted(started_calls):
                    tool_name, call_id, arguments = started_calls[idx]
                    result = results_by_index[idx]
                    yield ToolEvent(
                        step=step,
                        phase="finish",
                        tool_name=tool_name,
                        call_id=call_id,
                        arguments=arguments,
                        result=result,
                    )
                    history.append(
                        {
                            "kind": "tool_result",
                            "step": step,
                            "tool_name": tool_name,
                            "call_id": call_id,
                            "ok": result.ok,
                            "output": result.output,
                            "error": result.error,
                        }
                    )

            if isinstance(state, ResponseState):
                last_response = state.response
                yield ResponseEvent(step=step, response=state.response)
                history.append({"kind": "response", "step": step, "response": state.response})

            if isinstance(state, FinishState):
                yield TaskFinishedEvent(
                    step=step,
                    final_output=state.final_output,
                    completion_reason=state.completion_reason,
                )
                return

            current_state = state.next_state

        yield TaskFinishedEvent(
            step=self._max_steps,
            final_output=last_response,
            completion_reason="max_steps_reached",
        )

    @staticmethod
    def _tool_failure_result(error: str) -> ToolExecutionResult:
        return ToolExecutionResult(ok=False, error=error, latency_ms=0.0)
