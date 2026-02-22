from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fabrix.errors import RetryableLLMOutputError
from fabrix.events.models import (
    AgentEvent,
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    ToolEvent,
)
from fabrix.graph.state import NextState, ReasoningState, ResponseState, ToolCallState
from fabrix.llm import StateProvider
from fabrix.messages import ImageMessage, TextMessage
from fabrix.tools.registry import ToolRegistry
from fabrix.tools.runtime import ToolExecutionResult, execute_tool

_DEFAULT_LLM_RETRY_ATTEMPTS = 4


class GraphExecutor:
    def __init__(
        self,
        *,
        state_provider: StateProvider,
        tool_registry: ToolRegistry,
        max_steps: int,
        llm_retry_attempts: int = _DEFAULT_LLM_RETRY_ATTEMPTS,
    ) -> None:
        self._state_provider = state_provider
        self._tool_registry = tool_registry
        self._max_steps = max_steps
        self._llm_retry_attempts = max(0, llm_retry_attempts)

    async def run_stream(
        self,
        *,
        messages: list[TextMessage | ImageMessage],
    ) -> AsyncIterator[AgentEvent]:
        history: list[dict[str, Any]] = []
        current_state = NextState.reasoning
        tool_schemas = self._tool_registry.schemas()

        for step in range(1, self._max_steps + 1):
            try:
                envelope = await self._generate_state_with_retry(
                    messages=messages,
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
                    message=f"expected state_type `{current_state.value}`, got `{state.state_type}`",
                )
                return

            if isinstance(state, ReasoningState):
                yield ReasoningEvent(
                    step=step,
                    reasoning=state.reasoning,
                    focus=state.focus,
                    next_state=state.next_state,
                )
                history.append(
                    {
                        "kind": "reasoning",
                        "step": step,
                        "reasoning": state.reasoning,
                        "focus": state.focus,
                        "next_state": state.next_state.value,
                    }
                )

            if isinstance(state, ToolCallState):
                tool_call_history: list[dict[str, Any]] = []
                tool_result_history: list[dict[str, Any]] = []
                for idx, call in enumerate(state.tool_calls, start=1):
                    call_id = call.call_id or f"s{step}_c{idx}"
                    raw_arguments: Any = call.arguments
                    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
                    tool_call_history.append(
                        {
                            "name": call.name,
                            "call_id": call_id,
                            "arguments": arguments,
                            "why": call.why,
                        }
                    )

                    yield ToolEvent(
                        step=step,
                        phase="start",
                        tool_name=call.name,
                        call_id=call_id,
                        arguments=arguments,
                    )

                    spec = self._tool_registry.get(call.name)
                    if not isinstance(raw_arguments, dict):
                        result = self._tool_failure_result("tool arguments must be a JSON object")
                    elif spec is None:
                        result = self._tool_failure_result(f"tool not found: {call.name}")
                    else:
                        result = await execute_tool(spec, raw_arguments)

                    yield ToolEvent(
                        step=step,
                        phase="finish",
                        tool_name=call.name,
                        call_id=call_id,
                        arguments=arguments,
                        result=result,
                    )
                    tool_result_history.append(
                        {
                            "step": step,
                            "tool_name": call.name,
                            "call_id": call_id,
                            "ok": result.ok,
                            "output": (
                                result.output.model_dump(mode="json")
                                if result.output is not None
                                else None
                            ),
                            "error": result.error,
                        }
                    )
                history.append(
                    {
                        "kind": "tool_call",
                        "step": step,
                        "next_state": state.next_state.value,
                        "tool_calls": tool_call_history,
                        "tool_results": tool_result_history,
                    }
                )

            if isinstance(state, ResponseState):
                yield ResponseEvent(
                    step=step,
                    response=state.response,
                    parts=state.parts,
                )
                history.append(
                    {
                        "kind": "response",
                        "step": step,
                        "next_state": state.next_state.value if state.next_state is not None else None,
                        "response": state.response,
                        "parts": (
                            [part.model_dump(mode="json") for part in state.parts]
                            if state.parts is not None
                            else None
                        ),
                    }
                )
                if state.next_state is None:
                    return

            current_state = state.next_state

    @staticmethod
    def _tool_failure_result(error: str) -> ToolExecutionResult:
        return ToolExecutionResult(ok=False, error=error, latency_ms=0.0)

    async def _generate_state_with_retry(
        self,
        *,
        messages: list[TextMessage | ImageMessage],
        history: list[dict[str, Any]],
        current_state: NextState,
        step: int,
        tool_schemas: list[dict[str, Any]],
    ) -> Any:
        max_attempts = self._llm_retry_attempts + 1
        last_error: RetryableLLMOutputError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return await self._state_provider.generate_state(
                    messages=messages,
                    history=history,
                    current_state=current_state,
                    step=step,
                    tool_schemas=tool_schemas,
                )
            except RetryableLLMOutputError as exc:
                last_error = exc
                if attempt < max_attempts:
                    continue

        if last_error is None:
            raise RuntimeError("retry loop exited without state or error")

        raise RetryableLLMOutputError(
            f"{last_error} (retry attempts exhausted: {self._llm_retry_attempts})"
        ) from last_error
