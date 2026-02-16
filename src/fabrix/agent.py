from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from fabrix.events.models import AgentEvent
from fabrix.graph.executor import GraphExecutor
from fabrix.llm.oauth_codex import DEFAULT_MODEL, OAuthCodexStateProvider
from fabrix.tools.registry import ToolRegistry

_DEFAULT_MAX_STEPS = 24


class Agent:
    def __init__(
        self,
        *,
        instructions: str,
        model: str = DEFAULT_MODEL,
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        self.tool_registry = ToolRegistry.from_callables(tools)

        provider = OAuthCodexStateProvider(
            instructions=instructions,
            model=model,
        )
        provider.validate_tool_schemas(self.tool_registry.schemas())

        self._executor = GraphExecutor(
            state_provider=provider,
            tool_registry=self.tool_registry,
            max_steps=_DEFAULT_MAX_STEPS,
        )

    async def run_task_stream(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._executor.run_stream(task=task, context=context):
            yield event
