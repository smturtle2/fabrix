from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from oauth_codex import OAuthCodexClient

from fabrix.config import AgentConfig
from fabrix.events.models import AgentEvent
from fabrix.graph.executor import GraphExecutor
from fabrix.llm import StateProvider
from fabrix.llm.oauth_codex import OAuthCodexStateProvider
from fabrix.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        *,
        instructions: str,
        tools: list[Callable[..., Any]] | None = None,
        model: str = "gpt-5.3-codex",
        max_steps: int = 24,
        reasoning_effort: str = "medium",
        client: OAuthCodexClient | None = None,
        state_provider: StateProvider | None = None,
    ) -> None:
        self.config = AgentConfig(
            instructions=instructions,
            model=model,
            max_steps=max_steps,
            reasoning_effort=reasoning_effort,
        )
        self.tool_registry = ToolRegistry.from_callables(tools)

        provider = state_provider or OAuthCodexStateProvider(
            instructions=self.config.instructions,
            client=client,
            model=self.config.model,
            reasoning_effort=self.config.reasoning_effort,
        )

        self._executor = GraphExecutor(
            state_provider=provider,
            tool_registry=self.tool_registry,
            max_steps=self.config.max_steps,
        )

    async def run_task_stream(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._executor.run_stream(task=task, context=context):
            yield event
