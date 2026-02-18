from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from fabrix.events.models import AgentEvent
from fabrix.graph.executor import GraphExecutor
from fabrix.llm.oauth_codex import DEFAULT_MODEL, OAuthCodexStateProvider
from fabrix.messages import ImageMessage, TextMessage
from fabrix.tools.registry import ToolRegistry

_DEFAULT_MAX_STEPS = 128


class Agent:
    def __init__(
        self,
        *,
        instructions: str | Callable[[], str],
        model: str = DEFAULT_MODEL,
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        self.tool_registry = ToolRegistry.from_callables(tools)
        self._validate_instructions(instructions)

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

    async def run_stream(
        self,
        *,
        messages: list[TextMessage | ImageMessage],
    ) -> AsyncIterator[AgentEvent]:
        if not messages:
            raise ValueError("messages must be a non-empty list of TextMessage/ImageMessage objects")
        if any(not isinstance(message, (TextMessage, ImageMessage)) for message in messages):
            raise TypeError("messages must be a list of TextMessage/ImageMessage objects")

        async for event in self._executor.run_stream(messages=messages):
            yield event

    @staticmethod
    def _validate_instructions(instructions: str | Callable[[], str]) -> None:
        if isinstance(instructions, str) or callable(instructions):
            return
        raise TypeError("instructions must be a string or a callable returning a string")
