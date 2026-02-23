"""User-facing Fabrix agent runtime APIs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from fabrix.events.models import AgentEvent
from fabrix.graph.executor import GraphExecutor
from fabrix.graph.state import NextState
from fabrix.llm.oauth_codex import DEFAULT_MODEL, OAuthCodexStateProvider
from fabrix.messages import ImageMessage, TextMessage
from fabrix.tools.registry import ToolRegistry

_DEFAULT_MAX_STEPS = 128


class Agent:
    """Main entry point for running Fabrix agent workflows.

    The agent coordinates model state generation, tool execution, and event
    streaming behind a single async interface.

    Example:
        ```python
        from fabrix import Agent
        from fabrix.messages import TextMessage

        agent = Agent(instructions="You are a precise assistant.")
        async for event in agent.run_stream(messages=[TextMessage(text="Hello")]):
            ...
        ```
    """

    def __init__(
        self,
        *,
        instructions: str | Callable[[], str],
        default_model: str = DEFAULT_MODEL,
        state_models: Mapping[NextState | str, str] | None = None,
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        """Create an agent instance.

        Args:
            instructions: Static instructions string, or a zero-argument callable
                returning instructions at runtime.
            default_model: Fallback model for graph states that do not have an
                explicit override in ``state_models``.
            state_models: Optional per-state model overrides keyed by
                ``NextState`` or exact state name strings
                (``reasoning``, ``tool_call``, ``response``).
            tools: Optional list of tools. Each tool must accept exactly one
                Pydantic ``BaseModel`` argument and return ``ToolOutput``.

        Raises:
            TypeError: If ``instructions`` has an unsupported type, or if one or
                more tools have unsupported signatures/schemas.
            ValueError: If model settings are invalid, such as empty model names
                or conflicting state-model mappings.
        """
        self.tool_registry = ToolRegistry.from_callables(tools)
        self._validate_instructions(instructions)

        provider = OAuthCodexStateProvider(
            instructions=instructions,
            default_model=default_model,
            state_models=state_models,
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
        """Run the agent and stream typed execution events.

        Args:
            messages: Non-empty list of user input messages represented by
                ``TextMessage`` and/or ``ImageMessage`` objects.

        Yields:
            AgentEvent: Stream of reasoning, tool, response, and failure events
            produced step-by-step by the execution graph.

        Raises:
            ValueError: If ``messages`` is empty.
            TypeError: If any item in ``messages`` is not ``TextMessage`` or
                ``ImageMessage``.

        Example:
            ```python
            messages = [TextMessage(text="Use add_numbers to compute 3 + 9")]
            async for event in agent.run_stream(messages=messages):
                print(event.event_type)
            ```
        """
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
