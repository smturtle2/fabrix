from __future__ import annotations

from typing import Any, Protocol

from fabrix.graph.state import NextState, StateEnvelope
from fabrix.messages import ImageMessage, TextMessage


class StateProvider(Protocol):
    def validate_tool_schemas(self, tool_schemas: list[dict[str, Any]]) -> None: ...

    async def generate_state(
        self,
        *,
        messages: list[TextMessage | ImageMessage],
        history: list[dict[str, Any]],
        current_state: NextState,
        step: int,
        tool_schemas: list[dict[str, Any]],
    ) -> StateEnvelope: ...


__all__ = ["StateProvider"]
