from __future__ import annotations

from typing import Any, Protocol

from fabrix.graph.state import NextState, StateEnvelope


class StateProvider(Protocol):
    async def generate_state(
        self,
        *,
        task: str,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        current_state: NextState,
        step: int,
        tool_schemas: list[dict[str, Any]],
    ) -> StateEnvelope: ...
