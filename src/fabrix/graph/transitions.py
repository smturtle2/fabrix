from __future__ import annotations

from .state import NextState

ALLOWED_TRANSITIONS: dict[NextState, set[NextState]] = {
    NextState.reasoning: {
        NextState.reasoning,
        NextState.tool_call,
        NextState.response,
    },
    NextState.tool_call: {
        NextState.reasoning,
        NextState.tool_call,
        NextState.response,
    },
    NextState.response: {
        NextState.reasoning,
        NextState.tool_call,
        NextState.response,
    },
}


def validate_transition(current: NextState, next_state: NextState) -> tuple[bool, str | None]:
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if next_state in allowed:
        return True, None
    return False, "invalid_transition"


def allowed_next_states(current: NextState) -> tuple[NextState, ...]:
    return tuple(sorted(ALLOWED_TRANSITIONS[current], key=lambda item: item.value))
