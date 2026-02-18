from __future__ import annotations

from fabrix.graph.state import NextState
from fabrix.graph.transitions import validate_transition


def test_valid_transitions() -> None:
    for current in NextState:
        for next_state in NextState:
            assert validate_transition(current, next_state) == (True, None)


def test_invalid_transition_returns_standard_code() -> None:
    ok, code = validate_transition(NextState.tool_call, "invalid")  # type: ignore[arg-type]
    assert ok is False
    assert code == "invalid_transition"
