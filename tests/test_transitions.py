from __future__ import annotations

from fabrix.graph.state import NextState
from fabrix.graph.transitions import validate_transition


def test_valid_transitions() -> None:
    assert validate_transition(NextState.reasoning, NextState.tool_call) == (True, None)
    assert validate_transition(NextState.response, NextState.finish) == (True, None)
    assert validate_transition(NextState.finish, NextState.finish) == (True, None)


def test_invalid_transition_returns_standard_code() -> None:
    ok, code = validate_transition(NextState.tool_call, NextState.finish)
    assert ok is False
    assert code == "invalid_transition"
