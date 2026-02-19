from __future__ import annotations

import pytest
from pydantic import ValidationError

from fabrix.graph.state import (
    MAX_TOOL_CALLS_PER_STEP,
    NextState,
    ReasoningState,
    ResponseState,
    StateEnvelope,
    ToolCallState,
)


def test_state_requires_next_state() -> None:
    with pytest.raises(ValidationError):
        ReasoningState(state_type="reasoning", reasoning="x", focus="y")


def test_state_envelope_discriminator_parse() -> None:
    payload = {
        "state": {
            "state_type": "reasoning",
            "next_state": "response",
            "reasoning": "Need to explain result",
            "focus": "clarity",
        }
    }
    parsed = StateEnvelope.model_validate(payload)
    assert parsed.state.state_type == "reasoning"
    assert parsed.state.next_state == NextState.response


def test_response_state_allows_null_next_state() -> None:
    state = ResponseState(next_state=None, response="done", audience="user")
    assert state.next_state is None


def test_response_state_allows_empty_payload() -> None:
    state = ResponseState(next_state=None, response=None, parts=None, audience="user")
    assert state.response is None
    assert state.parts is None


def test_response_state_allows_parts_only() -> None:
    state = ResponseState(
        next_state=None,
        response=None,
        parts=[{"type": "image", "image_url": "https://example.com/out.png"}],
        audience="user",
    )
    assert state.response is None
    assert state.parts is not None
    assert state.parts[0].type == "image"


def test_reasoning_and_tool_call_reject_null_next_state() -> None:
    with pytest.raises(ValidationError):
        ReasoningState(next_state=None, reasoning="x", focus="y")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ToolCallState(next_state=None, tool_calls=[{"name": "x", "arguments": {}}])  # type: ignore[arg-type]


def test_reasoning_and_focus_allow_non_ascii() -> None:
    state = ReasoningState(next_state=NextState.response, reasoning="작업 중", focus="정리")
    assert state.reasoning == "작업 중"
    assert state.focus == "정리"


def test_tool_call_state_rejects_too_many_tool_calls() -> None:
    with pytest.raises(ValidationError):
        ToolCallState(
            next_state=NextState.response,
            tool_calls=[
                {"name": "x", "arguments": {}}
                for _ in range(MAX_TOOL_CALLS_PER_STEP + 1)
            ],
        )
