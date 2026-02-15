from __future__ import annotations

import pytest
from pydantic import ValidationError

from fabrix.graph.state import NextState, ReasoningState, StateEnvelope


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
