from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from fabrix.graph.state import NextState, ReasoningState, StateEnvelope

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class SequenceProvider:
    def __init__(self, states: list[Any]) -> None:
        self._states = iter(states)

    async def generate_state(self, **_: Any) -> StateEnvelope:
        return StateEnvelope(state=next(self._states))


class LoopReasoningProvider:
    async def generate_state(self, **_: Any) -> StateEnvelope:
        return StateEnvelope(
            state=ReasoningState(
                next_state=NextState.reasoning,
                reasoning="keep working",
                focus="iteration",
            )
        )


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def agenerate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "state": {
                "state_type": "reasoning",
                "next_state": "finish",
                "reasoning": "done",
                "focus": "finalize",
            }
        }


@pytest.fixture
def sequence_provider_cls() -> type[SequenceProvider]:
    return SequenceProvider


@pytest.fixture
def loop_reasoning_provider_cls() -> type[LoopReasoningProvider]:
    return LoopReasoningProvider


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()
