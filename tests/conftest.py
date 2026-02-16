from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
def fake_client() -> FakeClient:
    return FakeClient()
