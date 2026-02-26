from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Any = None,
        headers: Any = None,
        json_data: Any = None,
        data: Any = None,
        files: Any = None,
        timeout: Any = None,
    ) -> Any:
        assert method == "POST"
        assert path == "/responses"
        assert isinstance(json_data, dict)

        self.calls.append(
            {
                "model": json_data.get("model"),
                "messages": json_data.get("input"),
                "instructions": json_data.get("instructions"),
                "text": json_data.get("text"),
                "reasoning": json_data.get("reasoning"),
                "store": json_data.get("store"),
                "stream": json_data.get("stream"),
            }
        )

        payload = json.dumps(
            {
                "state": {
                    "state_type": "reasoning",
                    "next_state": "response",
                    "reasoning": "done",
                    "focus": "finalize",
                }
            },
            ensure_ascii=True,
        )

        delta_event = json.dumps(
            {"type": "response.output_text.delta", "delta": payload},
            ensure_ascii=True,
        )
        completed_event = json.dumps(
            {"type": "response.completed", "response": {"output_text": payload}},
            ensure_ascii=True,
        )
        sse_text = (
            "event: response.output_text.delta\n"
            f"data: {delta_event}\n\n"
            "event: response.completed\n"
            f"data: {completed_event}\n\n"
        )
        return _FakeResponse(sse_text)


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()
