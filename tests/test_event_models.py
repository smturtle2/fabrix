from __future__ import annotations

from fabrix.events import ResponseEvent


def test_response_event_allows_empty_payload() -> None:
    event = ResponseEvent(step=1, response=None, parts=None)
    assert event.response is None
    assert event.parts is None


def test_response_event_allows_parts_only() -> None:
    event = ResponseEvent(
        step=2,
        response=None,
        parts=[{"type": "image", "image_url": "https://example.com/resp.png"}],
    )
    assert event.response is None
    assert event.parts is not None
    assert event.parts[0].type == "image"
