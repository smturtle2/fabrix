from __future__ import annotations

from pathlib import Path

import pytest

from fabrix.messages import ImageMessage, TextMessage, to_oauth_message


def test_text_message_accepts_text() -> None:
    msg = TextMessage(text="hello")
    assert msg.role == "user"
    assert msg.text == "hello"


def test_image_message_rejects_extra_fields() -> None:
    with pytest.raises(TypeError):
        ImageMessage(image="https://example.com/img.png", extra="x")


def test_text_message_rejects_extra_fields() -> None:
    with pytest.raises(TypeError):
        TextMessage(text="hello", extra="x")


def test_to_oauth_message_for_text() -> None:
    payload = to_oauth_message(TextMessage(role="user", text="hello"))
    assert payload == {"role": "user", "content": "hello"}


def test_to_oauth_message_for_image_url() -> None:
    payload = to_oauth_message(ImageMessage(image="https://example.com/img.png"))
    assert payload["role"] == "user"
    assert payload["content"][0]["type"] == "input_image"
    assert payload["content"][0]["image_url"] == "https://example.com/img.png"


def test_to_oauth_message_for_image_path(tmp_path: Path) -> None:
    image_file = tmp_path / "a.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00")

    payload = to_oauth_message(ImageMessage(image=image_file))
    assert payload["content"][0]["type"] == "input_image"
    assert payload["content"][0]["image_url"].startswith("data:image/png;base64,")


def test_to_oauth_message_for_image_bytes() -> None:
    payload = to_oauth_message(ImageMessage(image=b"\xff\xd8\xff\xe0\x00"))
    assert payload["content"][0]["type"] == "input_image"
    assert payload["content"][0]["image_url"].startswith("data:image/jpeg;base64,")


def test_to_oauth_message_for_missing_image_path() -> None:
    with pytest.raises(FileNotFoundError):
        to_oauth_message(ImageMessage(image="./this/path/does/not/exist.png"))
