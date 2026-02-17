from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .media import ImageInput, coerce_image_to_url, validate_image_input


def _require_non_empty_text(value: str, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class TextMessage:
    text: str
    role: str = "user"

    def __post_init__(self) -> None:
        _require_non_empty_text(self.role, name="role")
        _require_non_empty_text(self.text, name="text")

@dataclass(frozen=True, slots=True)
class ImageMessage:
    image: ImageInput
    text: str | None = None
    role: str = "user"

    def __post_init__(self) -> None:
        _require_non_empty_text(self.role, name="role")
        validate_image_input(self.image)
        if self.text is not None:
            _require_non_empty_text(self.text, name="text")


def to_oauth_message(message: TextMessage | ImageMessage) -> dict[str, Any]:
    if isinstance(message, TextMessage):
        return {"role": message.role, "content": message.text}

    image_url = coerce_image_to_url(message.image)
    content: list[dict[str, str]] = []
    if message.text is not None:
        content.append({"type": "input_text", "text": message.text})
    content.append({"type": "input_image", "image_url": image_url})
    return {"role": message.role, "content": content}
