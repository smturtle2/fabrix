"""User-facing message models for Fabrix agent inputs."""

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
    """Plain-text input message for the model.

    Args:
        text: Non-empty message content.
        role: Message role label passed to the model. Defaults to ``"user"``.

    Raises:
        TypeError: If ``role`` or ``text`` is not a string.
        ValueError: If ``role`` or ``text`` is empty after trimming.
    """

    text: str
    role: str = "user"

    def __post_init__(self) -> None:
        _require_non_empty_text(self.role, name="role")
        _require_non_empty_text(self.text, name="text")

@dataclass(frozen=True, slots=True)
class ImageMessage:
    """Multimodal message containing an image and optional text.

    Args:
        image: Image input as URL/path string, ``Path``, or raw ``bytes``.
        text: Optional non-empty text that accompanies the image.
        role: Message role label passed to the model. Defaults to ``"user"``.

    Raises:
        TypeError: If ``role`` or ``text`` has an invalid type, or ``image`` has
            an unsupported type.
        ValueError: If ``role`` or ``text`` is empty, or if ``image`` bytes/string
            are empty.
    """

    image: ImageInput
    text: str | None = None
    role: str = "user"

    def __post_init__(self) -> None:
        _require_non_empty_text(self.role, name="role")
        validate_image_input(self.image)
        if self.text is not None:
            _require_non_empty_text(self.text, name="text")


def to_oauth_message(message: TextMessage | ImageMessage) -> dict[str, Any]:
    """Convert a Fabrix message model into oauth-codex wire format.

    Args:
        message: Source ``TextMessage`` or ``ImageMessage``.

    Returns:
        dict[str, Any]: Message payload compatible with ``oauth-codex`` client
        ``messages`` input.

    Raises:
        FileNotFoundError: If an ``ImageMessage`` references a local file path
            that does not exist.
        ValueError: If image normalization fails due to invalid image input.

    Example:
        ```python
        payload = to_oauth_message(TextMessage(text="Hello"))
        # {"role": "user", "content": "Hello"}
        ```
    """
    if isinstance(message, TextMessage):
        return {"role": message.role, "content": message.text}

    image_url = coerce_image_to_url(message.image)
    content: list[dict[str, str]] = []
    if message.text is not None:
        content.append({"type": "input_text", "text": message.text})
    content.append({"type": "input_image", "image_url": image_url})
    return {"role": message.role, "content": content}
