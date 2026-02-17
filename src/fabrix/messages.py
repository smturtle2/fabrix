from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

type ImageInput = str | Path | bytes


def _require_non_empty_text(value: str, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_image_input(image: ImageInput) -> None:
    if isinstance(image, bytes):
        if not image:
            raise ValueError("image bytes must be non-empty")
        return
    if isinstance(image, Path):
        return
    if isinstance(image, str):
        if not image.strip():
            raise ValueError("image string must be non-empty")
        return
    raise TypeError("image must be a URL/path string, Path, or bytes")


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
        _validate_image_input(self.image)
        if self.text is not None:
            _require_non_empty_text(self.text, name="text")


def to_oauth_message(message: TextMessage | ImageMessage) -> dict[str, Any]:
    if isinstance(message, TextMessage):
        return {"role": message.role, "content": message.text}

    image_url = _coerce_image_to_url(message.image)
    content: list[dict[str, str]] = []
    if message.text is not None:
        content.append({"type": "input_text", "text": message.text})
    content.append({"type": "input_image", "image_url": image_url})
    return {"role": message.role, "content": content}


def _coerce_image_to_url(image: ImageInput) -> str:
    if isinstance(image, bytes):
        return _image_bytes_to_data_url(image)
    if isinstance(image, Path):
        return _image_path_to_data_url(image)
    value = image.strip()
    if value.startswith(("http://", "https://", "data:")):
        return value
    return _image_path_to_data_url(Path(value).expanduser())


def _image_path_to_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    raw = path.read_bytes()
    guessed = mimetypes.guess_type(path.name)[0]
    mime_type = guessed or _detect_image_mime(raw)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_bytes_to_data_url(raw: bytes) -> str:
    mime_type = _detect_image_mime(raw)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _detect_image_mime(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"
