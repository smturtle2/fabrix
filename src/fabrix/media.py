from __future__ import annotations

import base64
import mimetypes
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

type ImageInput = str | Path | bytes

_URI_WITH_AUTHORITY_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_MIME_TO_SUFFIX: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def validate_image_input(image: ImageInput) -> None:
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


def coerce_image_to_url(image: ImageInput) -> str:
    validate_image_input(image)
    if isinstance(image, bytes):
        return _image_bytes_to_data_url(image)
    if isinstance(image, Path):
        return _image_path_to_data_url(image)
    value = image.strip()
    if value.startswith(("http://", "https://", "data:")):
        return value
    if value.startswith("file://"):
        return _image_path_to_data_url(file_url_to_path(value))
    if _has_uri_with_authority(value):
        return value
    return _image_path_to_data_url(resolve_local_image_path(value))


def coerce_image_to_tool_ref(image: ImageInput) -> str:
    validate_image_input(image)
    if isinstance(image, bytes):
        return str(bytes_to_temp_image_file(image))
    if isinstance(image, Path):
        return str(resolve_local_image_path(image))

    value = image.strip()
    if value.startswith(("http://", "https://", "data:")):
        return value
    if value.startswith("file://"):
        return str(resolve_local_image_path(file_url_to_path(value)))
    if _has_uri_with_authority(value):
        return value
    return str(resolve_local_image_path(value))


def resolve_local_image_path(value: str | Path) -> Path:
    candidate = value if isinstance(value, Path) else Path(value)
    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"image file not found: {candidate}")
    return resolved


def file_url_to_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme != "file":
        raise ValueError("file URL must start with file://")

    if parsed.netloc in ("", "localhost"):
        raw_path = parsed.path
    else:
        # Keep UNC-like authority paths intact.
        raw_path = f"//{parsed.netloc}{parsed.path}"
    if not raw_path:
        raise ValueError("file URL path must be non-empty")
    return Path(unquote(raw_path))


def bytes_to_temp_image_file(raw: bytes) -> Path:
    if not raw:
        raise ValueError("image bytes must be non-empty")

    mime_type = _detect_image_mime(raw)
    suffix = _MIME_TO_SUFFIX.get(mime_type, ".bin")
    fd, raw_path = tempfile.mkstemp(prefix="fabrix-image-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return Path(raw_path).resolve()


def _image_path_to_data_url(path: Path) -> str:
    resolved = resolve_local_image_path(path)
    raw = resolved.read_bytes()
    guessed = mimetypes.guess_type(resolved.name)[0]
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


def _has_uri_with_authority(value: str) -> bool:
    return bool(_URI_WITH_AUTHORITY_PATTERN.match(value))
