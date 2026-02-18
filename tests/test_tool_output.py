from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fabrix.tools import ToolImagePart, ToolJSONPart, ToolOutput, ToolTextPart


def test_tool_output_text_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        ToolOutput.text("")


def test_tool_output_image_rejects_empty_bytes() -> None:
    with pytest.raises(ValueError, match="image bytes must be non-empty"):
        ToolOutput.image(b"")


def test_tool_output_image_rejects_missing_path() -> None:
    with pytest.raises(FileNotFoundError, match="image file not found"):
        ToolOutput.image("./this/path/does/not/exist.png")


def test_tool_output_image_converts_local_path(tmp_path: Path) -> None:
    image_file = tmp_path / "b.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00")

    output = ToolOutput.image(image_file)
    assert output.parts[0].type == "image"
    assert isinstance(output.parts[0], ToolImagePart)
    assert output.parts[0].image_url.startswith("data:image/png;base64,")


def test_tool_output_json_rejects_non_serializable_data() -> None:
    with pytest.raises(ValidationError, match="JSON-serializable"):
        ToolOutput.json({"path": Path("x")})


def test_tool_output_compose_accepts_mixed_parts() -> None:
    output = ToolOutput.compose(
        parts=[
            ToolTextPart(text="summary"),
            ToolJSONPart(data={"score": 9}),
        ]
    )
    assert output.model_dump(mode="json") == {
        "parts": [
            {"type": "text", "text": "summary"},
            {"type": "json", "data": {"score": 9}},
        ]
    }


def test_tool_image_part_accepts_bytes_input() -> None:
    part = ToolImagePart(image_url=b"\x89PNG\r\n\x1a\n\x00")
    assert part.image_url.startswith("data:image/png;base64,")


def test_tool_image_part_accepts_path_input(tmp_path: Path) -> None:
    image_file = tmp_path / "c.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    part = ToolImagePart(image_url=image_file)
    assert part.image_url.startswith("data:image/png;base64,")


def test_tool_image_part_preserves_non_path_string() -> None:
    part = ToolImagePart(image_url="cdn.example.com/image.png")
    assert part.image_url == "cdn.example.com/image.png"
