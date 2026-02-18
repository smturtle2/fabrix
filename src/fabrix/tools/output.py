from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fabrix.media import ImageInput, coerce_image_to_tool_ref


class ToolTextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ToolImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    image_url: str = Field(min_length=1)
    caption: str | None = None

    @field_validator("image_url", mode="before")
    @classmethod
    def _coerce_image_url(cls, value: Any) -> str:
        if isinstance(value, (bytes, Path)):
            return coerce_image_to_tool_ref(value)

        if not isinstance(value, str):
            raise TypeError("image_url must be a URL/path string, Path, or bytes")

        stripped = value.strip()
        if not stripped:
            raise ValueError("image_url must be a non-empty string")
        return coerce_image_to_tool_ref(stripped)

    @field_validator("caption")
    @classmethod
    def _validate_caption(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("caption must be a non-empty string")
        return stripped


class ToolJSONPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json"] = "json"
    data: Any = Field(
        json_schema_extra={
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "integer"},
                {"type": "boolean"},
                {"type": "null"},
            ]
        }
    )

    @field_validator("data")
    @classmethod
    def _validate_json_serializable(cls, value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
        except TypeError as exc:
            raise ValueError("data must be JSON-serializable") from exc
        return value


ToolPart = Annotated[ToolTextPart | ToolImagePart | ToolJSONPart, Field(discriminator="type")]


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[ToolPart] = Field(min_length=1)

    @classmethod
    def text(cls, text: str) -> ToolOutput:
        return cls(parts=[ToolTextPart(text=text)])

    @classmethod
    def image(
        cls,
        image: ImageInput | str | Path | bytes,
        caption: str | None = None,
    ) -> ToolOutput:
        return cls(
            parts=[ToolImagePart(image_url=coerce_image_to_tool_ref(image), caption=caption)]
        )

    @classmethod
    def json(cls, data: Any) -> ToolOutput:
        return cls(parts=[ToolJSONPart(data=data)])

    @classmethod
    def compose(cls, parts: list[ToolPart]) -> ToolOutput:
        return cls(parts=parts)
