from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fabrix.media import ImageInput, coerce_image_to_url


class ToolTextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ToolImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    image_url: str = Field(min_length=1)
    caption: str | None = None

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
    data: Any

    @field_validator("data")
    @classmethod
    def _validate_json_serializable(cls, value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=True)
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
            parts=[ToolImagePart(image_url=coerce_image_to_url(image), caption=caption)]
        )

    @classmethod
    def json(cls, data: Any) -> ToolOutput:
        return cls(parts=[ToolJSONPart(data=data)])

    @classmethod
    def compose(cls, parts: list[ToolPart]) -> ToolOutput:
        return cls(parts=parts)
