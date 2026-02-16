from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from oauth_codex.tooling import callable_to_tool_schema
from pydantic import BaseModel


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    schema: dict[str, Any]
    payload_model: type[BaseModel]


class ToolRegistry:
    def __init__(self, specs: dict[str, ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = specs or {}

    @classmethod
    def from_callables(cls, tools: list[Callable[..., Any]] | None) -> ToolRegistry:
        specs: dict[str, ToolSpec] = {}
        if not tools:
            return cls(specs)

        for tool in tools:
            if not callable(tool):
                raise TypeError("tools must contain callable items")

            signature = inspect.signature(tool)
            fallback_name = str(getattr(tool, "__name__", "tool"))
            payload_model = cls._extract_payload_model(
                tool_name=fallback_name,
                signature=signature,
                tool=tool,
            )

            schema = callable_to_tool_schema(tool)
            name = str(schema.get("name") or fallback_name)
            if name in specs:
                raise ValueError(f"duplicate tool name: {name}")

            parameters = schema.get("parameters")
            if not isinstance(parameters, dict):
                raise TypeError(
                    f"tool `{name}` has invalid schema: `parameters` must be an object schema"
                )
            if parameters.get("type") != "object":
                raise TypeError(
                    f"tool `{name}` has invalid schema: `parameters.type` must be `object`"
                )

            description = str(
                schema.get("description")
                or (inspect.getdoc(tool) or f"Tool `{name}`").splitlines()[0]
            )

            specs[name] = ToolSpec(
                name=name,
                description=description,
                func=tool,
                schema=schema,
                payload_model=payload_model,
            )

        return cls(specs)

    @staticmethod
    def _extract_payload_model(
        *,
        tool_name: str,
        signature: inspect.Signature,
        tool: Callable[..., Any],
    ) -> type[BaseModel]:
        params = list(signature.parameters.values())
        if len(params) != 1:
            raise TypeError(
                f"tool `{tool_name}` must accept exactly one parameter typed as Pydantic BaseModel"
            )

        param = params[0]
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"tool `{tool_name}` has unsupported positional-only parameter `{param.name}`"
            )
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"tool `{tool_name}` has unsupported variadic parameter `{param.name}`"
            )
        if param.default is not inspect._empty:
            raise TypeError(f"tool `{tool_name}` parameter `{param.name}` must not have default value")

        try:
            type_hints = get_type_hints(tool)
        except Exception:
            type_hints = {}

        annotation = type_hints.get(param.name, param.annotation)
        if not isinstance(annotation, type) or not issubclass(annotation, BaseModel):
            raise TypeError(
                f"tool `{tool_name}` parameter `{param.name}` must be typed as Pydantic BaseModel"
            )

        return annotation

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema for spec in self._specs.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs.keys())

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs.values())
