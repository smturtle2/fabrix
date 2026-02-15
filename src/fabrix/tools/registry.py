from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from oauth_codex.tooling import callable_to_tool_schema


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    schema: dict[str, Any]
    signature: inspect.Signature
    type_hints: dict[str, Any]


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

            schema = callable_to_tool_schema(tool)
            name = str(schema.get("name") or getattr(tool, "__name__", "tool"))
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

            try:
                type_hints = get_type_hints(tool)
            except Exception:
                type_hints = {}

            specs[name] = ToolSpec(
                name=name,
                description=description,
                func=tool,
                schema=schema,
                signature=inspect.signature(tool),
                type_hints=type_hints,
            )

        return cls(specs)

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
