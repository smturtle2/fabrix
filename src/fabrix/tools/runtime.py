from __future__ import annotations

import inspect
import time
from typing import Any

from pydantic import BaseModel

from .registry import ToolSpec


class ToolExecutionResult(BaseModel):
    ok: bool
    output: Any | None = None
    error: str | None = None
    latency_ms: float


def _resolve_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _prepare_kwargs(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    params = [
        param
        for param in spec.signature.parameters.values()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    if not params:
        return {}

    allowed_names = {param.name for param in params}
    extra_names = [key for key in arguments if key not in allowed_names]

    if len(params) == 1:
        param = params[0]
        model_type = _resolve_model(spec.type_hints.get(param.name, param.annotation))
        if model_type is not None:
            payload = arguments.get(param.name, arguments)
            value = payload if isinstance(payload, model_type) else model_type.model_validate(payload)
            return {param.name: value}

    if extra_names:
        names = ", ".join(sorted(extra_names))
        raise ValueError(f"unexpected tool arguments: {names}")

    normalized: dict[str, Any] = {}
    for param in params:
        if param.name not in arguments:
            if param.default is inspect._empty:
                raise ValueError(f"missing required tool argument: {param.name}")
            continue

        raw_value = arguments[param.name]
        model_type = _resolve_model(spec.type_hints.get(param.name, param.annotation))
        if model_type is not None and not isinstance(raw_value, model_type):
            if not isinstance(raw_value, dict):
                raise TypeError(f"tool argument `{param.name}` must be an object")
            raw_value = model_type.model_validate(raw_value)

        normalized[param.name] = raw_value

    spec.signature.bind(**normalized)
    return normalized


def _normalize_output(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


async def execute_tool(spec: ToolSpec, arguments: dict[str, Any]) -> ToolExecutionResult:
    start = time.perf_counter()
    if not isinstance(arguments, dict):
        return ToolExecutionResult(
            ok=False,
            error="tool arguments must be a JSON object",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    try:
        kwargs = _prepare_kwargs(spec, arguments)
        value = spec.func(**kwargs)
        if inspect.isawaitable(value):
            value = await value
        return ToolExecutionResult(
            ok=True,
            output=_normalize_output(value),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as exc:
        return ToolExecutionResult(
            ok=False,
            error=str(exc) or exc.__class__.__name__,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
