from __future__ import annotations

import inspect
import time
from typing import Any

from pydantic import BaseModel

from .output import ToolOutput
from .registry import ToolSpec


class ToolExecutionResult(BaseModel):
    ok: bool
    output: ToolOutput | None = None
    error: str | None = None
    latency_ms: float

async def execute_tool(spec: ToolSpec, arguments: dict[str, Any]) -> ToolExecutionResult:
    start = time.perf_counter()
    if not isinstance(arguments, dict):
        return ToolExecutionResult(
            ok=False,
            error="tool arguments must be a JSON object",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    allowed_keys = set(spec.payload_model.model_fields.keys())
    extra_keys = sorted(key for key in arguments if key not in allowed_keys)
    if extra_keys:
        return ToolExecutionResult(
            ok=False,
            error=f"unexpected tool arguments: {', '.join(extra_keys)}",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    try:
        payload = spec.payload_model.model_validate(arguments)
        value = spec.func(payload)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, ToolOutput):
            return ToolExecutionResult(
                ok=False,
                error=f"tool `{spec.name}` must return ToolOutput",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        return ToolExecutionResult(
            ok=True,
            output=value,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as exc:
        return ToolExecutionResult(
            ok=False,
            error=str(exc) or exc.__class__.__name__,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
