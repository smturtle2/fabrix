from __future__ import annotations

import pytest
from pydantic import BaseModel

from fabrix.tools.registry import ToolRegistry
from fabrix.tools.runtime import execute_tool


class AddInput(BaseModel):
    a: int
    b: int


def add(payload: AddInput) -> int:
    return payload.a + payload.b


class MultInput(BaseModel):
    value: int


def multiply(*, payload: MultInput, factor: int) -> int:
    return payload.value * factor


@pytest.mark.asyncio
async def test_single_pydantic_parameter_is_validated() -> None:
    spec = ToolRegistry.from_callables([add]).get("add")
    assert spec is not None

    ok = await execute_tool(spec, {"a": 2, "b": 5})
    assert ok.ok is True
    assert ok.output == 7

    bad = await execute_tool(spec, {"a": "bad", "b": 5})
    assert bad.ok is False
    assert bad.error is not None


@pytest.mark.asyncio
async def test_multi_parameter_validation() -> None:
    spec = ToolRegistry.from_callables([multiply]).get("multiply")
    assert spec is not None

    ok = await execute_tool(spec, {"payload": {"value": 4}, "factor": 3})
    assert ok.ok is True
    assert ok.output == 12

    missing = await execute_tool(spec, {"payload": {"value": 4}})
    assert missing.ok is False
    assert "missing required" in (missing.error or "")


@pytest.mark.asyncio
async def test_multi_parameter_reports_unexpected_arguments() -> None:
    spec = ToolRegistry.from_callables([multiply]).get("multiply")
    assert spec is not None

    bad = await execute_tool(spec, {"payload": {"value": 4}, "factor": 3, "extra": 1})
    assert bad.ok is False
    assert bad.error == "unexpected tool arguments: extra"


@pytest.mark.asyncio
async def test_multi_parameter_reports_non_object_model_argument() -> None:
    spec = ToolRegistry.from_callables([multiply]).get("multiply")
    assert spec is not None

    bad = await execute_tool(spec, {"payload": 42, "factor": 2})
    assert bad.ok is False
    assert bad.error == "tool argument `payload` must be an object"


@pytest.mark.asyncio
async def test_tool_rejects_non_object_arguments() -> None:
    spec = ToolRegistry.from_callables([add]).get("add")
    assert spec is not None

    bad = await execute_tool(spec, "bad")  # type: ignore[arg-type]
    assert bad.ok is False
    assert bad.error == "tool arguments must be a JSON object"
