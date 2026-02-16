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


def add_async(payload: AddInput) -> int:
    return payload.a + payload.b


def multiply(*, payload: AddInput, factor: int) -> int:
    return (payload.a + payload.b) * factor


def positional_only_tool(value: int, /) -> int:
    return value


def variadic_args_tool(*values: int) -> int:
    return sum(values)


def variadic_kwargs_tool(**values: int) -> int:
    return sum(values.values())


def missing_hint_tool(value) -> int:  # type: ignore[no-untyped-def]
    return int(value)


def not_pydantic_tool(value: int) -> int:
    return value


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
async def test_single_parameter_reports_unexpected_arguments() -> None:
    spec = ToolRegistry.from_callables([add]).get("add")
    assert spec is not None

    bad = await execute_tool(spec, {"a": 2, "b": 5, "extra": 1})
    assert bad.ok is False
    assert bad.error == "unexpected tool arguments: extra"


@pytest.mark.asyncio
async def test_single_parameter_rejects_wrapped_payload() -> None:
    spec = ToolRegistry.from_callables([add]).get("add")
    assert spec is not None

    bad = await execute_tool(spec, {"payload": {"a": 3, "b": 4}})
    assert bad.ok is False
    assert bad.error == "unexpected tool arguments: payload"


@pytest.mark.asyncio
async def test_tool_rejects_non_object_arguments() -> None:
    spec = ToolRegistry.from_callables([add]).get("add")
    assert spec is not None

    bad = await execute_tool(spec, "bad")  # type: ignore[arg-type]
    assert bad.ok is False
    assert bad.error == "tool arguments must be a JSON object"


def test_registry_rejects_multi_parameter_tools() -> None:
    with pytest.raises(TypeError, match="must accept exactly one parameter"):
        ToolRegistry.from_callables([multiply])


def test_registry_rejects_positional_only_parameters() -> None:
    with pytest.raises(TypeError, match="unsupported positional-only parameter"):
        ToolRegistry.from_callables([positional_only_tool])


def test_registry_rejects_var_positional_parameters() -> None:
    with pytest.raises(TypeError, match="unsupported variadic parameter"):
        ToolRegistry.from_callables([variadic_args_tool])


def test_registry_rejects_var_keyword_parameters() -> None:
    with pytest.raises(TypeError, match="unsupported variadic parameter"):
        ToolRegistry.from_callables([variadic_kwargs_tool])


def test_registry_rejects_missing_required_type_hints() -> None:
    with pytest.raises(TypeError, match="must be typed as Pydantic BaseModel"):
        ToolRegistry.from_callables([missing_hint_tool])


def test_registry_rejects_non_pydantic_parameter() -> None:
    with pytest.raises(TypeError, match="must be typed as Pydantic BaseModel"):
        ToolRegistry.from_callables([not_pydantic_tool])
