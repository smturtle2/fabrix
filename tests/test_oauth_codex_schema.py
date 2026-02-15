from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from fabrix.graph.state import NextState
from fabrix.llm.oauth_codex import OAuthCodexStateProvider
from fabrix.tools.registry import ToolRegistry


class AddInput(BaseModel):
    a: int
    b: int


def add_numbers(payload: AddInput) -> int:
    return payload.a + payload.b


class RowsInput(BaseModel):
    rows: list[dict]


def consume_rows(payload: RowsInput) -> int:
    return len(payload.rows)


class MetricsInput(BaseModel):
    metrics: dict[str, float]


def consume_metrics(payload: MetricsInput) -> int:
    return len(payload.metrics)


class NestedRow(BaseModel):
    category: str
    value: float


class NestedRowsInput(BaseModel):
    rows: list[NestedRow]


def consume_nested_rows(payload: NestedRowsInput) -> int:
    return len(payload.rows)


def _assert_no_keyword(node: Any, keyword: str) -> None:
    if isinstance(node, dict):
        assert keyword not in node
        for value in node.values():
            _assert_no_keyword(value, keyword)
        return

    if isinstance(node, list):
        for item in node:
            _assert_no_keyword(item, keyword)


def _assert_object_nodes_are_strict_and_consistent(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            properties = node.get("properties")
            required = node.get("required")
            assert isinstance(properties, dict)
            assert isinstance(required, list)
            assert required == list(properties.keys())
            assert node.get("additionalProperties") is False

        for value in node.values():
            _assert_object_nodes_are_strict_and_consistent(value)
        return

    if isinstance(node, list):
        for item in node:
            _assert_object_nodes_are_strict_and_consistent(item)


def test_tool_call_items_use_anyof_not_oneof(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([add_numbers]).schemas()

    schema = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )

    any_of = schema["json_schema"]["schema"]["properties"]["state"]["properties"]["tool_calls"][
        "items"
    ]["anyOf"]
    assert len(any_of) == 1

    arguments_schema = any_of[0]["properties"]["arguments"]
    assert arguments_schema["type"] == "object"
    assert arguments_schema["additionalProperties"] is False


def test_schema_contains_no_oneof_or_const(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([add_numbers]).schemas()

    schema = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )
    _assert_no_keyword(schema, "oneOf")
    _assert_no_keyword(schema, "const")
    _assert_no_keyword(schema, "$ref")
    _assert_no_keyword(schema, "$defs")
    _assert_no_keyword(schema, "definitions")


def test_all_object_nodes_are_strict_and_consistent(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([add_numbers]).schemas()

    schema = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )
    _assert_object_nodes_are_strict_and_consistent(schema["json_schema"]["schema"])


def test_list_dict_tool_input_is_normalized_to_empty_object_items(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([consume_rows]).schemas()

    schema = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )
    any_of = schema["json_schema"]["schema"]["properties"]["state"]["properties"]["tool_calls"][
        "items"
    ]["anyOf"]
    arguments_schema = any_of[0]["properties"]["arguments"]
    rows_items = arguments_schema["properties"]["rows"]["items"]

    assert rows_items["type"] == "object"
    assert rows_items["properties"] == {}
    assert rows_items["required"] == []
    assert rows_items["additionalProperties"] is False


def test_dict_tool_input_is_normalized_to_empty_object(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([consume_metrics]).schemas()

    schema = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )
    any_of = schema["json_schema"]["schema"]["properties"]["state"]["properties"]["tool_calls"][
        "items"
    ]["anyOf"]
    arguments_schema = any_of[0]["properties"]["arguments"]
    metrics_schema = arguments_schema["properties"]["metrics"]

    assert metrics_schema["type"] == "object"
    assert metrics_schema["properties"] == {}
    assert metrics_schema["required"] == []
    assert metrics_schema["additionalProperties"] is False


def test_nested_pydantic_refs_are_inlined(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([consume_nested_rows]).schemas()

    schema = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )
    _assert_no_keyword(schema, "$ref")
    _assert_no_keyword(schema, "$defs")
    _assert_no_keyword(schema, "definitions")


@pytest.mark.asyncio
async def test_provider_uses_dynamic_output_schema(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([add_numbers]).schemas()

    await provider.generate_state(
        task="hello",
        context={},
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=tool_schemas,
    )

    output_schema = fake_client.calls[-1]["output_schema"]
    assert isinstance(output_schema, dict)
    assert output_schema["type"] == "json_schema"

    state_schema = output_schema["json_schema"]["schema"]["properties"]["state"]
    assert state_schema["properties"]["state_type"]["enum"] == ["reasoning"]
    assert "tool_call" in state_schema["properties"]["next_state"]["enum"]

    await provider.generate_state(
        task="hello",
        context={},
        history=[],
        current_state=NextState.reasoning,
        step=2,
        tool_schemas=[],
    )
    no_tool_output_schema = fake_client.calls[-1]["output_schema"]
    no_tool_state_schema = no_tool_output_schema["json_schema"]["schema"]["properties"]["state"]
    assert "tool_call" not in no_tool_state_schema["properties"]["next_state"]["enum"]
