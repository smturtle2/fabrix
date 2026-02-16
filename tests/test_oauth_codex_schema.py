from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

from fabrix.agent import Agent
from fabrix.errors import LLMOutputError
from fabrix.graph.state import NextState
from fabrix.graph.transitions import allowed_next_states
from fabrix.llm.oauth_codex import OAuthCodexStateProvider
from fabrix.tools.registry import ToolRegistry


class AddInput(BaseModel):
    a: int
    b: int


def add_numbers(payload: AddInput) -> int:
    return payload.a + payload.b


class RecursiveInput(BaseModel):
    value: int
    child: RecursiveInput | None = None


RecursiveInput.model_rebuild()


def consume_recursive(payload: RecursiveInput) -> int:
    return payload.value


def _assert_no_keywords(node: Any, keywords: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            assert key not in keywords
            _assert_no_keywords(value, keywords)
        return

    if isinstance(node, list):
        for item in node:
            _assert_no_keywords(item, keywords)


def test_tool_call_items_use_anyof_and_name_arguments_only(fake_client: Any) -> None:
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

    item_schema = any_of[0]
    assert set(item_schema["properties"]) == {"name", "arguments"}
    assert item_schema["required"] == ["name", "arguments"]


def test_state_schema_preserves_pydantic_min_length_constraints(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)

    reasoning_schema = provider._build_output_schema(
        current_state=NextState.reasoning,
        tool_schemas=[],
    )
    reasoning_props = reasoning_schema["json_schema"]["schema"]["properties"]["state"]["properties"]
    assert reasoning_props["reasoning"]["minLength"] == 1
    assert reasoning_props["focus"]["minLength"] == 1

    response_schema = provider._build_output_schema(
        current_state=NextState.response,
        tool_schemas=[],
    )
    response_props = response_schema["json_schema"]["schema"]["properties"]["state"]["properties"]
    assert response_props["response"]["minLength"] == 1


def test_output_schema_has_no_unsupported_keywords_for_all_states(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([add_numbers]).schemas()
    forbidden = {"$ref", "$defs", "definitions", "oneOf", "const"}

    for state in NextState:
        if state is NextState.tool_call:
            schema = provider._build_output_schema(current_state=state, tool_schemas=tool_schemas)
        else:
            schema = provider._build_output_schema(current_state=state, tool_schemas=[])
        _assert_no_keywords(schema, forbidden)


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
    state_type_schema = state_schema["properties"]["state_type"]
    assert state_type_schema.get("enum") == ["reasoning"]
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


@pytest.mark.asyncio
async def test_provider_prompt_serializes_non_json_values(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    now = datetime(2026, 1, 2, 3, 4, 5)

    await provider.generate_state(
        task="hello",
        context={"issued_at": now},
        history=[
            {
                "kind": "tool_result",
                "step": 1,
                "tool_name": "x",
                "call_id": "c1",
                "ok": True,
                "output": {"ts": now},
            }
        ],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    prompt = fake_client.calls[-1]["prompt"]
    assert "2026-01-02T03:04:05" in prompt


def test_prompt_graph_rules_are_derived_from_transitions(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    prompt = provider._build_prompt(
        task="hello",
        context={},
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    for state in NextState:
        allowed = "|".join(next_state.value for next_state in allowed_next_states(state))
        assert f"- {state.value} -> {allowed}" in prompt


def test_validate_tool_schemas_rejects_recursive_model(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([consume_recursive]).schemas()

    with pytest.raises(LLMOutputError, match="failed to strictify tool parameter schema"):
        provider.validate_tool_schemas(tool_schemas)


def test_agent_preflight_fails_fast_for_incompatible_tool_schema(fake_client: Any) -> None:
    with pytest.raises(LLMOutputError, match="failed to strictify tool parameter schema"):
        OAuthCodexStateProvider(instructions="x", client=fake_client).validate_tool_schemas(
            ToolRegistry.from_callables([consume_recursive]).schemas()
        )

    with pytest.raises(LLMOutputError, match="failed to strictify tool parameter schema"):
        Agent(
            instructions="x",
            model="gpt-5.3-codex",
            tools=[consume_recursive],
        )
