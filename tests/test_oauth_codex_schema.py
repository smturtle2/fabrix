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
from fabrix.messages import ImageMessage, TextMessage
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
    response_any_of = response_props["response"]["anyOf"]
    assert any(item.get("type") == "null" for item in response_any_of if isinstance(item, dict))
    assert any(item.get("type") == "string" for item in response_any_of if isinstance(item, dict))
    assert "parts" in response_props


def test_response_schema_json_part_data_has_explicit_type(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    response_schema = provider._build_output_schema(
        current_state=NextState.response,
        tool_schemas=[],
    )
    response_props = response_schema["json_schema"]["schema"]["properties"]["state"]["properties"]
    parts_any_of = response_props["parts"]["anyOf"]
    array_schema = next(
        item for item in parts_any_of if isinstance(item, dict) and item.get("type") == "array"
    )
    item_any_of = array_schema["items"]["anyOf"]
    json_part_schema = next(
        item
        for item in item_any_of
        if isinstance(item, dict)
        and item.get("properties", {}).get("type", {}).get("enum") == ["json"]
    )
    data_schema = json_part_schema["properties"]["data"]
    assert "type" in data_schema


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
        messages=[TextMessage(role="user", text="hello")],
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
    assert state_schema["properties"]["next_state"]["enum"] == [
        NextState.reasoning.value,
        NextState.tool_call.value,
        NextState.response.value,
    ]

    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=2,
        tool_schemas=[],
    )
    no_tool_output_schema = fake_client.calls[-1]["output_schema"]
    no_tool_state_schema = no_tool_output_schema["json_schema"]["schema"]["properties"]["state"]
    assert no_tool_state_schema["properties"]["next_state"]["enum"] == [
        NextState.reasoning.value,
        NextState.tool_call.value,
        NextState.response.value,
    ]


def test_response_schema_next_state_is_nullable(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    schema = provider._build_output_schema(current_state=NextState.response, tool_schemas=[])
    next_state_schema = schema["json_schema"]["schema"]["properties"]["state"]["properties"]["next_state"]
    any_of = next_state_schema.get("anyOf")
    assert isinstance(any_of, list)
    assert any(item.get("type") == "null" for item in any_of if isinstance(item, dict))
    non_null = [
        item for item in any_of if isinstance(item, dict) and item.get("type") != "null"
    ]
    assert len(non_null) == 1
    assert non_null[0].get("enum") == [
        NextState.reasoning.value,
        NextState.tool_call.value,
        NextState.response.value,
    ]


@pytest.mark.asyncio
async def test_provider_prompt_serializes_non_json_values(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    now = datetime(2026, 1, 2, 3, 4, 5)

    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
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

    prompt = fake_client.calls[-1]["messages"][0]["content"]
    assert "2026-01-02T03:04:05" in prompt


def test_prompt_graph_rules_are_derived_from_transitions(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    prompt = provider._build_prompt(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    for state in NextState:
        allowed = "|".join(next_state.value for next_state in allowed_next_states(state))
        assert f"- {state.value} -> {allowed}" in prompt


def test_prompt_includes_reasoning_loop_strategy(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    prompt = provider._build_prompt(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    assert "Reasoning loop strategy:" in prompt
    assert "Use Chain-of-Thought-style multi-step planning through short, visible decision traces." in prompt
    assert "Prefer English in reasoning and focus for consistency." in prompt
    assert "Keep each reasoning step to 1-2 sentences with one concrete focus." in prompt
    assert "If uncertainty remains, choose next_state=reasoning; usually resolve within several reasoning steps." in prompt
    assert "Each step must add new evidence or a new decision; do not repeat prior reasoning." in prompt
    assert (
        "With a finite step budget, avoid long reasoning-only loops and transition "
        "to tool_call/response as confidence grows."
    ) in prompt
    assert "Infer user intent from input messages before choosing next_state." in prompt
    assert "For image output, prefer using parts with type=image." in prompt
    assert "Empty responses are allowed (response=null and parts=null)." in prompt
    assert "Terminate by setting next_state=null in response state." in prompt


@pytest.mark.asyncio
async def test_provider_passes_messages_to_client_in_order(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    messages = [
        TextMessage(role="user", text="Describe this image"),
        ImageMessage(
            role="user",
            image="https://example.com/image.png",
            text="Focus on warnings",
        ),
        TextMessage(role="user", text="Focus on warnings"),
    ]

    await provider.generate_state(
        messages=messages,
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    sent_messages = fake_client.calls[-1]["messages"]
    assert [item["role"] for item in sent_messages[1:]] == ["user", "user", "user"]
    assert sent_messages[1]["content"] == "Describe this image"
    content = sent_messages[2]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["input_text", "input_image"]
    assert sent_messages[3]["content"] == "Focus on warnings"


@pytest.mark.asyncio
async def test_provider_appends_multimodal_history_image_parts(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[
            {
                "kind": "tool_result",
                "step": 2,
                "tool_name": "vision_tool",
                "call_id": "c2",
                "ok": True,
                "output": {
                    "parts": [
                        {
                            "type": "image",
                            "image_url": "https://example.com/tool.png",
                            "caption": "detected object",
                        }
                    ]
                },
            }
        ],
        current_state=NextState.reasoning,
        step=3,
        tool_schemas=[],
    )

    sent_messages = fake_client.calls[-1]["messages"]
    content = sent_messages[-1]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["input_text", "input_text", "input_image"]
    assert content[-1]["image_url"] == "https://example.com/tool.png"


@pytest.mark.asyncio
async def test_provider_appends_multimodal_history_text_and_json_parts(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[
            {
                "kind": "tool_result",
                "step": 4,
                "tool_name": "formatter",
                "call_id": "c4",
                "ok": True,
                "output": {
                    "parts": [
                        {"type": "text", "text": "formatted"},
                        {"type": "json", "data": {"a": 1}},
                    ]
                },
            }
        ],
        current_state=NextState.reasoning,
        step=5,
        tool_schemas=[],
    )

    sent_messages = fake_client.calls[-1]["messages"]
    content = sent_messages[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert "tool_result step=4" in content[0]["text"]
    assert any(part["type"] == "input_text" and part["text"] == "formatted" for part in content)
    assert any(part["type"] == "input_text" and part["text"] == '{"a": 1}' for part in content)


@pytest.mark.asyncio
async def test_provider_appends_failed_tool_history_as_single_text(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[
            {
                "kind": "tool_result",
                "step": 6,
                "tool_name": "broken",
                "call_id": "c6",
                "ok": False,
                "error": "boom",
            }
        ],
        current_state=NextState.reasoning,
        step=7,
        tool_schemas=[],
    )

    sent_messages = fake_client.calls[-1]["messages"]
    content = sent_messages[-1]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "input_text"
    assert "ok=False" in content[0]["text"]
    assert "error=boom" in content[0]["text"]


@pytest.mark.asyncio
async def test_provider_rejects_non_message_objects(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)

    with pytest.raises(TypeError, match="messages must contain TextMessage/ImageMessage objects"):
        await provider.generate_state(
            messages=[{"role": "user", "text": "hello"}],  # type: ignore[list-item]
            history=[],
            current_state=NextState.reasoning,
            step=1,
            tool_schemas=[],
        )


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
