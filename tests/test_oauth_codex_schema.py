from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import fabrix.llm.oauth_codex as oauth_codex_module
from fabrix.agent import Agent
from fabrix.errors import LLMOutputError
from fabrix.graph.state import MAX_TOOL_CALLS_PER_STEP, NextState
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


def _find_message_with_text(messages: list[dict[str, Any]], needle: str) -> dict[str, Any]:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and needle in content:
            return message
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and needle in text:
                return message
    raise AssertionError(f"message containing `{needle}` not found")


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


def test_tool_call_schema_has_max_items_constraint(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([add_numbers]).schemas()

    schema = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )
    tool_calls_schema = schema["json_schema"]["schema"]["properties"]["state"]["properties"][
        "tool_calls"
    ]
    assert tool_calls_schema["minItems"] == 1
    assert tool_calls_schema["maxItems"] == MAX_TOOL_CALLS_PER_STEP


def test_response_schema_json_part_data_uses_scalar_anyof(fake_client: Any) -> None:
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
    any_of = data_schema.get("anyOf")
    assert isinstance(any_of, list)
    types = {item.get("type") for item in any_of if isinstance(item, dict)}
    assert types == {"string", "number", "integer", "boolean", "null"}
    assert "array" not in types
    assert "object" not in types


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
        state.value for state in allowed_next_states(NextState.reasoning)
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
        state.value for state in allowed_next_states(NextState.reasoning)
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
    assert non_null[0].get("enum") == [state.value for state in allowed_next_states(NextState.response)]


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
    assert sent_messages[0]["role"] == "system"
    assert isinstance(sent_messages[0]["content"], str)
    assert [item["role"] for item in sent_messages[1:4]] == ["user", "user", "user"]
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
    tool_result_message = _find_message_with_text(sent_messages, "tool_result step=2 tool=vision_tool")
    content = tool_result_message["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["input_text", "input_text", "input_image"]
    assert content[-1]["image_url"] == "https://example.com/tool.png"


@pytest.mark.asyncio
async def test_provider_converts_local_image_history_part_to_data_url(
    fake_client: Any, tmp_path: Path
) -> None:
    image_file = tmp_path / "tool.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00")

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
                            "image_url": str(image_file),
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
    tool_result_message = _find_message_with_text(sent_messages, "tool_result step=2 tool=vision_tool")
    content = tool_result_message["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["input_text", "input_image"]
    assert content[-1]["image_url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_provider_skips_missing_local_image_history_part_with_warning(
    fake_client: Any, tmp_path: Path
) -> None:
    missing_path = tmp_path / "missing.png"

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
                            "image_url": str(missing_path),
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
    tool_result_message = _find_message_with_text(sent_messages, "tool_result step=2 tool=vision_tool")
    content = tool_result_message["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["input_text", "input_text", "input_text"]
    assert content[1]["text"] == "detected object"
    assert "tool_result image skipped:" in content[2]["text"]
    assert not any(part["type"] == "input_image" for part in content)


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
    tool_result_message = _find_message_with_text(sent_messages, "tool_result step=4 tool=formatter")
    content = tool_result_message["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert "tool_result step=4" in content[0]["text"]
    assert any(part["type"] == "input_text" and part["text"] == "formatted" for part in content)
    assert any(part["type"] == "input_text" and part["text"] == '{"a":1}' for part in content)


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
    tool_result_message = _find_message_with_text(sent_messages, "tool_result step=6 tool=broken")
    content = tool_result_message["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "input_text"
    assert "ok=False" in content[0]["text"]
    assert "error=boom" in content[0]["text"]


@pytest.mark.asyncio
async def test_provider_serializes_tool_call_history_with_tool_results(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[
            {
                "kind": "tool_call",
                "step": 2,
                "next_state": "reasoning",
                "tool_calls": [
                    {
                        "name": "vision_tool",
                        "call_id": "s2_c1",
                        "arguments": {"target": "error"},
                        "why": "inspect image",
                    }
                ],
                "tool_results": [
                    {
                        "step": 2,
                        "tool_name": "vision_tool",
                        "call_id": "s2_c1",
                        "ok": True,
                        "output": {"parts": [{"type": "text", "text": "detected"}]},
                        "error": None,
                    }
                ],
            }
        ],
        current_state=NextState.reasoning,
        step=3,
        tool_schemas=[],
    )

    sent_messages = fake_client.calls[-1]["messages"]
    tool_call_summary = _find_message_with_text(sent_messages, "tool_call step=2")
    summary_parts = tool_call_summary["content"]
    assert isinstance(summary_parts, list)
    assert any("planned_tool_call name=vision_tool" in part.get("text", "") for part in summary_parts)
    tool_result = _find_message_with_text(sent_messages, "tool_result step=2 tool=vision_tool")
    tool_result_parts = tool_result["content"]
    assert isinstance(tool_result_parts, list)
    assert any(part.get("text") == "detected" for part in tool_result_parts)


@pytest.mark.asyncio
async def test_provider_serializes_reasoning_and_response_history(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[
            {
                "kind": "reasoning",
                "step": 1,
                "reasoning": "need one more check",
                "focus": "validate result",
                "next_state": "response",
            },
            {
                "kind": "response",
                "step": 2,
                "next_state": "reasoning",
                "response": "intermediate",
                "parts": [
                    {"type": "json", "data": {"a": 1}},
                    {"type": "image", "image_url": "https://example.com/out.png", "caption": "preview"},
                ],
            },
        ],
        current_state=NextState.reasoning,
        step=3,
        tool_schemas=[],
    )

    sent_messages = fake_client.calls[-1]["messages"]
    reasoning_message = _find_message_with_text(sent_messages, "reasoning step=1")
    assert reasoning_message["role"] == "assistant"
    response_message = _find_message_with_text(sent_messages, "intermediate")
    response_parts = response_message["content"]
    assert isinstance(response_parts, list)
    assert any(part.get("type") == "output_text" and part.get("text") == "preview" for part in response_parts)
    assert any(part.get("type") == "input_image" for part in response_parts)


@pytest.mark.asyncio
async def test_provider_keeps_unknown_history_items_as_raw_text(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[{"kind": "x_custom", "value": {"a": 1}}],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    sent_messages = fake_client.calls[-1]["messages"]
    unknown_message = _find_message_with_text(sent_messages, "history_item index=0 raw=")
    content = unknown_message["content"]
    assert isinstance(content, list)
    assert '"kind":"x_custom"' in content[0]["text"]


@pytest.mark.asyncio
async def test_provider_serializes_input_messages_once_per_generate_state(
    fake_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    call_count = 0
    original = oauth_codex_module.to_oauth_message

    def wrapped(message: TextMessage | ImageMessage) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return original(message)

    monkeypatch.setattr(oauth_codex_module, "to_oauth_message", wrapped)

    messages = [
        TextMessage(role="user", text="hello"),
        ImageMessage(role="user", image="https://example.com/a.png"),
    ]
    await provider.generate_state(
        messages=messages,
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )
    assert call_count == len(messages)


def test_provider_output_schema_cache_returns_deep_copy(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    tool_schemas = ToolRegistry.from_callables([add_numbers]).schemas()

    schema_first = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )
    schema_first["json_schema"]["schema"]["properties"]["state"]["properties"]["tool_calls"][
        "minItems"
    ] = 999

    schema_second = provider._build_output_schema(
        current_state=NextState.tool_call,
        tool_schemas=tool_schemas,
    )
    assert (
        schema_second["json_schema"]["schema"]["properties"]["state"]["properties"]["tool_calls"][
            "minItems"
        ]
        == 1
    )


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
