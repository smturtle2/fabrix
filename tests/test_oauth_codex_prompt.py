from __future__ import annotations

from collections.abc import Mapping
import json
from datetime import datetime
from typing import Any

import pytest

from fabrix.graph.state import NextState
from fabrix.graph.transitions import allowed_next_states
from fabrix.llm.oauth_codex import DEFAULT_MODEL, OAuthCodexStateProvider
from fabrix.messages import TextMessage

_TOOL_SCHEMAS = [
    {
        "name": "echo",
        "description": "Echo tool",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    }
]


class _StateAwareClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Any = None,
        headers: Any = None,
        json_data: Any = None,
        data: Any = None,
        files: Any = None,
        timeout: Any = None,
    ) -> Any:
        assert method == "POST"
        assert path == "/responses"
        assert isinstance(json_data, dict)

        self.calls.append(json_data)

        state_type = json_data["text"]["format"]["schema"]["properties"]["state"]["properties"][
            "state_type"
        ]["enum"][0]

        if state_type == "reasoning":
            payload: dict[str, Any] = {
                "state": {
                    "state_type": "reasoning",
                    "next_state": "response",
                    "reasoning": "done",
                    "focus": "finalize",
                }
            }
        elif state_type == "tool_call":
            payload = {
                "state": {
                    "state_type": "tool_call",
                    "next_state": "response",
                    "tool_calls": [{"name": "echo", "arguments": {"value": "ok"}}],
                }
            }
        else:
            payload = {
                "state": {
                    "state_type": "response",
                    "next_state": None,
                    "response": "done",
                    "parts": None,
                    "audience": "user",
                }
            }

        content = json.dumps(payload, ensure_ascii=True)
        delta_event = json.dumps(
            {"type": "response.output_text.delta", "delta": content},
            ensure_ascii=True,
        )
        sse_text = f"event: response.output_text.delta\ndata: {delta_event}\n\n"

        class _Resp:
            def __init__(self, text: str) -> None:
                self.text = text

        return _Resp(sse_text)


class _ConflictingStateModels(Mapping[NextState | str, str]):
    def __getitem__(self, key: NextState | str) -> str:
        if isinstance(key, NextState) and key is NextState.reasoning:
            return "model-a"
        if isinstance(key, str) and key == "reasoning":
            return "model-b"
        raise KeyError(key)

    def __iter__(self):
        yield NextState.reasoning
        yield "reasoning"

    def __len__(self) -> int:
        return 2


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


def _build_prompt(fake_client: Any, *, has_tools: bool) -> str:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    return provider._build_prompt(has_tools=has_tools)


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

    sent_messages = fake_client.calls[-1]["messages"]
    history_message = _find_message_with_text(sent_messages, "tool_result step=1")
    content = history_message["content"]
    assert isinstance(content, list)
    assert any("2026-01-02T03:04:05" in part.get("text", "") for part in content)


@pytest.mark.asyncio
async def test_provider_resolves_instruction_callable_for_each_prompt(fake_client: Any) -> None:
    call_count = 0

    def dynamic_instructions() -> str:
        nonlocal call_count
        call_count += 1
        return f"dynamic-{call_count}"

    provider = OAuthCodexStateProvider(instructions=dynamic_instructions, client=fake_client)

    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=2,
        tool_schemas=[],
    )

    system_first = fake_client.calls[0]["instructions"]
    system_second = fake_client.calls[1]["instructions"]
    assert isinstance(system_first, str)
    assert isinstance(system_second, str)
    assert "dynamic-1" in system_first
    assert "dynamic-2" in system_second
    assert call_count == 2


@pytest.mark.asyncio
async def test_provider_does_not_send_state_control_message(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    sent_messages = fake_client.calls[-1]["messages"]
    for message in sent_messages:
        content = message.get("content")
        if isinstance(content, str):
            assert "state_control:" not in content
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                assert "state_control:" not in text


@pytest.mark.asyncio
async def test_provider_rejects_instruction_callable_returning_non_string(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(
        instructions=lambda: 123,  # type: ignore[return-value]
        client=fake_client,
    )

    with pytest.raises(
        TypeError, match="instructions must be a string or a callable returning a string"
    ):
        await provider.generate_state(
            messages=[TextMessage(role="user", text="hello")],
            history=[],
            current_state=NextState.reasoning,
            step=1,
            tool_schemas=[],
        )
    assert not fake_client.calls


@pytest.mark.asyncio
async def test_provider_uses_low_reasoning_effort_by_default(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)

    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    assert fake_client.calls[-1]["reasoning"]["effort"] == "medium"


@pytest.mark.asyncio
async def test_provider_routes_model_per_state() -> None:
    client = _StateAwareClient()
    provider = OAuthCodexStateProvider(
        instructions="x",
        client=client,
        state_models={
            NextState.reasoning: "reasoning-model",
            "tool_call": "tool-model",
            NextState.response: "response-model",
        },
    )

    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.tool_call,
        step=2,
        tool_schemas=_TOOL_SCHEMAS,
    )
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.response,
        step=3,
        tool_schemas=[],
    )

    assert [call["model"] for call in client.calls] == [
        "reasoning-model",
        "tool-model",
        "response-model",
    ]


@pytest.mark.asyncio
async def test_provider_uses_default_model_for_unset_state() -> None:
    client = _StateAwareClient()
    provider = OAuthCodexStateProvider(
        instructions="x",
        client=client,
        state_models={"reasoning": "reasoning-model"},
    )

    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.response,
        step=2,
        tool_schemas=[],
    )

    assert [call["model"] for call in client.calls] == ["reasoning-model", DEFAULT_MODEL]


@pytest.mark.asyncio
async def test_provider_uses_custom_default_model_for_unset_state() -> None:
    client = _StateAwareClient()
    provider = OAuthCodexStateProvider(
        instructions="x",
        client=client,
        default_model="gpt-5.3-codex-custom",
        state_models={"reasoning": "reasoning-model"},
    )

    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )
    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.response,
        step=2,
        tool_schemas=[],
    )

    assert [call["model"] for call in client.calls] == ["reasoning-model", "gpt-5.3-codex-custom"]


def test_provider_rejects_empty_default_model(fake_client: Any) -> None:
    with pytest.raises(ValueError, match="default_model must be a non-empty string"):
        OAuthCodexStateProvider(
            instructions="x",
            client=fake_client,
            default_model="   ",
        )


def test_provider_rejects_non_string_default_model(fake_client: Any) -> None:
    with pytest.raises(TypeError, match="default_model must be a non-empty string"):
        OAuthCodexStateProvider(
            instructions="x",
            client=fake_client,
            default_model=123,  # type: ignore[arg-type]
        )


def test_provider_rejects_conflicting_state_model_keys(fake_client: Any) -> None:
    with pytest.raises(ValueError, match="conflicting model mappings for state `reasoning`"):
        OAuthCodexStateProvider(
            instructions="x",
            client=fake_client,
            state_models=_ConflictingStateModels(),
        )


def test_provider_rejects_unknown_state_model_key(fake_client: Any) -> None:
    with pytest.raises(ValueError, match="unsupported state_models key"):
        OAuthCodexStateProvider(
            instructions="x",
            client=fake_client,
            state_models={"reason": "model-a"},
        )


def test_provider_rejects_non_string_state_model_key(fake_client: Any) -> None:
    with pytest.raises(TypeError, match="state_models keys must be NextState or string"):
        OAuthCodexStateProvider(
            instructions="x",
            client=fake_client,
            state_models={1: "model-a"},  # type: ignore[dict-item]
        )


def test_provider_rejects_empty_state_model_value(fake_client: Any) -> None:
    with pytest.raises(
        ValueError, match="state_models value for `reasoning` must be a non-empty string"
    ):
        OAuthCodexStateProvider(
            instructions="x",
            client=fake_client,
            state_models={"reasoning": "   "},
        )


def test_provider_rejects_non_string_state_model_value(fake_client: Any) -> None:
    with pytest.raises(
        TypeError, match="state_models value for `reasoning` must be a non-empty string"
    ):
        OAuthCodexStateProvider(
            instructions="x",
            client=fake_client,
            state_models={"reasoning": 123},  # type: ignore[dict-item]
        )


def test_prompt_includes_full_transition_rules(fake_client: Any) -> None:
    prompt = _build_prompt(fake_client, has_tools=False)

    shared_allowed = " | ".join(
        next_state.value for next_state in allowed_next_states(NextState.reasoning)
    )
    assert f"{{ reasoning | tool_call }} -> {{ {shared_allowed} }}" in prompt
    assert f"{{ response }} -> {{ {shared_allowed} | null }}" in prompt


@pytest.mark.parametrize(
    "marker",
    [
        "Input messages JSON:",
        "Execution history JSON:",
        "Available tools JSON schema:",
        "Instructions:",
    ],
)
def test_prompt_excludes_dynamic_context_dumps(fake_client: Any, marker: str) -> None:
    prompt = _build_prompt(fake_client, has_tools=True)
    assert marker not in prompt


def test_prompt_reasoning_strategy_avoids_rigid_plan_prefixes(fake_client: Any) -> None:
    prompt = _build_prompt(fake_client, has_tools=True)
    assert "define or refine a short plan" not in prompt

    required_fragments = [
        "Treat `reasoning` as a short decision journal",
        "Consider a task non-trivial when it involves multiple tools",
        "do not transition on the same reasoning step where you first form a candidate decision",
        "End reasoning only when you can state a concrete transition reason",
        "If that challenge pass invalidates the transition reason, continue reasoning",
    ]
    for fragment in required_fragments:
        assert fragment in prompt


def test_prompt_frames_autonomous_agent_identity(fake_client: Any) -> None:
    prompt = _build_prompt(fake_client, has_tools=True)
    required_fragments = [
        "You are an autonomous agent operating on a graph state machine.",
        "Act with bounded autonomy and a consistent personality",
        "instructions section in this system message",
    ]
    for fragment in required_fragments:
        assert fragment in prompt


@pytest.mark.asyncio
async def test_system_message_includes_instructions_and_runtime_context(
    fake_client: Any,
) -> None:
    provider = OAuthCodexStateProvider(instructions="policy", client=fake_client)

    await provider.generate_state(
        messages=[TextMessage(role="user", text="hello")],
        history=[],
        current_state=NextState.reasoning,
        step=1,
        tool_schemas=[],
    )

    system_message = fake_client.calls[-1]["instructions"]
    assert isinstance(system_message, str)
    assert "Use the runtime context and instructions below as primary directives." in system_message
    assert "Instructions:\npolicy" in system_message
