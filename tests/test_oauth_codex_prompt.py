from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from fabrix.graph.state import NextState
from fabrix.graph.transitions import allowed_next_states
from fabrix.llm.oauth_codex import OAuthCodexStateProvider
from fabrix.messages import TextMessage


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

    system_first = fake_client.calls[0]["messages"][0]["content"]
    system_second = fake_client.calls[1]["messages"][0]["content"]
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

    with pytest.raises(TypeError, match="instructions must be a string or a callable returning a string"):
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

    assert fake_client.calls[-1]["reasoning_effort"] == "medium"


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

    system_message = fake_client.calls[-1]["messages"][0]["content"]
    assert isinstance(system_message, str)
    assert "Use the runtime context and instructions below as primary directives." in system_message
    assert "Instructions:\npolicy" in system_message
