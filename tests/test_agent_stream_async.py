from __future__ import annotations

import pytest
from pydantic import BaseModel

from fabrix.agent import Agent
from fabrix.events import (
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
    ToolEvent,
)
from fabrix.graph.state import (
    FinishState,
    NextState,
    ReasoningState,
    ResponseState,
    StateEnvelope,
    ToolCallState,
)
from fabrix.llm.oauth_codex import OAuthCodexStateProvider


class AddInput(BaseModel):
    a: int
    b: int


def add_numbers(payload: AddInput) -> int:
    return payload.a + payload.b


class DoubleInput(BaseModel):
    value: int


async def slow_double(payload: DoubleInput) -> int:
    return payload.value * 2


def _patch_provider_states(monkeypatch: pytest.MonkeyPatch, states: list[object]) -> None:
    iterator = iter(states)

    async def fake_generate_state(self, **_):  # type: ignore[no-untyped-def]
        return StateEnvelope(state=next(iterator))

    monkeypatch.setattr(OAuthCodexStateProvider, "generate_state", fake_generate_state)


def _patch_provider_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_state(self, **_):  # type: ignore[no-untyped-def]
        return StateEnvelope(
            state=ReasoningState(
                next_state=NextState.reasoning,
                reasoning="keep working",
                focus="iteration",
            )
        )

    monkeypatch.setattr(OAuthCodexStateProvider, "generate_state", fake_generate_state)


def test_agent_rejects_removed_init_parameters() -> None:
    with pytest.raises(TypeError):
        Agent(
            instructions="x",
            model="gpt-5.3-codex",
            tools=[add_numbers],
            max_steps=1,  # type: ignore[call-arg]
        )


@pytest.mark.asyncio
async def test_stream_emits_events_in_expected_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_states(
        monkeypatch,
        [
            ReasoningState(
                next_state=NextState.tool_call,
                reasoning="Need calculator",
                focus="compute",
            ),
            ToolCallState(
                next_state=NextState.reasoning,
                tool_calls=[{"name": "add_numbers", "arguments": {"a": 2, "b": 5}}],
            ),
            ReasoningState(
                next_state=NextState.response,
                reasoning="Now I can answer",
                focus="user response",
            ),
            ResponseState(
                next_state=NextState.finish,
                response="The sum is 7.",
                audience="user",
            ),
            FinishState(
                next_state=NextState.finish,
                final_output="7",
                completion_reason="done",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_task_stream("add 2 and 5")]

    assert any(isinstance(event, ReasoningEvent) for event in events)
    assert any(isinstance(event, ResponseEvent) for event in events)
    tool_events = [event for event in events if isinstance(event, ToolEvent)]
    assert [event.phase for event in tool_events] == ["start", "finish"]
    assert tool_events[-1].result is not None
    assert tool_events[-1].result.ok is True
    assert isinstance(events[-1], TaskFinishedEvent)
    assert events[-1].final_output == "7"


@pytest.mark.asyncio
async def test_stream_executes_multiple_tools_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_states(
        monkeypatch,
        [
            ReasoningState(
                next_state=NextState.tool_call,
                reasoning="Need two calls",
                focus="sequential",
            ),
            ToolCallState(
                next_state=NextState.reasoning,
                tool_calls=[
                    {"name": "slow_double", "arguments": {"value": 2}},
                    {"name": "slow_double", "arguments": {"value": 3}},
                ],
            ),
            ReasoningState(
                next_state=NextState.finish,
                reasoning="Tools done",
                focus="finalize",
            ),
            FinishState(
                next_state=NextState.finish,
                final_output="done",
                completion_reason="done",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow graph.",
        model="gpt-5.3-codex",
        tools=[slow_double],
    )

    events = [event async for event in agent.run_task_stream("run tools")]

    tool_events = [event for event in events if isinstance(event, ToolEvent)]
    assert [event.phase for event in tool_events] == ["start", "finish", "start", "finish"]
    for event in tool_events:
        if event.phase == "finish":
            assert event.result is not None
            assert event.result.ok is True


@pytest.mark.asyncio
async def test_invalid_transition_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_states(
        monkeypatch,
        [
            ReasoningState(
                next_state=NextState.tool_call,
                reasoning="Need tool",
                focus="math",
            ),
            ToolCallState(
                next_state=NextState.finish,
                tool_calls=[{"name": "add_numbers", "arguments": {"a": 1, "b": 2}}],
            ),
        ],
    )

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_task_stream("invalid transition")]
    assert isinstance(events[-1], TaskFailedEvent)
    assert events[-1].error_code == "invalid_transition"


@pytest.mark.asyncio
async def test_tool_errors_are_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_states(
        monkeypatch,
        [
            ReasoningState(
                next_state=NextState.tool_call,
                reasoning="Try bad tool calls",
                focus="validation",
            ),
            ToolCallState(
                next_state=NextState.reasoning,
                tool_calls=[{"name": "missing_tool", "arguments": {"a": 1}}],
            ),
            ReasoningState(
                next_state=NextState.finish,
                reasoning="done",
                focus="finish",
            ),
            FinishState(
                next_state=NextState.finish,
                final_output="done",
                completion_reason="done",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_task_stream("tool errors")]
    finished_tools = [
        event for event in events if isinstance(event, ToolEvent) and event.phase == "finish"
    ]
    assert len(finished_tools) == 1
    assert finished_tools[0].result is not None
    assert finished_tools[0].result.ok is False
    assert finished_tools[0].result.error == "tool not found: missing_tool"


@pytest.mark.asyncio
async def test_max_steps_without_response_emits_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_loop(monkeypatch)

    agent = Agent(
        instructions="Keep reasoning.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_task_stream("loop")]
    assert isinstance(events[-1], TaskFailedEvent)
    assert events[-1].step == 24
    assert events[-1].error_code == "max_steps_reached"


@pytest.mark.asyncio
async def test_max_steps_uses_last_response_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_states(
        monkeypatch,
        [
            ReasoningState(
                next_state=NextState.response,
                reasoning="Send intermediate response",
                focus="respond",
            ),
            ResponseState(
                next_state=NextState.reasoning,
                response="Working on it.",
                audience="user",
            ),
            *[
                ReasoningState(
                    next_state=NextState.reasoning,
                    reasoning="Still working",
                    focus="continue",
                )
                for _ in range(22)
            ],
        ],
    )

    agent = Agent(
        instructions="Keep reasoning.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_task_stream("loop")]
    assert isinstance(events[-1], TaskFinishedEvent)
    assert events[-1].step == 24
    assert events[-1].completion_reason == "max_steps_reached"
    assert events[-1].final_output == "Working on it."
