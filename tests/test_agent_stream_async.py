from __future__ import annotations

import asyncio

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
    ToolCallState,
)


class AddInput(BaseModel):
    a: int
    b: int


def add_numbers(payload: AddInput) -> int:
    return payload.a + payload.b


class DoubleInput(BaseModel):
    value: int


async def slow_double(payload: DoubleInput) -> int:
    await asyncio.sleep(0.01)
    return payload.value * 2


@pytest.mark.asyncio
async def test_stream_emits_events_in_expected_order(sequence_provider_cls: type) -> None:
    provider = sequence_provider_cls(
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
            {
                "state_type": "response",
                "next_state": "finish",
                "response": "The sum is 7.",
                "audience": "user",
            },
            FinishState(
                next_state=NextState.finish,
                final_output="7",
                completion_reason="done",
            ),
        ]
    )

    agent = Agent(
        instructions="Follow the graph.",
        tools=[add_numbers],
        max_steps=10,
        state_provider=provider,
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
async def test_stream_executes_multiple_tools(sequence_provider_cls: type) -> None:
    provider = sequence_provider_cls(
        [
            ReasoningState(
                next_state=NextState.tool_call,
                reasoning="Need two calls",
                focus="parallel",
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
        ]
    )

    agent = Agent(
        instructions="Follow graph.",
        tools=[slow_double],
        state_provider=provider,
        max_steps=10,
    )

    events = [event async for event in agent.run_task_stream("run async tools")]

    tool_events = [event for event in events if isinstance(event, ToolEvent)]
    assert [event.phase for event in tool_events] == ["start", "start", "finish", "finish"]
    for event in tool_events[2:]:
        assert event.result is not None
        assert event.result.ok is True


@pytest.mark.asyncio
async def test_invalid_transition_fails_fast(sequence_provider_cls: type) -> None:
    provider = sequence_provider_cls(
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
        ]
    )

    agent = Agent(
        instructions="Follow the graph.",
        tools=[add_numbers],
        max_steps=10,
        state_provider=provider,
    )

    events = [event async for event in agent.run_task_stream("invalid transition")]
    assert isinstance(events[-1], TaskFailedEvent)
    assert events[-1].error_code == "invalid_transition"


@pytest.mark.asyncio
async def test_tool_errors_are_emitted(sequence_provider_cls: type) -> None:
    provider = sequence_provider_cls(
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
        ]
    )

    agent = Agent(
        instructions="Follow the graph.",
        tools=[add_numbers],
        max_steps=10,
        state_provider=provider,
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
async def test_max_steps_terminates_with_completion_reason(loop_reasoning_provider_cls: type) -> None:
    agent = Agent(
        instructions="Keep reasoning.",
        tools=[add_numbers],
        max_steps=2,
        state_provider=loop_reasoning_provider_cls(),
    )

    events = [event async for event in agent.run_task_stream("loop")]
    assert isinstance(events[-1], TaskFinishedEvent)
    assert events[-1].completion_reason == "max_steps_reached"
