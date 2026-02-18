from __future__ import annotations

import pytest
from pydantic import BaseModel

from fabrix.agent import _DEFAULT_MAX_STEPS, Agent
from fabrix.errors import RetryableLLMOutputError
from fabrix.events import ReasoningEvent, ResponseEvent, TaskFailedEvent, ToolEvent
from fabrix.graph.state import (
    NextState,
    ReasoningState,
    ResponseState,
    StateEnvelope,
    ToolCallState,
)
from fabrix.llm.oauth_codex import OAuthCodexStateProvider
from fabrix.messages import ImageMessage, TextMessage
from fabrix.tools import ToolOutput


class AddInput(BaseModel):
    a: int
    b: int


def add_numbers(payload: AddInput) -> ToolOutput:
    return ToolOutput.json({"sum": payload.a + payload.b})


class DoubleInput(BaseModel):
    value: int


async def slow_double(payload: DoubleInput) -> ToolOutput:
    return ToolOutput.json({"value": payload.value * 2})


def _text_message(text: str) -> TextMessage:
    return TextMessage(role="user", text=text)


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


def test_agent_accepts_instruction_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_provider_init(self: OAuthCodexStateProvider, **kwargs: object) -> None:
        captured["instructions"] = kwargs["instructions"]

    def fake_validate_tool_schemas(
        self: OAuthCodexStateProvider, tool_schemas: list[dict[str, object]]
    ) -> None:
        del self, tool_schemas

    monkeypatch.setattr(OAuthCodexStateProvider, "__init__", fake_provider_init)
    monkeypatch.setattr(OAuthCodexStateProvider, "validate_tool_schemas", fake_validate_tool_schemas)

    Agent(
        instructions=lambda: "Use strict policies.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    instruction_source = captured["instructions"]
    assert callable(instruction_source)
    assert instruction_source() == "Use strict policies."


def test_agent_rejects_non_string_non_callable_instructions() -> None:
    with pytest.raises(TypeError, match="instructions must be a string or a callable returning a string"):
        Agent(
            instructions=123,  # type: ignore[arg-type]
            model="gpt-5.3-codex",
            tools=[add_numbers],
        )


@pytest.mark.asyncio
async def test_stream_rejects_when_messages_missing() -> None:
    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    with pytest.raises(ValueError, match="messages must be a non-empty list of TextMessage/ImageMessage objects"):
        async for _event in agent.run_stream(messages=[]):
            pass


@pytest.mark.asyncio
async def test_stream_rejects_dict_messages() -> None:
    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    with pytest.raises(TypeError, match="messages must be a list of TextMessage/ImageMessage objects"):
        async for _event in agent.run_stream(messages=[{"role": "user", "content": "x"}]):  # type: ignore[list-item]
            pass


@pytest.mark.asyncio
async def test_stream_accepts_image_only_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_states(
        monkeypatch,
        [
            ReasoningState(
                next_state=NextState.response,
                reasoning="Analyze the image first",
                focus="visual inspection",
            ),
            ResponseState(
                next_state=None,
                response="image analyzed",
                audience="user",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    messages = [ImageMessage(role="user", image="https://example.com/input.png")]
    events = [event async for event in agent.run_stream(messages=messages)]

    assert isinstance(events[-1], ResponseEvent)
    assert events[-1].response == "image analyzed"


@pytest.mark.asyncio
async def test_stream_allows_empty_response_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_states(
        monkeypatch,
        [
            ReasoningState(
                next_state=NextState.response,
                reasoning="No content needed",
                focus="emit empty response",
            ),
            ResponseState(
                next_state=None,
                response=None,
                parts=None,
                audience="user",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("empty response")])]
    assert isinstance(events[-1], ResponseEvent)
    assert events[-1].response is None
    assert events[-1].parts is None


@pytest.mark.asyncio
async def test_stream_preserves_response_parts_in_history(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_state(self, **kwargs):  # type: ignore[no-untyped-def]
        step = kwargs["step"]
        if step == 1:
            return StateEnvelope(
                state=ReasoningState(
                    next_state=NextState.response,
                    reasoning="Respond with an image part",
                    focus="emit multimodal response",
                )
            )
        if step == 2:
            return StateEnvelope(
                state=ResponseState(
                    next_state=NextState.reasoning,
                    response=None,
                    parts=[{"type": "image", "image_url": "https://example.com/result.png"}],
                    audience="user",
                )
            )
        if step == 3:
            history = kwargs["history"]
            response_items = [item for item in history if item.get("kind") == "response"]
            assert response_items
            assert response_items[-1]["response"] is None
            assert response_items[-1]["parts"] == [
                {"type": "image", "image_url": "https://example.com/result.png", "caption": None}
            ]
            return StateEnvelope(
                state=ReasoningState(
                    next_state=NextState.response,
                    reasoning="Wrap up after image response",
                    focus="final answer",
                )
            )
        if step == 4:
            return StateEnvelope(
                state=ResponseState(
                    next_state=None,
                    response="done",
                    audience="user",
                )
            )

        raise AssertionError(f"unexpected step: {step}")

    monkeypatch.setattr(OAuthCodexStateProvider, "generate_state", fake_generate_state)

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("response parts")])]
    response_events = [event for event in events if isinstance(event, ResponseEvent)]
    assert len(response_events) == 2
    assert response_events[0].response is None
    assert response_events[0].parts is not None
    assert response_events[0].parts[0].type == "image"
    assert response_events[-1].response == "done"


@pytest.mark.asyncio
async def test_stream_forwards_messages_to_state_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_generate_state(self, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        if kwargs["current_state"] == NextState.reasoning:
            return StateEnvelope(
                state=ReasoningState(
                    next_state=NextState.response,
                    reasoning="ready",
                    focus="respond now",
                )
            )
        return StateEnvelope(
            state=ResponseState(
                next_state=None,
                response="done",
                audience="user",
            )
        )

    monkeypatch.setattr(OAuthCodexStateProvider, "generate_state", fake_generate_state)

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    messages = [_text_message("hello")]
    events = [event async for event in agent.run_stream(messages=messages)]

    assert isinstance(events[-1], ResponseEvent)
    assert calls
    assert calls[0]["messages"] == messages


@pytest.mark.asyncio
async def test_response_none_terminates_without_extra_state_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def fake_generate_state(self, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if kwargs["current_state"] == NextState.reasoning:
            return StateEnvelope(
                state=ReasoningState(
                    next_state=NextState.response,
                    reasoning="answering now",
                    focus="final response",
                )
            )
        if kwargs["current_state"] == NextState.response:
            return StateEnvelope(
                state=ResponseState(
                    next_state=None,
                    response="final answer",
                    audience="user",
                )
            )
        raise AssertionError("generate_state must not be called after terminal response")

    monkeypatch.setattr(OAuthCodexStateProvider, "generate_state", fake_generate_state)

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("end now")])]

    assert call_count == 2
    assert isinstance(events[-1], ResponseEvent)
    assert events[-1].response == "final answer"


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
                next_state=None,
                response="The sum is 7.",
                audience="user",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("add 2 and 5")])]

    assert any(isinstance(event, ReasoningEvent) for event in events)
    assert any(isinstance(event, ResponseEvent) for event in events)
    tool_events = [event for event in events if isinstance(event, ToolEvent)]
    assert [event.phase for event in tool_events] == ["start", "finish"]
    assert tool_events[-1].result is not None
    assert tool_events[-1].result.ok is True
    assert isinstance(tool_events[-1].result.output, ToolOutput)
    assert isinstance(events[-1], ResponseEvent)
    assert events[-1].response == "The sum is 7."


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
                next_state=NextState.tool_call,
                tool_calls=[{"name": "slow_double", "arguments": {"value": 2}}],
            ),
            ToolCallState(
                next_state=NextState.response,
                tool_calls=[{"name": "slow_double", "arguments": {"value": 3}}],
            ),
            ResponseState(
                next_state=None,
                response="done",
                audience="user",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow graph.",
        model="gpt-5.3-codex",
        tools=[slow_double],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("run tools")])]

    tool_events = [event for event in events if isinstance(event, ToolEvent)]
    assert [event.phase for event in tool_events] == ["start", "finish", "start", "finish"]
    for event in tool_events:
        if event.phase == "finish":
            assert event.result is not None
            assert event.result.ok is True
    assert isinstance(events[-1], ResponseEvent)


@pytest.mark.asyncio
async def test_all_to_all_state_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_states(
        monkeypatch,
        [
            ReasoningState(
                next_state=NextState.tool_call,
                reasoning="Route to tool call",
                focus="transition 1",
            ),
            ToolCallState(
                next_state=NextState.response,
                tool_calls=[{"name": "add_numbers", "arguments": {"a": 1, "b": 1}}],
            ),
            ResponseState(
                next_state=NextState.reasoning,
                response="intermediate",
                audience="user",
            ),
            ReasoningState(
                next_state=NextState.response,
                reasoning="switch to response",
                focus="transition 4",
            ),
            ResponseState(
                next_state=NextState.tool_call,
                response="one more tool",
                audience="user",
            ),
            ToolCallState(
                next_state=NextState.response,
                tool_calls=[{"name": "add_numbers", "arguments": {"a": 2, "b": 3}}],
            ),
            ResponseState(
                next_state=None,
                response="final",
                audience="user",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("all transitions")])]
    assert isinstance(events[-1], ResponseEvent)
    assert events[-1].response == "final"


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
                next_state=NextState.response,
                tool_calls=[{"name": "missing_tool", "arguments": {"a": 1}}],
            ),
            ResponseState(
                next_state=None,
                response="done",
                audience="user",
            ),
        ],
    )

    agent = Agent(
        instructions="Follow the graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("tool errors")])]
    finished_tools = [
        event for event in events if isinstance(event, ToolEvent) and event.phase == "finish"
    ]
    assert len(finished_tools) == 1
    assert finished_tools[0].result is not None
    assert finished_tools[0].result.ok is False
    assert finished_tools[0].result.error == "tool not found: missing_tool"


@pytest.mark.asyncio
async def test_max_steps_without_response_emits_no_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provider_loop(monkeypatch)

    agent = Agent(
        instructions="Keep reasoning.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("loop")])]
    assert len(events) == _DEFAULT_MAX_STEPS
    assert isinstance(events[-1], ReasoningEvent)
    assert not any(isinstance(event, TaskFailedEvent) for event in events)


@pytest.mark.asyncio
async def test_retryable_llm_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_generate_state(self, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if kwargs["current_state"] == NextState.reasoning and call_count == 1:
            raise RetryableLLMOutputError("retry once")
        if kwargs["current_state"] == NextState.reasoning:
            return StateEnvelope(
                state=ReasoningState(
                    next_state=NextState.response,
                    reasoning="ready",
                    focus="answer",
                )
            )
        return StateEnvelope(
            state=ResponseState(
                next_state=None,
                response="done",
                audience="user",
            )
        )

    monkeypatch.setattr(OAuthCodexStateProvider, "generate_state", fake_generate_state)

    agent = Agent(
        instructions="Follow graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("retry")])]
    assert isinstance(events[-1], ResponseEvent)
    assert events[-1].response == "done"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retryable_llm_errors_fail_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_generate_state(self, **_):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        raise RetryableLLMOutputError("always bad")

    monkeypatch.setattr(OAuthCodexStateProvider, "generate_state", fake_generate_state)

    agent = Agent(
        instructions="Follow graph.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    events = [event async for event in agent.run_stream(messages=[_text_message("retry fail")])]
    assert isinstance(events[-1], TaskFailedEvent)
    assert events[-1].error_code == "llm_error"
    assert "retry attempts exhausted" in events[-1].message
    assert call_count == 3
