from __future__ import annotations

from collections.abc import Sequence

from fabrix import Agent
from fabrix.events import (
    AgentEvent,
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    ToolEvent,
)
from fabrix.messages import ImageMessage, TextMessage


def print_event(event: AgentEvent) -> None:
    print(f"[step={event.step}] {event.event_type}")

    if isinstance(event, ReasoningEvent):
        print("reasoning:", event.reasoning)
        print("focus:", event.focus)
        print("next:", event.next_state)
        return

    if isinstance(event, ToolEvent):
        if event.phase == "start":
            print("tool call:", event.tool_name, event.arguments)
            return
        if event.result is not None:
            print("tool result:", event.result.model_dump())
        return

    if isinstance(event, ResponseEvent):
        if event.response is not None:
            print("response:", event.response)
        if event.parts is not None:
            print("parts:", [part.model_dump(mode="json") for part in event.parts])
        if event.response is None and event.parts is None:
            print("response: <empty>")
        return

    if isinstance(event, TaskFailedEvent):
        print("failed:", event.error_code, event.message)


async def run_case(
    agent: Agent,
    *,
    messages: Sequence[TextMessage | ImageMessage],
    title: str | None = None,
) -> None:
    if title:
        print(f"\n=== {title} ===")

    async for event in agent.run_stream(messages=list(messages)):
        print_event(event)
