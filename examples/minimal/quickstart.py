from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from fabrix import Agent
from fabrix.events import (
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    ToolEvent,
)
from fabrix.messages import TextMessage
from fabrix.tools import ToolOutput


class AddInput(BaseModel):
    a: int = Field(ge=-10_000, le=10_000)
    b: int = Field(ge=-10_000, le=10_000)


def add_numbers(payload: AddInput) -> ToolOutput:
    """Add two integers and return the sum."""
    return ToolOutput.json({"sum": payload.a + payload.b})


async def main() -> None:
    agent = Agent(
        instructions=(
            "Solve the task accurately. Prefer tool usage for arithmetic tasks. "
            "When done, return response state with next_state set to null."
        ),
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    messages = [TextMessage(role="user", text="Use add_numbers to compute 38 + 4 and explain briefly.")]

    async for event in agent.run_stream(messages=messages):
        print(f"[step={event.step}] {event.event_type}")

        if isinstance(event, ReasoningEvent):
            print("reasoning:", event.reasoning)
            print("focus:", event.focus)
            print("next:", event.next_state)
        elif isinstance(event, ToolEvent):
            if event.phase == "start":
                print("tool call:", event.tool_name, event.arguments)
            elif event.result is not None:
                print("tool result:", event.result.model_dump())
        elif isinstance(event, ResponseEvent):
            if event.response is not None:
                print("response:", event.response)
            if event.parts is not None:
                print("parts:", [part.model_dump(mode="json") for part in event.parts])
            if event.response is None and event.parts is None:
                print("response: <empty>")
        elif isinstance(event, TaskFailedEvent):
            print("failed:", event.error_code, event.message)


if __name__ == "__main__":
    asyncio.run(main())
