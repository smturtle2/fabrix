from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from fabrix import Agent
from fabrix.events import (
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
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
            "When done, return finish state with concise final_output."
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
            print("response:", event.response)
        elif isinstance(event, TaskFinishedEvent):
            print("completion reason:", event.completion_reason)
            print("final:", event.final_output)
        elif isinstance(event, TaskFailedEvent):
            print("failed:", event.error_code, event.message)


if __name__ == "__main__":
    asyncio.run(main())
