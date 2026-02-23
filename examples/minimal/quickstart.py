from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel, Field

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common import run_case
from fabrix import Agent
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
        state_models={"reasoning": "gpt-5.3-codex"},
        tools=[add_numbers],
    )

    messages = [TextMessage(role="user", text="Use add_numbers to compute 38 + 4 and explain briefly.")]
    await run_case(agent, messages=messages)


if __name__ == "__main__":
    asyncio.run(main())
