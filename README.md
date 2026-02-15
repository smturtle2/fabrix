# Fabrix

Fabrix is a graph-based agent framework built on top of `oauth-codex>=2.2.0`.

## Key Features

- Graph-based state transitions
- Structured state outputs via Pydantic
- Tool execution with strict, tool-specific argument schemas
- Tool execution with Pydantic parameter validation
- Async-only, streaming-first API

## Installation

```bash
pip install fabrix
```

## Quickstart

```python
import asyncio

from pydantic import BaseModel

from fabrix import Agent
from fabrix.events import ReasoningEvent, TaskFinishedEvent, ToolEvent


class AddInput(BaseModel):
    a: int
    b: int


def add_numbers(payload: AddInput) -> int:
    return payload.a + payload.b


async def main() -> None:
    agent = Agent(
        instructions="You are a precise assistant.",
        tools=[add_numbers],
    )

    async for event in agent.run_task_stream("Use add_numbers to compute 3 + 9"):
        print(f"[step={event.step}] {event.event_type}")
        if isinstance(event, ReasoningEvent):
            print("reasoning:", event.reasoning)
        elif isinstance(event, ToolEvent):
            if event.phase == "start":
                print("tool call:", event.tool_name, event.arguments)
            elif event.result is not None:
                print("tool result:", event.result.model_dump())
        elif isinstance(event, TaskFinishedEvent):
            print("final:", event.final_output)


asyncio.run(main())
```

See `examples/minimal/quickstart.py` and `examples/advanced/data_workflow.py` for more.

## Notes

- Structured state output is compiled to a Structured Outputs-compatible schema subset.
- Tool-call items use strict schemas (`anyOf`, no `oneOf`/`const`) for provider compatibility.
- Every object node is normalized to explicit `properties` + `required` consistency.
- Map-like `dict[...]` shapes may be narrowed to empty-object constraints for strict compatibility.
- Define tool parameters with explicit fields (Pydantic models recommended) for reliable validation.
