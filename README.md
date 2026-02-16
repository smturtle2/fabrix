# Fabrix

Fabrix is a graph-based agent framework built on top of `oauth-codex>=2.2.0`.

## Key Features

- Graph-based 4-state execution (`reasoning`, `tool_call`, `response`, `finish`)
- Structured state outputs via Pydantic
- Sequential tool execution
- Async streaming event API

## Installation

```bash
pip install fabrix-ai
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
        model="gpt-5.3-codex",
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

See `examples/minimal/quickstart.py`, `examples/advanced/data_workflow.py`, and
`examples/advanced/incident_response.py` for more.

## Public API

- `Agent(instructions, model="gpt-5.3-codex", tools=None)`
- `run_task_stream(task, context=None)`

Execution defaults are fixed internally:

- `max_steps=24`
- no per-tool timeout

## Tool Signature Rule

Fabrix accepts tools only in this shape:

```python
def tool(payload: BaseModel) -> Any: ...
```

Tool arguments must be a JSON object matching the payload model fields.

## Event Types

- `reasoning`
- `tool` (`start` / `finish`)
- `response`
- `task_finished`
- `task_failed`

`task_failed` includes:

- `error_code`
- `message`

## Notes

- Output schema enforces state-specific `next_state` transitions.
- `tool_call` items allow only `name` and `arguments`.
- `tool_call` arguments are strictified from each tool parameter schema.
