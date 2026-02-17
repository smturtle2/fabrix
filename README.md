# Fabrix

Language: English | [한국어](README.ko.md)  
API Guides: [English](docs/api.md) | [한국어](docs/api.ko.md)

## Overview

Fabrix is a graph-based agent framework built on top of `oauth-codex>=2.3.0`.
It provides a structured execution graph with streaming events for tool-driven workflows.

## Key Features

- Graph-based 4-state execution: `reasoning`, `tool_call`, `response`, `finish`
- Structured state outputs powered by Pydantic models
- Sequential tool execution with strict payload validation
- Async streaming event API for step-by-step observability
- Multimodal input with explicit message models: `TextMessage`, `ImageMessage`

## Installation

```bash
pip install fabrix-ai
```

## Quickstart

```python
import asyncio

from pydantic import BaseModel

from fabrix import Agent
from fabrix.events import (
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
    ToolEvent,
)
from fabrix.messages import TextMessage


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

    messages = [TextMessage(text="Use add_numbers to compute 3 + 9")]
    async for event in agent.run_stream(messages=messages):
        print(f"[step={event.step}] {event.event_type}")

        if isinstance(event, ReasoningEvent):
            print("reasoning:", event.reasoning)
            print("focus:", event.focus)
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


asyncio.run(main())
```

## Message Models

Fabrix input is now `list[TextMessage | ImageMessage]`.

- `TextMessage(role: str = "user", text: str)`
- `ImageMessage(role: str = "user", image: str | Path | bytes, text: str | None = None)`
- Unknown message fields are rejected at construction time.

`ImageMessage.image` accepts:

- remote URL (`https://...`)
- local path (`Path` or string path)
- raw bytes (`bytes`), encoded to a data URL internally

## Multimodal Input

```python
from fabrix.messages import ImageMessage, TextMessage

messages = [
    TextMessage(text="Describe this screenshot"),
    ImageMessage(image="https://example.com/screenshot.png"),
    TextMessage(text="Focus on errors"),
]

async for event in agent.run_stream(messages=messages):
    ...
```

## Tool Contract

Fabrix accepts tools in this shape:

```python
def tool(payload: BaseModel) -> Any: ...
```

- The tool must accept exactly one parameter.
- The parameter type must be a Pydantic `BaseModel`.
- Runtime arguments must be a JSON object matching payload fields.
- Extra argument keys are rejected.
- Both sync and async tools are supported.

## Event Stream

`run_stream(...)` yields these event types:

- `reasoning`
- `tool` (`phase="start"` / `phase="finish"`)
- `response`
- `task_finished`
- `task_failed`

`reasoning` is a step-level decision trace / plan summary, not raw internal chain-of-thought.

## Migration (Breaking)

`run_task_stream(task, images, context)` has been removed.

- Before: `agent.run_task_stream(task=..., images=..., context=...)`
- After: `agent.run_stream(messages=[...])`

Mapping:

- `task` text -> `TextMessage(text="...")`
- `images` -> `ImageMessage(image="..." | Path(...) | b"...")`
- `context` -> include serialized context in `TextMessage.text`

## Documentation

- API usage guide (English): [`docs/api.md`](docs/api.md)
- API 사용 가이드 (한국어): [`docs/api.ko.md`](docs/api.ko.md)
- Korean README: [`README.ko.md`](README.ko.md)

## Examples

- Minimal quickstart: [`examples/minimal/quickstart.py`](examples/minimal/quickstart.py)
- Multimodal vision: [`examples/minimal/multimodal.py`](examples/minimal/multimodal.py)
- Data workflow: [`examples/advanced/data_workflow.py`](examples/advanced/data_workflow.py)
- Incident response workflow: [`examples/advanced/incident_response.py`](examples/advanced/incident_response.py)

## Notes

- Public runtime entry point is `fabrix.Agent`.
- Execution defaults are fixed internally: `max_steps=128` and no public per-tool timeout option.
- If `max_steps` is reached and at least one `response` was emitted, the stream ends with `task_finished` (`completion_reason="max_steps_reached"`) using the last response.
- If `max_steps` is reached before any response/final output, the stream ends with `task_failed` (`error_code="max_steps_reached"`).
