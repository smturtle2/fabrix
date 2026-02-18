# Fabrix API Usage Guide

Language: English | [한국어](api.ko.md)  
Home: [README](../README.md) | [README.ko.md](../README.ko.md)

## Purpose & Scope

This guide documents the public usage surface of Fabrix.
It covers agent construction, `run_stream`, message models, and stream events.

## Requirements

- Python `>=3.12`
- Package: `pip install fabrix-ai`
- Valid `oauth-codex` authentication
- Async runtime (examples use `asyncio`)

## Public Imports

```python
from fabrix import Agent
from fabrix.events import (
    AgentEvent,
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    ToolEvent,
)
from fabrix.messages import ImageMessage, TextMessage
from fabrix.tools import ToolImagePart, ToolJSONPart, ToolOutput, ToolTextPart
```

## Agent Construction

```python
Agent(
    *,
    instructions: str,
    model: str = "gpt-5.3-codex",
    tools: list[Callable[..., Any]] | None = None,
)
```

Notes:

- Internal execution default: `max_steps=128`
- No public per-tool timeout constructor option
- Incompatible tool schemas fail fast at setup

## Message Models

Fabrix runtime input is a list of message model objects.

```python
class TextMessage(BaseModel):
    role: str = "user"
    text: str

class ImageMessage(BaseModel):
    role: str = "user"
    image: str | Path | bytes
    text: str | None = None
```

Rules:

- `run_stream` accepts only `TextMessage` / `ImageMessage` instances.
- Unknown fields are rejected at construction time.
- `ImageMessage.image` supports URL/path/bytes.
- `bytes` and local paths are converted to data URLs before model call.

## Running Streams

```python
run_stream(
    *,
    messages: list[TextMessage | ImageMessage],
) -> AsyncIterator[AgentEvent]
```

Validation:

- Empty list -> `ValueError`
- Non-message objects -> `TypeError`

Text-only example:

```python
messages = [TextMessage(text="Use add_numbers to compute 3 + 9")]
async for event in agent.run_stream(messages=messages):
    ...
```

Multimodal example:

```python
messages = [
    TextMessage(text="Describe this image"),
    ImageMessage(image="https://example.com/cat.png"),
    TextMessage(text="Focus on visible warnings"),
]
async for event in agent.run_stream(messages=messages):
    ...
```

## Tool Definition Rules

Accepted tool shape:

```python
def tool(payload: BaseModel) -> ToolOutput: ...
```

Runtime rules:

- Exactly one parameter
- Parameter type must be Pydantic `BaseModel`
- Return type must be `ToolOutput`
- Tool arguments must be JSON objects with matching field names
- Extra keys are rejected

`ToolOutput` supports structured multimodal parts:

- `ToolTextPart(type="text", text="...")`
- `ToolImagePart(type="image", image_url="...", caption=None)`
- `ToolJSONPart(type="json", data={...})`
- `ToolImagePart.image_url` accepts URL/path/bytes inputs and normalizes local files/bytes to data URLs.

## Event Reference

Common fields:

- `event_type: str`
- `step: int`
- `timestamp: datetime` (UTC)

Event-specific fields:

| event_type | Model | Key fields |
| --- | --- | --- |
| `reasoning` | `ReasoningEvent` | `reasoning`, `focus`, `next_state` |
| `tool` | `ToolEvent` | `phase`, `tool_name`, `call_id`, `arguments`, `result` |
| `response` | `ResponseEvent` | `response`, `parts` |
| `task_failed` | `TaskFailedEvent` | `error_code`, `message` |

`ResponseEvent.response` and `ResponseEvent.parts` are both optional; both can be `None`.

Completion rule:

- A successful run ends immediately after a `response` event emitted from `response` state with `next_state=null`.

## Failure Handling

Terminal failures are emitted as `TaskFailedEvent`.
Current error codes include:

- `llm_error`
- `invalid_state_type`

Tool failures are non-terminal and reported in `ToolEvent(phase="finish")` with `result.ok == False`.
Successful tool results expose typed payloads in `result.output: ToolOutput`.

## Migration (Breaking)

Removed API:

- `run_task_stream(task, images, context)`

New API:

- `run_stream(messages=[...])`

Input mapping:

- `task` -> `TextMessage(text="...")`
- `images` -> `ImageMessage(image="..." | Path(...) | b"...")`
- `context` -> include serialized context text in `TextMessage.text`

Tool return mapping:

- Before: tool returned scalar/string/dict-like outputs
- After: tool must return `ToolOutput` via `ToolOutput.text(...)`, `ToolOutput.json(...)`, or `ToolOutput.image(...)`

## Troubleshooting

- `ValueError: messages must be a non-empty list of TextMessage/ImageMessage objects`:
  pass at least one message model object.
- `TypeError: messages must be a list of TextMessage/ImageMessage objects`:
  do not pass dicts; instantiate message models.
- Stream ends without a terminal event:
  ensure the final `response` state sets `next_state` to `null`.
- `task_failed` with `llm_error`:
  verify auth/model setup and tool-schema compatibility.
