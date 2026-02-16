# Fabrix API Usage Guide

Language: English | [한국어](api.ko.md)  
Home: [README](../README.md) | [README.ko.md](../README.ko.md)

## Purpose & Scope

This guide documents the public usage surface of Fabrix.
It covers how to construct `Agent`, run tasks with `run_task_stream`, define tools, and handle streamed events.
Internal implementation details are out of scope except where they affect observable behavior.

## Requirements

- Python `>=3.12`
- Package: `pip install fabrix-ai`
- Valid `oauth-codex` authentication for model calls in real runs
- Async runtime (examples use `asyncio`)

## Public Imports

```python
from fabrix import Agent
from fabrix.events import (
    AgentEvent,
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
    ToolEvent,
)
```

Public runtime entry point:

- `fabrix.Agent`

Public event models for stream consumers:

- `fabrix.events.AgentEvent`
- `fabrix.events.ReasoningEvent`
- `fabrix.events.ToolEvent`
- `fabrix.events.ResponseEvent`
- `fabrix.events.TaskFinishedEvent`
- `fabrix.events.TaskFailedEvent`

## Agent Construction

Constructor signature:

```python
Agent(
    *,
    instructions: str,
    model: str = "gpt-5.3-codex",
    tools: list[Callable[..., Any]] | None = None,
)
```

Parameter behavior:

- `instructions`: required developer instruction text.
- `model`: model name passed to the provider. Default is `"gpt-5.3-codex"`.
- `tools`: optional list of tool callables validated at construction time.

Notes:

- Execution defaults are fixed internally: `max_steps=24`.
- There is no public constructor parameter for per-tool timeout.
- Incompatible tool schemas fail fast during agent setup.

## Tool Definition Rules

Accepted tool shape:

```python
def tool(payload: BaseModel) -> Any: ...
```

Rules enforced by runtime:

- Exactly one parameter is required.
- The parameter must be typed as a Pydantic `BaseModel` subclass.
- Positional-only parameters are rejected.
- Variadic `*args` and `**kwargs` are rejected.
- The parameter must not have a default value.
- Sync and async callables are both supported.

Runtime argument validation:

- Tool arguments must be a JSON object.
- Argument keys must match payload model fields.
- Extra keys return an error such as `"unexpected tool arguments: extra"`.

## Running Tasks

Method signature:

```python
run_task_stream(
    task: str,
    *,
    context: dict[str, Any] | None = None,
) -> AsyncIterator[AgentEvent]
```

Usage notes:

- `task` is the user task string.
- `context` is optional structured data exposed to the model each step.
- The stream yields events until a terminal event is emitted.

Example:

```python
async for event in agent.run_task_stream(
    "Analyze context.raw_rows and return a summary",
    context={"raw_rows": [{"category": "A", "value": 3.2}]},
):
    ...
```

## Streaming Event Handling

Recommended dispatch pattern:

```python
async for event in agent.run_task_stream("Use add_numbers to compute 3 + 9"):
    if isinstance(event, ReasoningEvent):
        print(event.reasoning, event.focus, event.next_state)
    elif isinstance(event, ToolEvent):
        if event.phase == "start":
            print("calling", event.tool_name, event.arguments)
        elif event.result is not None:
            print("tool ok:", event.result.ok, "error:", event.result.error)
    elif isinstance(event, ResponseEvent):
        print(event.response)
    elif isinstance(event, TaskFinishedEvent):
        print(event.final_output, event.completion_reason)
    elif isinstance(event, TaskFailedEvent):
        print(event.error_code, event.message)
```

## Event Reference

Common fields for all events:

- `event_type: str`
- `step: int` (current executor uses 1-based steps)
- `timestamp: datetime` (UTC)

Event-specific fields:

| event_type | Model | Key fields |
| --- | --- | --- |
| `reasoning` | `ReasoningEvent` | `reasoning`, `focus`, `next_state` |
| `tool` | `ToolEvent` | `phase`, `tool_name`, `call_id`, `arguments`, `result` |
| `response` | `ResponseEvent` | `response` |
| `task_finished` | `TaskFinishedEvent` | `final_output`, `completion_reason` |
| `task_failed` | `TaskFailedEvent` | `error_code`, `message` |

`ToolEvent.result` is populated on `phase="finish"` and includes:

- `ok: bool`
- `output: Any | None`
- `error: str | None`
- `latency_ms: float`

## Failure Handling

Terminal failures are emitted as `TaskFailedEvent`.
Current `error_code` values include:

- `llm_error`: model/provider failed to produce a valid structured state.
- `invalid_state_type`: model returned a state type different from expected current state.
- `invalid_transition`: state transition violated graph rules.
- `max_steps_reached`: no response/final output was produced by step 24.

Tool call failures are non-terminal by themselves and are emitted on `ToolEvent(phase="finish")` with `result.ok == False`.
Common tool errors include:

- `tool not found: <name>`
- `tool arguments must be a JSON object`
- `unexpected tool arguments: <extra_key>`

## Behavioral Guarantees

- Each run starts from `reasoning` state.
- Tool calls inside one `tool_call` state execute sequentially in listed order.
- Invalid state transitions terminate the run with `task_failed`.
- When `finish` state is produced, `task_finished` is emitted and the stream ends.
- If step limit is reached and at least one `response` exists, stream ends with `task_finished` using last response and `completion_reason="max_steps_reached"`.
- If step limit is reached without response/final output, stream ends with `task_failed` and `error_code="max_steps_reached"`.

## Complete Example

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

## Troubleshooting

- `TypeError` about tool signature:
  Define tools with exactly one Pydantic payload parameter.
- `unexpected tool arguments: ...`:
  Ensure model-generated arguments exactly match payload model field names.
- `tool arguments must be a JSON object`:
  Ensure tool call arguments are object-shaped JSON.
- `task_failed` with `invalid_transition`:
  Tighten instructions so the model follows allowed graph transitions.
- `task_failed` with `llm_error`:
  Check model/auth configuration and whether tool schemas are compatible with strict JSON schema conversion.
