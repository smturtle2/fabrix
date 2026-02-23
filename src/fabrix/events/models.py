"""Typed event models emitted by ``Agent.run_stream``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from fabrix.graph.state import NextState
from fabrix.tools.output import ToolPart
from fabrix.tools.runtime import ToolExecutionResult


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BaseEvent(BaseModel):
    """Base schema shared by all stream events.

    Attributes:
        event_type: Discriminator used for event union parsing.
        step: 1-based execution step index.
        timestamp: UTC timestamp generated when the event object is created.
    """

    event_type: str
    step: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=_utc_now)


class ReasoningEvent(BaseEvent):
    """Reasoning trace event for a single graph step.

    This event exposes a short decision summary and the next graph state chosen
    by the model.
    """

    event_type: Literal["reasoning"] = "reasoning"
    reasoning: str
    focus: str
    next_state: NextState


class ToolEvent(BaseEvent):
    """Tool execution lifecycle event.

    ``phase="start"`` is emitted immediately before tool invocation and does
    not include a result. ``phase="finish"`` is emitted after invocation and
    includes ``result`` with success or error details.
    """

    event_type: Literal["tool"] = "tool"
    phase: Literal["start", "finish"]
    tool_name: str
    call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: ToolExecutionResult | None = None


class ResponseEvent(BaseEvent):
    """User-facing response event.

    A response can be plain text (``response``), structured parts (``parts``),
    or both. In intermediate steps, either field may be ``None``.
    """

    event_type: Literal["response"] = "response"
    response: str | None = None
    parts: list[ToolPart] | None = None


class TaskFailedEvent(BaseEvent):
    """Terminal failure event for unrecoverable execution errors."""

    event_type: Literal["task_failed"] = "task_failed"
    error_code: str
    message: str


# Discriminated union of all event payloads produced by the runtime stream.
AgentEvent = Annotated[
    (
        ReasoningEvent
        | ToolEvent
        | ResponseEvent
        | TaskFailedEvent
    ),
    Field(discriminator="event_type"),
]
