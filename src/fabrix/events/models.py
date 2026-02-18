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
    event_type: str
    step: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=_utc_now)


class ReasoningEvent(BaseEvent):
    event_type: Literal["reasoning"] = "reasoning"
    reasoning: str
    focus: str
    next_state: NextState


class ToolEvent(BaseEvent):
    event_type: Literal["tool"] = "tool"
    phase: Literal["start", "finish"]
    tool_name: str
    call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: ToolExecutionResult | None = None


class ResponseEvent(BaseEvent):
    event_type: Literal["response"] = "response"
    response: str | None = None
    parts: list[ToolPart] | None = None


class TaskFailedEvent(BaseEvent):
    event_type: Literal["task_failed"] = "task_failed"
    error_code: str
    message: str


AgentEvent = Annotated[
    (
        ReasoningEvent
        | ToolEvent
        | ResponseEvent
        | TaskFailedEvent
    ),
    Field(discriminator="event_type"),
]
