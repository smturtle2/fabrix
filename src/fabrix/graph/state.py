from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class NextState(StrEnum):
    reasoning = "reasoning"
    tool_call = "tool_call"
    response = "response"
    finish = "finish"


class BaseState(BaseModel):
    state_type: str
    next_state: NextState


class PlannedToolCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    why: str | None = Field(default=None)
    call_id: str | None = Field(default=None)


class ReasoningState(BaseState):
    state_type: Literal["reasoning"] = "reasoning"
    reasoning: str = Field(min_length=1)
    focus: str = Field(min_length=1)


class ToolCallState(BaseState):
    state_type: Literal["tool_call"] = "tool_call"
    tool_calls: list[PlannedToolCall] = Field(min_length=1)


class ResponseState(BaseState):
    state_type: Literal["response"] = "response"
    response: str = Field(min_length=1)
    audience: Literal["user", "system"] = "user"


class FinishState(BaseState):
    state_type: Literal["finish"] = "finish"
    final_output: str = Field(min_length=1)
    completion_reason: str = Field(min_length=1)


AgentState = Annotated[
    ReasoningState | ToolCallState | ResponseState | FinishState,
    Field(discriminator="state_type"),
]


class StateEnvelope(BaseModel):
    state: AgentState
