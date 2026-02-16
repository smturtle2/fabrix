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
    state_type: str = Field(
        description="Current node type in the execution graph."
    )
    next_state: NextState = Field(
        description="Next node that the executor should transition to after this state."
    )


class PlannedToolCall(BaseModel):
    name: str = Field(
        min_length=1,
        description="Registered tool name to invoke.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON object payload passed to the selected tool.",
    )
    why: str | None = Field(
        default=None,
        description="Optional rationale for why this tool call is needed now.",
    )
    call_id: str | None = Field(
        default=None,
        description="Optional stable identifier for correlating tool start/finish events.",
    )


class ReasoningState(BaseState):
    state_type: Literal["reasoning"] = Field(
        default="reasoning",
        description="Reasoning node: capture internal plan and pick the next state.",
    )
    reasoning: str = Field(
        min_length=1,
        description="Concise chain-of-thought style summary for this step.",
    )
    focus: str = Field(
        min_length=1,
        description="Short statement of what this step is trying to achieve.",
    )


class ToolCallState(BaseState):
    state_type: Literal["tool_call"] = Field(
        default="tool_call",
        description="Tool-call node: execute one or more external tool invocations.",
    )
    tool_calls: list[PlannedToolCall] = Field(
        min_length=1,
        description="Ordered tool calls to execute in this step.",
    )


class ResponseState(BaseState):
    state_type: Literal["response"] = Field(
        default="response",
        description="Response node: emit an intermediate user-facing message.",
    )
    response: str = Field(
        min_length=1,
        description="Natural-language message content for this intermediate response.",
    )
    audience: Literal["user", "system"] = Field(
        default="user",
        description="Target audience for this response message.",
    )


class FinishState(BaseState):
    state_type: Literal["finish"] = Field(
        default="finish",
        description="Terminal node: complete the task with final output.",
    )
    final_output: str = Field(
        min_length=1,
        description="Final user-facing output returned when the task completes.",
    )
    completion_reason: str = Field(
        min_length=1,
        description="Short machine-readable reason describing why execution finished.",
    )


AgentState = Annotated[
    ReasoningState | ToolCallState | ResponseState | FinishState,
    Field(
        discriminator="state_type",
        description="Tagged union of all graph state payloads, discriminated by state_type.",
    ),
]


class StateEnvelope(BaseModel):
    state: AgentState = Field(
        description="Single state object emitted by the model for the current step."
    )
