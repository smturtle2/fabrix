from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from fabrix.tools.output import ToolPart


class NextState(StrEnum):
    reasoning = "reasoning"
    tool_call = "tool_call"
    response = "response"


class BaseState(BaseModel):
    state_type: str = Field(
        description="Current node type in the execution graph."
    )
    next_state: NextState = Field(
        description=(
            "Next node that the executor should transition to after this state. "
            "Set next_state=null only in response state to terminate after emitting the response event."
        )
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
        description="Reasoning node: record step-level decision trace and pick the next state.",
    )
    reasoning: str = Field(
        min_length=1,
        description="Short 1-2 sentence decision trace summary for this step. Prefer English text.",
    )
    focus: str = Field(
        min_length=1,
        description="Short, concrete objective for this current step. Prefer English text.",
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
    next_state: NextState | None = Field(
        description=(
            "Next node that the executor should transition to after this response. "
            "You must set next_state to null to terminate immediately after emitting this response event."
        )
    )
    response: str | None = Field(
        default=None,
        description="Optional natural-language message content for this intermediate response.",
    )
    parts: list[ToolPart] | None = Field(
        default=None,
        description="Optional structured multimodal parts (text/image/json) for this response.",
    )
    audience: Literal["user", "system"] = Field(
        default="user",
        description="Target audience for this response message.",
    )


AgentState = Annotated[
    ReasoningState | ToolCallState | ResponseState,
    Field(
        discriminator="state_type",
        description="Tagged union of all graph state payloads, discriminated by state_type.",
    ),
]


class StateEnvelope(BaseModel):
    state: AgentState = Field(
        description="Single state object emitted by the model for the current step."
    )
