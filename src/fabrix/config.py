from __future__ import annotations

from pydantic import BaseModel, Field

from .types import ReasoningEffort


class AgentConfig(BaseModel):
    instructions: str = Field(min_length=1)
    model: str = "gpt-5.3-codex"
    max_steps: int = Field(default=24, ge=1)
    reasoning_effort: ReasoningEffort = "medium"
