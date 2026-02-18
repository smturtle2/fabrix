from __future__ import annotations


class FabrixError(Exception):
    """Base exception for Fabrix."""


class LLMOutputError(FabrixError):
    """Raised when structured LLM output is invalid."""


class RetryableLLMOutputError(LLMOutputError):
    """Raised when structured output generation can be retried."""
