from __future__ import annotations


class FabrixError(Exception):
    """Base exception for Fabrix."""


class LLMOutputError(FabrixError):
    """Raised when structured LLM output is invalid."""
