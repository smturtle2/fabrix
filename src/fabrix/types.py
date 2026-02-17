from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

type JSONDict = dict[str, Any]
type ReasoningEffort = Literal["low", "medium", "high"]
type ImageInput = str | Path | bytes
