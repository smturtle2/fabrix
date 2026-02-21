from __future__ import annotations

import asyncio
import json
import sys
from statistics import mean
from pathlib import Path

from pydantic import BaseModel, Field

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common import run_case
from fabrix import Agent
from fabrix.messages import TextMessage
from fabrix.tools import ToolOutput

RAW_DATA = [
    {"category": "A", "value": 12.4},
    {"category": "A", "value": -1.0},
    {"category": "B", "value": 19.8},
    {"category": "B", "value": 21.5},
    {"category": "A", "value": 14.9},
    {"category": "C", "value": 9.7},
]


class DataRow(BaseModel):
    category: str = Field(min_length=1)
    value: float


class CleanRecordsInput(BaseModel):
    rows: list[DataRow] = Field(min_length=1)


def clean_records(payload: CleanRecordsInput) -> ToolOutput:
    """Drop invalid records and normalize value precision."""
    cleaned: list[dict[str, float | str]] = []
    for row in payload.rows:
        if row.value < 0:
            continue
        cleaned.append({"category": row.category, "value": round(float(row.value), 2)})
    return ToolOutput.json(cleaned)


class AggregateInput(BaseModel):
    rows: list[DataRow] = Field(min_length=1)


def aggregate_by_category(payload: AggregateInput) -> ToolOutput:
    """Compute count and average per category as list rows."""
    buckets: dict[str, list[float]] = {}
    for row in payload.rows:
        buckets.setdefault(row.category, []).append(float(row.value))

    summary_rows: list[dict[str, float | int | str]] = []
    for category, values in sorted(buckets.items()):
        summary_rows.append({"category": category, "count": len(values), "avg": round(mean(values), 2)})
    return ToolOutput.json(summary_rows)


class SummaryRow(BaseModel):
    category: str = Field(min_length=1)
    count: int = Field(ge=1)
    avg: float


class RenderInput(BaseModel):
    summary_rows: list[SummaryRow] = Field(min_length=1)


def render_report(payload: RenderInput) -> ToolOutput:
    """Render category summary rows into a compact, deterministic JSON report."""
    report_rows = [row.model_dump(mode="json") for row in payload.summary_rows]
    return ToolOutput.text(json.dumps(report_rows, ensure_ascii=True, sort_keys=True))


async def main() -> None:
    agent = Agent(
        instructions=(
            "You are a data workflow orchestrator. Execute this pipeline exactly once: "
            "1) clean_records with rows from the user message, "
            "2) aggregate_by_category with the cleaned rows, "
            "3) render_report with summary_rows from aggregation output. "
            "Do not call the same tool repeatedly with the same arguments. "
            "After render_report succeeds, immediately emit response state with next_state "
            "set to null and response equal to render_report output."
        ),
        model="gpt-5.3-codex",
        tools=[clean_records, aggregate_by_category, render_report],
    )

    raw_rows_json = json.dumps(RAW_DATA, ensure_ascii=True)
    messages = [
        TextMessage(
            role="user",
            text=(
                "Analyze the following raw rows and produce a category-level report. "
                "Use the registered tools. "
                f"raw_rows={raw_rows_json}"
            ),
        )
    ]
    await run_case(agent, messages=messages)


if __name__ == "__main__":
    asyncio.run(main())
