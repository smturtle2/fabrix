from __future__ import annotations

import asyncio
import json
from statistics import mean

from pydantic import BaseModel, Field

from fabrix import Agent
from fabrix.events import (
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
    ToolEvent,
)

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


def clean_records(payload: CleanRecordsInput) -> list[dict[str, float | str]]:
    """Drop invalid records and normalize value precision."""
    cleaned: list[dict[str, float | str]] = []
    for row in payload.rows:
        if row.value < 0:
            continue
        cleaned.append({"category": row.category, "value": round(float(row.value), 2)})
    return cleaned


class AggregateInput(BaseModel):
    rows: list[DataRow] = Field(min_length=1)


def aggregate_by_category(payload: AggregateInput) -> list[dict[str, float | int | str]]:
    """Compute count and average per category as list rows."""
    buckets: dict[str, list[float]] = {}
    for row in payload.rows:
        buckets.setdefault(row.category, []).append(float(row.value))

    summary_rows: list[dict[str, float | int | str]] = []
    for category, values in sorted(buckets.items()):
        summary_rows.append({"category": category, "count": len(values), "avg": round(mean(values), 2)})
    return summary_rows


class SummaryRow(BaseModel):
    category: str = Field(min_length=1)
    count: int = Field(ge=1)
    avg: float


class RenderInput(BaseModel):
    summary_rows: list[SummaryRow] = Field(min_length=1)


def render_report(payload: RenderInput) -> str:
    """Render category summary rows into a compact, deterministic JSON report."""
    report_rows = [row.model_dump(mode="json") for row in payload.summary_rows]
    return json.dumps(report_rows, ensure_ascii=True, sort_keys=True)


async def main() -> None:
    agent = Agent(
        instructions=(
            "You are a data workflow orchestrator. Execute this pipeline exactly once: "
            "1) clean_records with context.raw_rows, "
            "2) aggregate_by_category with the cleaned rows, "
            "3) render_report with summary_rows from aggregation output. "
            "Do not call the same tool repeatedly with the same arguments. "
            "After render_report succeeds, immediately emit finish state with final_output "
            "equal to render_report output."
        ),
        tools=[clean_records, aggregate_by_category, render_report],
        max_steps=14,
    )

    task = "Analyze context.raw_rows and produce a category-level report."
    context = {"raw_rows": RAW_DATA}

    async for event in agent.run_task_stream(task, context=context):
        if isinstance(event, ReasoningEvent):
            print(f"[step={event.step}] reasoning={event.reasoning}")
            print(f"[step={event.step}] focus={event.focus} next={event.next_state}")
        elif isinstance(event, ToolEvent):
            if event.phase == "start":
                print(f"[step={event.step}] tool_call={event.tool_name} args={event.arguments}")
            elif event.result is not None:
                print(f"[step={event.step}] tool={event.tool_name} ok={event.result.ok}")
                if event.result.error:
                    print(f"[step={event.step}] tool_error={event.result.error}")
        elif isinstance(event, ResponseEvent):
            print(f"[step={event.step}] response={event.response}")
        elif isinstance(event, TaskFinishedEvent):
            print("completion reason:", event.completion_reason)
            print("final report:", event.final_output)
        elif isinstance(event, TaskFailedEvent):
            print("failed:", event.error_code, event.message)


if __name__ == "__main__":
    asyncio.run(main())
