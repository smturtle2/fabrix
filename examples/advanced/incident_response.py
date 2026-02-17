from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, Field

from fabrix import Agent
from fabrix.events import (
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
    ToolEvent,
)
from fabrix.messages import TextMessage

RAW_ALERTS = [
    {
        "service": "checkout",
        "status": "error",
        "latency_ms": 1320,
        "users_affected": 900,
        "region": "us-east-1",
    },
    {
        "service": "checkout",
        "status": "degraded",
        "latency_ms": 980,
        "users_affected": 620,
        "region": "us-west-2",
    },
    {
        "service": "payments",
        "status": "degraded",
        "latency_ms": 730,
        "users_affected": 240,
        "region": "us-east-1",
    },
    {
        "service": "search",
        "status": "error",
        "latency_ms": 610,
        "users_affected": 180,
        "region": "us-east-1",
    },
    {
        "service": "checkout",
        "status": "error",
        "latency_ms": 1490,
        "users_affected": 780,
        "region": "eu-central-1",
    },
    {
        "service": "inventory",
        "status": "ok",
        "latency_ms": 210,
        "users_affected": 0,
        "region": "us-west-2",
    },
]

SERVICE_DEPENDENCIES = {
    "checkout": ["cart", "payments", "inventory"],
    "payments": ["ledger", "risk-engine"],
    "search": ["catalog-index"],
    "inventory": ["warehouse-sync"],
}

ONCALL_OWNERS = {
    "checkout": "checkout-oncall",
    "payments": "payments-oncall",
    "search": "search-oncall",
    "inventory": "inventory-oncall",
}


class Alert(BaseModel):
    service: str = Field(min_length=1)
    status: str = Field(pattern="^(error|degraded|ok)$")
    latency_ms: int = Field(ge=0)
    users_affected: int = Field(ge=0)
    region: str = Field(min_length=2)


class SummarizeAlertsInput(BaseModel):
    alerts: list[Alert] = Field(min_length=1)


def summarize_alerts(payload: SummarizeAlertsInput) -> list[dict[str, int | str]]:
    """Rank services by incident pressure score from raw alerts."""
    buckets: dict[str, dict[str, int]] = {}
    for alert in payload.alerts:
        bucket = buckets.setdefault(
            alert.service,
            {"errors": 0, "degraded": 0, "latency_peak_ms": 0, "users_affected": 0},
        )
        if alert.status == "error":
            bucket["errors"] += 1
        elif alert.status == "degraded":
            bucket["degraded"] += 1
        if alert.latency_ms > bucket["latency_peak_ms"]:
            bucket["latency_peak_ms"] = alert.latency_ms
        bucket["users_affected"] += alert.users_affected

    ranked: list[dict[str, int | str]] = []
    for service, stats in buckets.items():
        score = (
            stats["errors"] * 4
            + stats["degraded"] * 2
            + (2 if stats["latency_peak_ms"] >= 1200 else 0)
            + (stats["users_affected"] // 700)
        )
        ranked.append(
            {
                "service": service,
                "errors": stats["errors"],
                "degraded": stats["degraded"],
                "latency_peak_ms": stats["latency_peak_ms"],
                "users_affected": stats["users_affected"],
                "score": score,
            }
        )
    ranked.sort(key=lambda row: (-int(row["score"]), -int(row["users_affected"]), str(row["service"])))
    return ranked


class RankedService(BaseModel):
    service: str = Field(min_length=1)
    errors: int = Field(ge=0)
    degraded: int = Field(ge=0)
    latency_peak_ms: int = Field(ge=0)
    users_affected: int = Field(ge=0)
    score: int = Field(ge=0)


class ChooseIncidentInput(BaseModel):
    ranked_services: list[RankedService] = Field(min_length=1)


def choose_primary_incident(payload: ChooseIncidentInput) -> dict[str, int | str]:
    """Pick the highest-pressure service and derive severity."""
    top = payload.ranked_services[0]
    if top.score >= 14 or top.users_affected >= 3000:
        severity = "SEV-1"
    elif top.score >= 9 or top.users_affected >= 1500:
        severity = "SEV-2"
    else:
        severity = "SEV-3"

    return {
        "service": top.service,
        "severity": severity,
        "users_affected": top.users_affected,
        "reason": (
            f"errors={top.errors}, degraded={top.degraded}, users={top.users_affected}, score={top.score}"
        ),
    }


class DependenciesInput(BaseModel):
    service: str = Field(min_length=1)


def fetch_service_dependencies(payload: DependenciesInput) -> dict[str, list[str] | str]:
    """Return known dependencies and coarse blast-radius tier."""
    dependencies = SERVICE_DEPENDENCIES.get(payload.service, [])
    if len(dependencies) >= 3:
        blast_radius = "high"
    elif len(dependencies) == 2:
        blast_radius = "medium"
    else:
        blast_radius = "low"
    return {
        "service": payload.service,
        "dependencies": dependencies,
        "blast_radius": blast_radius,
    }


class BuildPlanInput(BaseModel):
    service: str = Field(min_length=1)
    severity: str = Field(pattern="^SEV-[1-3]$")
    users_affected: int = Field(ge=0)
    dependencies: list[str] = Field(default_factory=list)


def build_mitigation_plan(payload: BuildPlanInput) -> dict[str, int | str | list[str]]:
    """Build a short, deterministic mitigation plan for the selected service."""
    owner = ONCALL_OWNERS.get(payload.service, "platform-oncall")
    eta_by_severity = {"SEV-1": 20, "SEV-2": 45, "SEV-3": 90}

    actions = [
        f"Page {owner} and open incident bridge.",
        f"Apply temporary rate limiting on {payload.service}.",
    ]
    if payload.dependencies:
        actions.append(f"Check dependency health: {', '.join(payload.dependencies)}.")
    actions.append("Prepare rollback if error budget burn remains elevated for 10 minutes.")

    return {
        "owner": owner,
        "escalation_channel": f"#inc-{payload.service}",
        "eta_minutes": eta_by_severity[payload.severity],
        "actions": actions,
    }


class DraftUpdateInput(BaseModel):
    service: str = Field(min_length=1)
    severity: str = Field(pattern="^SEV-[1-3]$")
    users_affected: int = Field(ge=0)
    owner: str = Field(min_length=1)
    escalation_channel: str = Field(min_length=1)
    eta_minutes: int = Field(ge=1)
    actions: list[str] = Field(min_length=1)


def draft_external_update(payload: DraftUpdateInput) -> str:
    """Render a compact external status-page update as JSON."""
    message = {
        "incident": f"{payload.service} disruption",
        "severity": payload.severity,
        "status": "investigating",
        "affected_users": payload.users_affected,
        "commander": payload.owner,
        "coordination_channel": payload.escalation_channel,
        "next_update_in_minutes": payload.eta_minutes,
        "current_actions": payload.actions,
    }
    return json.dumps(message, ensure_ascii=True, sort_keys=True)


async def main() -> None:
    agent = Agent(
        instructions=(
            "You are an incident commander. Execute this sequence exactly once: "
            "1) summarize_alerts with alerts from user message, "
            "2) choose_primary_incident with ranked_services from step 1, "
            "3) fetch_service_dependencies with service from step 2, "
            "4) build_mitigation_plan with service, severity, users_affected from step 2 plus "
            "dependencies from step 3, "
            "5) draft_external_update with service, severity, users_affected from step 2 and "
            "owner, escalation_channel, eta_minutes, actions from step 4. "
            "Do not repeat a tool call with identical arguments. After step 5 succeeds, "
            "immediately emit finish state with final_output equal to draft_external_update output."
        ),
        model="gpt-5.3-codex",
        tools=[
            summarize_alerts,
            choose_primary_incident,
            fetch_service_dependencies,
            build_mitigation_plan,
            draft_external_update,
        ],
    )

    alerts_json = json.dumps(RAW_ALERTS, ensure_ascii=True)
    messages = [
        TextMessage(
            role="user",
            text=(
                "Triage the following alerts, run the incident response workflow, "
                "and publish an external update. "
                f"alerts={alerts_json}"
            ),
        )
    ]

    async for event in agent.run_stream(messages=messages):
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
            print("external update:", event.final_output)
        elif isinstance(event, TaskFailedEvent):
            print("failed:", event.error_code, event.message)


if __name__ == "__main__":
    asyncio.run(main())
