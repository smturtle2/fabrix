from __future__ import annotations

import asyncio
import json
import sys
from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common import print_event
from fabrix import Agent
from fabrix.messages import TextMessage
from fabrix.tools import ToolOutput

RiskLevel = Literal["low", "medium", "high"]
LoadLevel = Literal["low", "medium", "high"]

RISK_RANK: dict[RiskLevel, int] = {"low": 0, "medium": 1, "high": 2}
COMPLIANCE_SCORE: dict[RiskLevel, float] = {"low": 95.0, "medium": 70.0, "high": 35.0}

PROJECT_PLANS = [
    {
        "plan_id": "alpha-rehost",
        "infra_cost_k": 210,
        "launch_weeks": 6,
        "reliability_gain": 72,
        "compliance_risk": "low",
        "engineering_load": "high",
    },
    {
        "plan_id": "beta-hybrid",
        "infra_cost_k": 170,
        "launch_weeks": 8,
        "reliability_gain": 58,
        "compliance_risk": "medium",
        "engineering_load": "medium",
    },
    {
        "plan_id": "gamma-refactor",
        "infra_cost_k": 250,
        "launch_weeks": 12,
        "reliability_gain": 86,
        "compliance_risk": "low",
        "engineering_load": "high",
    },
    {
        "plan_id": "delta-minimal",
        "infra_cost_k": 120,
        "launch_weeks": 4,
        "reliability_gain": 40,
        "compliance_risk": "high",
        "engineering_load": "low",
    },
]

HARD_CONSTRAINTS = {
    "max_cost_k": 220,
    "max_launch_weeks": 8,
    "max_compliance_risk": "medium",
}

BASE_WEIGHTS = {
    "reliability": 0.45,
    "cost": 0.2,
    "delivery": 0.2,
    "compliance": 0.15,
}

SOFT_OBJECTIVES = {
    "prefer_lower_engineering_load_when_scores_close": True,
    "target_min_reliability_gain": 55,
    "favor_schedule_headroom_for_peak_season_readiness": True,
    "if_winner_is_unstable_under_stress_test": "include fallback plan",
}


class Plan(BaseModel):
    plan_id: str = Field(min_length=1)
    infra_cost_k: int = Field(ge=0)
    launch_weeks: int = Field(ge=1)
    reliability_gain: int = Field(ge=0, le=100)
    compliance_risk: RiskLevel
    engineering_load: LoadLevel


class Constraints(BaseModel):
    max_cost_k: int = Field(ge=1)
    max_launch_weeks: int = Field(ge=1)
    max_compliance_risk: RiskLevel


class FilterFeasibleInput(BaseModel):
    plans: list[Plan] = Field(min_length=1)
    constraints: Constraints


def filter_feasible_plans(payload: FilterFeasibleInput) -> ToolOutput:
    feasible: list[dict[str, int | str]] = []
    risk_limit = RISK_RANK[payload.constraints.max_compliance_risk]

    for plan in payload.plans:
        if plan.infra_cost_k > payload.constraints.max_cost_k:
            continue
        if plan.launch_weeks > payload.constraints.max_launch_weeks:
            continue
        if RISK_RANK[plan.compliance_risk] > risk_limit:
            continue

        feasible.append(
            {
                "plan_id": plan.plan_id,
                "infra_cost_k": plan.infra_cost_k,
                "launch_weeks": plan.launch_weeks,
                "reliability_gain": plan.reliability_gain,
                "compliance_risk": plan.compliance_risk,
                "engineering_load": plan.engineering_load,
                "cost_headroom_k": payload.constraints.max_cost_k - plan.infra_cost_k,
                "schedule_headroom_weeks": payload.constraints.max_launch_weeks - plan.launch_weeks,
            }
        )

    return ToolOutput.json(feasible)


class Weights(BaseModel):
    reliability: float = Field(gt=0)
    cost: float = Field(gt=0)
    delivery: float = Field(gt=0)
    compliance: float = Field(gt=0)


class FeasiblePlan(BaseModel):
    plan_id: str = Field(min_length=1)
    infra_cost_k: int = Field(ge=0)
    launch_weeks: int = Field(ge=1)
    reliability_gain: int = Field(ge=0, le=100)
    compliance_risk: RiskLevel
    engineering_load: LoadLevel
    cost_headroom_k: int
    schedule_headroom_weeks: int


class RankPlansInput(BaseModel):
    feasible_plans: list[FeasiblePlan] = Field(min_length=1)
    weights: Weights


def _weighted_score(
    *,
    infra_cost_k: int,
    launch_weeks: int,
    reliability_gain: int,
    compliance_risk: RiskLevel,
    weights: Weights,
) -> float:
    cost_score = max(0.0, min(100.0, 100.0 - float(infra_cost_k) * 0.35))
    delivery_score = max(0.0, min(100.0, 100.0 - float(launch_weeks) * 8.0))
    reliability_score = float(reliability_gain)
    compliance_score = COMPLIANCE_SCORE[compliance_risk]
    return (
        reliability_score * weights.reliability
        + cost_score * weights.cost
        + delivery_score * weights.delivery
        + compliance_score * weights.compliance
    )


def _score_plan(
    *,
    plan: FeasiblePlan,
    weights: Weights,
) -> float:
    return _weighted_score(
        infra_cost_k=plan.infra_cost_k,
        launch_weeks=plan.launch_weeks,
        reliability_gain=plan.reliability_gain,
        compliance_risk=plan.compliance_risk,
        weights=weights,
    )


def rank_feasible_plans(payload: RankPlansInput) -> ToolOutput:
    ranked: list[dict[str, int | float | str]] = []

    for plan in payload.feasible_plans:
        total_score = round(_score_plan(plan=plan, weights=payload.weights), 2)
        ranked.append(
            {
                "plan_id": plan.plan_id,
                "score": total_score,
                "reliability_gain": plan.reliability_gain,
                "infra_cost_k": plan.infra_cost_k,
                "launch_weeks": plan.launch_weeks,
                "compliance_risk": plan.compliance_risk,
                "engineering_load": plan.engineering_load,
                "cost_headroom_k": plan.cost_headroom_k,
                "schedule_headroom_weeks": plan.schedule_headroom_weeks,
            }
        )

    ranked.sort(key=lambda row: (-float(row["score"]), int(row["infra_cost_k"]), str(row["plan_id"])))
    return ToolOutput.json(ranked)


class RankedPlan(BaseModel):
    plan_id: str = Field(min_length=1)
    score: float
    reliability_gain: int = Field(ge=0, le=100)
    infra_cost_k: int = Field(ge=0)
    launch_weeks: int = Field(ge=1)
    compliance_risk: RiskLevel
    engineering_load: LoadLevel
    cost_headroom_k: int
    schedule_headroom_weeks: int


class StressTestInput(BaseModel):
    ranked_plans: list[RankedPlan] = Field(min_length=1)


def stress_test_top_plans(payload: StressTestInput) -> ToolOutput:
    top = payload.ranked_plans[:2]
    if not top:
        return ToolOutput.json({"scenarios": []})

    scenarios = {
        "reliability_heavy": Weights(reliability=0.6, cost=0.15, delivery=0.1, compliance=0.15),
        "cost_crunch": Weights(reliability=0.3, cost=0.4, delivery=0.2, compliance=0.1),
        "delivery_urgent": Weights(reliability=0.35, cost=0.15, delivery=0.4, compliance=0.1),
    }

    results: list[dict[str, str | float]] = []
    for scenario_name, weights in scenarios.items():
        winner = max(
            top,
            key=lambda plan: _weighted_score(
                infra_cost_k=plan.infra_cost_k,
                launch_weeks=plan.launch_weeks,
                reliability_gain=plan.reliability_gain,
                compliance_risk=plan.compliance_risk,
                weights=weights,
            ),
        )
        results.append(
            {
                "scenario": scenario_name,
                "winner_plan_id": winner.plan_id,
                "winner_score": round(
                    _weighted_score(
                        infra_cost_k=winner.infra_cost_k,
                        launch_weeks=winner.launch_weeks,
                        reliability_gain=winner.reliability_gain,
                        compliance_risk=winner.compliance_risk,
                        weights=weights,
                    ),
                    2,
                ),
            }
        )

    stability = len({item["winner_plan_id"] for item in results}) == 1
    return ToolOutput.json({"scenarios": results, "stable_winner": stability})


class ScenarioResult(BaseModel):
    scenario: str = Field(min_length=1)
    winner_plan_id: str = Field(min_length=1)
    winner_score: float


class RenderBriefInput(BaseModel):
    ranked_plans: list[RankedPlan] = Field(min_length=1)
    stress_scenarios: list[ScenarioResult] = Field(min_length=1)
    stable_winner: bool


def render_decision_brief(payload: RenderBriefInput) -> ToolOutput:
    top = payload.ranked_plans[0]
    scenario_winners = ", ".join(
        f"{row.scenario}:{row.winner_plan_id}" for row in payload.stress_scenarios
    )
    stability_note = "stable across scenarios" if payload.stable_winner else "sensitive to weights"
    brief = (
        f"Recommend `{top.plan_id}` (score={top.score}). "
        f"Trade-offs: cost={top.infra_cost_k}k, launch={top.launch_weeks}w, "
        f"reliability_gain={top.reliability_gain}, compliance={top.compliance_risk}. "
        f"Stress-test winners -> {scenario_winners}. "
        f"Decision robustness: {stability_note}."
    )
    return ToolOutput.text(brief)


async def main() -> None:
    agent = Agent(
        instructions=(
            "You are a principal architect making a multi-constraint migration decision. "
            "Use the registered tools to evaluate feasibility, rank candidates, and validate robustness. "
            "A good flow usually starts by filtering feasibility, then ranking feasible candidates, "
            "then stress-testing likely winners, and finally rendering the decision brief. "
            "Before finalizing, make sure the recommendation accounts for hard constraints and soft objectives, "
            "including what to do if stress-test winners are unstable. "
            "Do not repeat the same tool call with identical arguments. "
            "When done, set response to render_decision_brief output and set next_state to null."
        ),
        state_models={"reasoning": "gpt-5.3-codex"},
        tools=[
            filter_feasible_plans,
            rank_feasible_plans,
            stress_test_top_plans,
            render_decision_brief,
        ],
    )

    plans_json = json.dumps(PROJECT_PLANS, ensure_ascii=True)
    constraints_json = json.dumps(HARD_CONSTRAINTS, ensure_ascii=True)
    weights_json = json.dumps(BASE_WEIGHTS, ensure_ascii=True)
    objectives_json = json.dumps(SOFT_OBJECTIVES, ensure_ascii=True)

    messages = [
        TextMessage(
            role="user",
            text=(
                "Select the best migration plan with transparent trade-offs. "
                f"plans={plans_json} constraints={constraints_json} "
                f"weights={weights_json} soft_objectives={objectives_json}"
            ),
        )
    ]

    step_flow: list[str] = []
    last_step_event: tuple[int, str] | None = None
    current_reasoning_streak = 0
    max_reasoning_streak = 0
    initial_reasoning_streak = 0
    in_initial_reasoning_block = True

    async for event in agent.run_stream(messages=messages):
        print_event(event)
        step_event = (event.step, event.event_type)
        if step_event == last_step_event:
            continue

        last_step_event = step_event
        step_flow.append(event.event_type)

        if event.event_type == "reasoning":
            current_reasoning_streak += 1
            max_reasoning_streak = max(max_reasoning_streak, current_reasoning_streak)
            if in_initial_reasoning_block:
                initial_reasoning_streak += 1
            continue

        current_reasoning_streak = 0
        in_initial_reasoning_block = False

    print(f"\nstep_flow={' -> '.join(step_flow)}")
    print(f"initial_reasoning_streak={initial_reasoning_streak}")
    print(f"max_reasoning_streak={max_reasoning_streak}")


if __name__ == "__main__":
    asyncio.run(main())
