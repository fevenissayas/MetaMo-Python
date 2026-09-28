"""Task, adaptation, counterfactual, curiosity, transfer, and safety metrics."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.state import MotivationalState
from mdrl.config import (
    EXTERNAL_EVALUATION_WEIGHTS,
    NUM_OUTCOMES,
    OUTCOME_COMPONENTS,
)
from mdrl.envs.curious_gridworld import Zone
from mdrl.stabilizer import boundary_band_rate

@dataclass
class EpisodeMetrics:
    """One evaluation episode."""

    scalar_return: float = 0.0
    discounted_return: float = 0.0
    outcome_totals: np.ndarray = field(default_factory=lambda: np.zeros(NUM_OUTCOMES))
    minerals: int = 0
    resources: int = 0
    env_steps: int = 0
    decisions: int = 0
    lava_steps: int = 0
    unsafe_steps: int = 0
    survived: bool = True
    final_energy: float = 0.0

    skill_decisions: int = 0
    certified_decisions: int = 0
    safe_route_choices: int = 0
    direct_route_choices: int = 0
    noisy_inspections: int = 0
    descriptor_regret: float = 0.0

    zone_dwell: Dict[str, int] = field(default_factory=dict)
    zone_coverage: Dict[str, float] = field(default_factory=dict)
    noisy_dwell_late: int = 0
    noisy_abandon_step: Optional[int] = None
    dynamic_reengagement: int = 0

    curiosity_report: Dict[str, Dict[str, float]] = field(default_factory=dict)
    appraisal_curiosity_report: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stability: Dict[str, float] = field(default_factory=dict)

    def success_rate(self) -> float:
        """Minerals collected per 100 environment steps."""
        if self.env_steps == 0:
            return 0.0
        return 100.0 * self.minerals / self.env_steps

    def energy_efficiency(self) -> float:
        """Minerals per unit of energy consumed."""
        consumed = max(1.0, 100.0 - self.final_energy + 40.0 * self.resources)
        return self.minerals / consumed

    def external_return(self) -> float:
        """Return under one fixed evaluator shared by every condition."""
        weights = np.asarray(EXTERNAL_EVALUATION_WEIGHTS, dtype=float)
        return float(weights @ self.outcome_totals)

    def unsafe_rate(self) -> float:
        if self.env_steps == 0:
            return 0.0
        return self.unsafe_steps / self.env_steps

    def lava_rate(self) -> float:
        if self.env_steps == 0:
            return 0.0
        return self.lava_steps / self.env_steps

    def skill_rate(self) -> float:
        if self.decisions == 0:
            return 0.0
        return self.skill_decisions / self.decisions

    def certified_fraction(self) -> float:
        if self.skill_decisions == 0:
            return 0.0
        return self.certified_decisions / self.skill_decisions

    def safe_route_rate(self) -> float:
        total = self.safe_route_choices + self.direct_route_choices
        if total == 0:
            return float("nan")
        return self.safe_route_choices / total

    def noisy_dwell_rate(self) -> float:
        total = sum(self.zone_dwell.values())
        if total == 0:
            return 0.0
        return self.zone_dwell.get("noisy", 0) / total

    def as_dict(self) -> Dict[str, float]:
        record = {
            "scalar_return": self.scalar_return,
            "external_return": self.external_return(),
            "discounted_return": self.discounted_return,
            "minerals": float(self.minerals),
            "resources": float(self.resources),
            "env_steps": float(self.env_steps),
            "decisions": float(self.decisions),
            "success_rate": self.success_rate(),
            "energy_efficiency": self.energy_efficiency(),
            "unsafe_rate": self.unsafe_rate(),
            "lava_rate": self.lava_rate(),
            "survived": float(self.survived),
            "skill_rate": self.skill_rate(),
            "certified_fraction": self.certified_fraction(),
            "safe_route_rate": self.safe_route_rate(),
            "descriptor_regret": self.descriptor_regret,
            "noisy_dwell_rate": self.noisy_dwell_rate(),
            "noisy_dwell_late": float(self.noisy_dwell_late),
            "noisy_abandon_step": (
                float(self.noisy_abandon_step) if self.noisy_abandon_step is not None else float("nan")
            ),
            "dynamic_reengagement": float(self.dynamic_reengagement),
            "learnable_coverage": self.zone_coverage.get("learnable", 0.0),
            "dynamic_coverage": self.zone_coverage.get("dynamic", 0.0),
        }
        for index, name in enumerate(OUTCOME_COMPONENTS):
            record[f"outcome_{name}"] = float(self.outcome_totals[index])
        for zone, report in self.curiosity_report.items():
            record[f"wm_error_{zone}"] = report.get("final_error", 0.0)
            record[f"wm_progress_{zone}"] = report.get("mean_progress", 0.0)
        for zone, report in self.appraisal_curiosity_report.items():
            record[f"appraisal_wm_error_{zone}"] = report.get("final_error", 0.0)
            record[f"appraisal_wm_progress_{zone}"] = report.get(
                "mean_progress", 0.0
            )
        record.update(self.stability)
        return record

def collect_episode_metrics(agent, env, gamma: float = 0.97) -> EpisodeMetrics:
    """Assemble one episode's metrics from the agent's and environment's logs."""
    records = agent.episode_records
    metrics = EpisodeMetrics()

    metrics.decisions = len(records)
    metrics.env_steps = env.step_count
    metrics.minerals = env.minerals_collected
    metrics.resources = env.resources_collected
    metrics.lava_steps = env.lava_steps
    metrics.final_energy = float(env.energy)
    metrics.survived = env.energy > 0.0

    elapsed = 0
    for record in records:
        metrics.outcome_totals = metrics.outcome_totals + record.outcome
        metrics.scalar_return += float(record.weights @ record.outcome)
        metrics.discounted_return += (gamma**elapsed) * float(record.weights @ record.outcome)
        elapsed += record.duration

        if record.candidate_kind == "skill":
            metrics.skill_decisions += 1
            if record.certified:
                metrics.certified_decisions += 1
        if record.candidate_id == "goto_mineral_safe":
            metrics.safe_route_choices += 1
        elif record.candidate_id == "goto_mineral_direct":
            metrics.direct_route_choices += 1
        elif record.candidate_id == "inspect_noisy":
            metrics.noisy_inspections += 1

    metrics.unsafe_steps = env.lava_steps + sum(
        1 for record in records if record.lava_distance <= 2 and not record.in_lava
    )

    metrics.zone_dwell = env.zone_summary()
    metrics.zone_coverage = env.zone_coverage()

    late_start = int(0.66 * max(1, len(env.zone_step_log)))
    metrics.noisy_dwell_late = env.zone_dwell_after(late_start, Zone.NOISY)
    metrics.noisy_abandon_step = env.last_step_in_zone(Zone.NOISY)
    if env.rule_switch_steps:
        metrics.dynamic_reengagement = env.zone_dwell_after(
            env.rule_switch_steps[0], Zone.DYNAMIC
        )

    metrics.descriptor_regret = _descriptor_regret(records)
    metrics.curiosity_report = agent.curiosity.zone_report()
    appraisal_curiosity = agent.active_appraisal_curiosity()
    if appraisal_curiosity is not None:
        metrics.appraisal_curiosity_report = appraisal_curiosity.zone_report()
    metrics.stability = dict(agent.stabilizer.summary())
    metrics.stability["boundary_band_rate"] = boundary_band_rate(agent.motivational_trace)
    metrics.stability.update(agent.appraisal.summary())
    metrics.stability.update(agent.curiosity.diagnostics())
    if appraisal_curiosity is not None:
        metrics.stability.update(
            {
                f"appraisal_{key}": value
                for key, value in appraisal_curiosity.diagnostics().items()
            }
        )
    return metrics

def _descriptor_regret(records) -> float:
    """Mean shortfall of the chosen candidate versus the best predicted effect."""
    if not records:
        return 0.0
    return float(np.mean([record.descriptor_regret for record in records]))

def counterfactual_sensitivity(
    agent,
    observations: Sequence[Dict[str, Any]],
    state_a: MotivationalState,
    state_b: MotivationalState,
) -> Dict[str, float]:
    """Policy-switch rate, and whether the switch improved predicted utility under the new motive."""
    switches = 0
    comparable = 0
    consistency: List[float] = []
    matched = 0

    for observation in observations:
        result_a = agent.score_under_state(observation, state_a)
        result_b = agent.score_under_state(observation, state_b)
        comparable += 1

        chosen_a = result_a["candidate"].id
        chosen_b = result_b["candidate"].id
        if chosen_a != chosen_b:
            switches += 1

        ids_b = [candidate.id for candidate in result_b["candidates"]]
        if chosen_a not in ids_b:
            continue
        position_a = ids_b.index(chosen_a)
        position_b = result_b["index"]
        matched += 1

        q_task = result_b["q_task"]
        q_lp = result_b["q_lp"]
        if q_task.size == 0:
            continue
        weights = result_b["weights"]
        if q_task.ndim == 2 and q_task.shape[1] == weights.shape[0]:
            task_gain = float(weights @ (q_task[position_b] - q_task[position_a]))
        else:
            task_gain = float(q_task[position_b].sum() - q_task[position_a].sum())
        lp_gain = float(result_b["beta"] * (q_lp[position_b] - q_lp[position_a]))
        consistency.append(task_gain + lp_gain)

    return {
        "policy_switch_rate": switches / comparable if comparable else 0.0,
        "motivational_consistency": float(np.mean(consistency)) if consistency else 0.0,
        "counterfactual_samples": float(comparable),
        "matched_samples": float(matched),
    }

def aggregate(episodes: Sequence[EpisodeMetrics]) -> Dict[str, float]:
    """Mean over episodes of every scalar metric, ignoring NaNs."""
    if not episodes:
        return {}
    records = [episode.as_dict() for episode in episodes]
    keys = sorted({key for record in records for key in record})
    summary: Dict[str, float] = {}
    for key in keys:
        values = [
            record[key]
            for record in records
            if key in record and not (isinstance(record[key], float) and np.isnan(record[key]))
        ]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary
