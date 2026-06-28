from dataclasses import dataclass
from typing import Final

import numpy as np

from core.actions import ActionSpec, derive_action_profile
from core.config import (
    G_CURIO,
    G_ETHIC,
    G_HELP,
    G_IND,
    G_NOVEL,
    G_SELF,
    G_SOC,
    G_TRANS,
    M_RESOLUTION,
    M_SECURING,
    M_THRESHOLD,
)
from core.state import Action, MotivationalState, Stimulus
from magus.decision import MagusDecision

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class InferenceTaskSpec:
    """Internal reasoning task scored through the existing MAGUS action scorer."""

    action_spec: ActionSpec
    base_cost: float
    breadth: float
    depth: float


@dataclass(frozen=True)
class InferenceSelection:
    id: str
    instruction: str
    score: float
    risk_estimate: float
    cost_estimate: float
    breadth: float
    depth: float
    status: str


@dataclass(frozen=True)
class InferencePlan:
    """Selected, deferred, and pruned internal reasoning work."""

    selected: tuple[InferenceSelection, ...]
    deferred: tuple[InferenceSelection, ...]
    pruned: tuple[InferenceSelection, ...]
    breadth: float
    depth: float
    confidence_threshold: float

    def selected_ids(self) -> tuple[str, ...]:
        return tuple(selection.id for selection in self.selected)

    def instruction_text(self) -> str:
        if not self.selected:
            return "Use the selected response action directly; no internal reasoning task was selected."
        return "\n".join(
            f"{index}. {selection.id}: {selection.instruction} "
            f"(score={selection.score:.3f}, risk={selection.risk_estimate:.2f}, cost={selection.cost_estimate:.2f})"
            for index, selection in enumerate(self.selected, start=1)
        )


INFERENCE_TASKS: Final[dict[str, InferenceTaskSpec]] = {
    "extract_relevant_context": InferenceTaskSpec(
        action_spec=ActionSpec(
            planning="Find the most relevant facts, definitions, and claims in the available context.",
            execution="Ground the answer in the most relevant supplied context before adding synthesis.",
            mode="stabilizing",
            dominant_goals=(G_HELP,),
            supporting_goals=(G_ETHIC,),
        ),
        base_cost=0.25,
        breadth=0.35,
        depth=0.70,
    ),
    "check_safety_and_ethics": InferenceTaskSpec(
        action_spec=ActionSpec(
            planning="Inspect whether the request or answer path has safety, ethics, or misuse risk.",
            execution="Check safety and ethics implications before answering, and constrain unsafe paths.",
            mode="protective",
            dominant_goals=(G_ETHIC,),
            supporting_goals=(G_HELP,),
            opposed_goals=(G_CURIO, G_NOVEL),
        ),
        base_cost=0.20,
        breadth=0.25,
        depth=0.75,
    ),
    "compare_reasoning_paths": InferenceTaskSpec(
        action_spec=ActionSpec(
            planning="Compare plausible interpretations, methods, or architectural alternatives.",
            execution="Compare the strongest relevant interpretations or options, then choose the most useful framing.",
            mode="balanced",
            dominant_goals=(G_HELP,),
            supporting_goals=(G_CURIO, G_NOVEL, G_SOC),
        ),
        base_cost=0.35,
        breadth=0.75,
        depth=0.45,
    ),
    "verify_constraints": InferenceTaskSpec(
        action_spec=ActionSpec(
            planning="Check claims against architectural constraints, paper equations, and implementation boundaries.",
            execution="Verify important claims against the paper and the current architecture before finalizing.",
            mode="stabilizing",
            dominant_goals=(G_ETHIC, G_HELP),
            supporting_goals=(G_SELF,),
        ),
        base_cost=0.45,
        breadth=0.30,
        depth=0.85,
    ),
    "explore_extensions": InferenceTaskSpec(
        action_spec=ActionSpec(
            planning="Explore novel implications, future work, or implementation extensions.",
            execution="Explore useful extensions, but label uncertainty and avoid overstating speculative claims.",
            mode="exploratory",
            dominant_goals=(G_CURIO, G_NOVEL),
            supporting_goals=(G_SELF, G_HELP),
        ),
        base_cost=0.40,
        breadth=0.85,
        depth=0.45,
    ),
    "synthesize_direct_answer": InferenceTaskSpec(
        action_spec=ActionSpec(
            planning="Synthesize a direct answer from selected evidence and the user's immediate intent.",
            execution="Produce a direct answer that uses the selected reasoning results without unnecessary detours.",
            mode="stabilizing",
            dominant_goals=(G_HELP,),
            supporting_goals=(G_ETHIC, G_SOC),
        ),
        base_cost=0.20,
        breadth=0.30,
        depth=0.55,
    ),
}


RISK_CONTEXT_GAIN_BY_MODE: Final[dict[str, float]] = {
    "protective": 0.00,
    "stabilizing": 0.15,
    "balanced": 0.25,
    "exploratory": 0.45,
}


def _task_affordance(spec: InferenceTaskSpec, stimulus: Stimulus) -> float:
    benign_novelty = stimulus.novelty * (1.0 - stimulus.risk)
    if spec.action_spec.mode == "protective":
        return stimulus.risk
    if spec.action_spec.mode == "exploratory":
        return benign_novelty
    if spec.action_spec.mode == "balanced":
        return (stimulus.conduciveness + stimulus.novelty + stimulus.effort) / 3.0
    return (stimulus.conduciveness + stimulus.effort + (1.0 - stimulus.risk)) / 3.0


def _task_risk(spec: InferenceTaskSpec, stimulus: Stimulus) -> float:
    profile = derive_action_profile(spec.action_spec)
    context_gain = RISK_CONTEXT_GAIN_BY_MODE[spec.action_spec.mode]
    return float(np.clip(profile.base_risk + (context_gain * stimulus.risk), 0.0, 1.0))


def _task_cost(spec: InferenceTaskSpec, stimulus: Stimulus) -> float:
    return float(np.clip(spec.base_cost + (0.25 * stimulus.effort * max(spec.depth, spec.breadth)), 0.0, 1.0))


def _task_action(task_id: str, spec: InferenceTaskSpec, stimulus: Stimulus) -> Action:
    profile = derive_action_profile(spec.action_spec)
    relevance = max(_task_affordance(spec, stimulus), 0.10)
    return Action(
        id=task_id,
        goal_correlations=np.array(profile.goal_correlations, dtype=float) * relevance,
        risk_estimate=_task_risk(spec, stimulus),
    )


class MotivatedInferenceController:
    """Allocates internal reasoning work using the existing MAGUS candidate scorer."""

    def __init__(self, decision: MagusDecision | None = None):
        self.decision = decision or MagusDecision()

    def _selection(
        self,
        task_id: str,
        spec: InferenceTaskSpec,
        stimulus: Stimulus,
        score: float,
        status: str,
    ) -> InferenceSelection:
        return InferenceSelection(
            id=task_id,
            instruction=spec.action_spec.execution,
            score=score,
            risk_estimate=_task_risk(spec, stimulus),
            cost_estimate=_task_cost(spec, stimulus),
            breadth=spec.breadth,
            depth=spec.depth,
            status=status,
        )

    def plan(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        top_k: int = DEFAULT_TOP_K,
    ) -> InferencePlan:
        caution_signal = (state.M[M_THRESHOLD] + state.M[M_SECURING]) / 2.0
        resolution_signal = state.M[M_RESOLUTION]
        confidence_threshold = float(np.clip(0.15 + (0.45 * caution_signal), 0.0, 0.85))
        risk_cutoff = float(np.clip(0.95 - (0.55 * state.G[G_IND] * caution_signal), 0.25, 0.95))
        selection_budget = top_k
        if state.G[G_TRANS] > state.G[G_IND] and stimulus.risk < 0.4:
            selection_budget += 1

        ranked = []
        for task_id, spec in INFERENCE_TASKS.items():
            action = _task_action(task_id, spec, stimulus)
            score = self.decision.score_candidate(state, action)
            score -= 0.25 * _task_cost(spec, stimulus) * max(caution_signal, 1.0 - resolution_signal)
            ranked.append((task_id, spec, score))
        ranked.sort(key=lambda item: item[2], reverse=True)

        selected = []
        deferred = []
        pruned = []
        for task_id, spec, score in ranked:
            risk = _task_risk(spec, stimulus)
            if risk > risk_cutoff:
                pruned.append(self._selection(task_id, spec, stimulus, score, "pruned"))
            elif len(selected) < selection_budget and score >= confidence_threshold:
                selected.append(self._selection(task_id, spec, stimulus, score, "selected"))
            else:
                deferred.append(self._selection(task_id, spec, stimulus, score, "deferred"))

        if not selected and deferred:
            selected.append(deferred.pop(0))

        breadth = float(np.mean([item.breadth for item in selected])) if selected else 0.0
        depth = float(np.mean([item.depth for item in selected])) if selected else 0.0
        if state.G[G_TRANS] > state.G[G_IND]:
            breadth = float(np.clip(breadth + (0.20 * state.G[G_TRANS]), 0.0, 1.0))
        else:
            depth = float(np.clip(depth + (0.20 * resolution_signal), 0.0, 1.0))

        return InferencePlan(
            selected=tuple(selected),
            deferred=tuple(deferred),
            pruned=tuple(pruned),
            breadth=breadth,
            depth=depth,
            confidence_threshold=confidence_threshold,
        )
