"""Evaluation regimes and the motive profiles sampled in training and testing."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.config import (
    G_CURIO,
    G_ETHIC,
    G_HELP,
    G_IND,
    G_NOVEL,
    G_SELF,
    G_TRANS,
    M_APPROACH,
    M_AROUSAL,
    M_SECURING,
    M_THRESHOLD,
)
from core.state import MotivationalState
from dynamics.stability import project_to_safe_region
from mdrl.candidates.descriptors import SkillClass
from mdrl.candidates.skills import (
    HELDOUT_SKILL_CLASSES,
    default_skill_library,
)
from mdrl.envs.curious_gridworld import LayoutVariant, MotiveIntervention
from core.state_profiles import create_reference_motivational_state

@dataclass(frozen=True)
class MotiveProfile:
    """A named displacement of the motivational state."""

    name: str
    goal_deltas: Dict[int, float]
    modulator_deltas: Dict[int, float]

    def apply(self, base: MotivationalState) -> MotivationalState:
        state = base.copy()
        for index, delta in self.goal_deltas.items():
            state.G[index] = float(np.clip(state.G[index] + delta, 0.0, 1.0))
        for index, delta in self.modulator_deltas.items():
            state.M[index] = float(np.clip(state.M[index] + delta, 0.0, 1.0))
        return project_to_safe_region(state)

    def as_intervention(self, step: int) -> MotiveIntervention:
        return MotiveIntervention(
            step=step,
            label=self.name,
            goal_deltas=dict(self.goal_deltas),
            modulator_deltas=dict(self.modulator_deltas),
        )

# Profiles outside the dense training support, for the out-of-support regime.
SAFETY_PROFILE = MotiveProfile(
    name="safety",
    goal_deltas={G_IND: 0.30, G_ETHIC: 0.12, G_TRANS: -0.30, G_CURIO: -0.35, G_NOVEL: -0.30},
    modulator_deltas={M_SECURING: 0.35, M_THRESHOLD: 0.35, M_APPROACH: -0.25},
)

CURIOSITY_PROFILE = MotiveProfile(
    name="curiosity",
    goal_deltas={G_CURIO: 0.40, G_NOVEL: 0.35, G_TRANS: 0.35, G_IND: -0.28},
    modulator_deltas={M_AROUSAL: 0.30, M_APPROACH: 0.30, M_SECURING: -0.25},
)

RESOURCE_PROFILE = MotiveProfile(
    name="resource",
    goal_deltas={G_SELF: 0.45, G_HELP: -0.20, G_CURIO: -0.20},
    modulator_deltas={M_SECURING: 0.15},
)

TASK_PROFILE = MotiveProfile(
    name="task",
    goal_deltas={G_HELP: 0.20, G_CURIO: -0.25, G_NOVEL: -0.25, G_SELF: -0.15},
    modulator_deltas={},
)

EXTREME_PROFILES: Tuple[MotiveProfile, ...] = (
    SAFETY_PROFILE,
    CURIOSITY_PROFILE,
    RESOURCE_PROFILE,
    TASK_PROFILE,
)

def sample_training_motive(rng: np.random.Generator) -> MotivationalState:
    """
    Draw an initial motivational state from the dense training support.

    Moderate perturbations only: the extreme profiles are held out so that
    evaluation regime 4 tests motive vectors the learner has genuinely not seen.
    """
    state = create_reference_motivational_state()
    state.G = np.clip(state.G + rng.uniform(-0.15, 0.15, size=state.G.shape), 0.0, 1.0)
    state.M = np.clip(state.M + rng.uniform(-0.10, 0.10, size=state.M.shape), 0.0, 1.0)
    return project_to_safe_region(state)

def sample_unseen_motive(rng: np.random.Generator) -> MotivationalState:
    """A fresh draw from the same support: unseen vectors, in distribution."""
    return sample_training_motive(rng)

def training_interventions(
    rng: np.random.Generator, max_steps: int, probability: float = 0.5
) -> List[MotiveIntervention]:
    """
    Occasionally shift the motive mid-episode during training.

    Without this the network would only ever see motive trajectories generated
    by appraisal, and H2 would be untestable.
    """
    if rng.random() >= probability:
        return []
    profile = EXTREME_PROFILES[int(rng.integers(len(EXTREME_PROFILES)))]
    step = int(rng.integers(max_steps // 4, max(max_steps // 4 + 1, 3 * max_steps // 4)))
    # Half strength during training; evaluation uses the full profile.
    scaled = MotiveProfile(
        name=profile.name,
        goal_deltas={index: 0.5 * delta for index, delta in profile.goal_deltas.items()},
        modulator_deltas={
            index: 0.5 * delta for index, delta in profile.modulator_deltas.items()
        },
    )
    return [scaled.as_intervention(step)]

def training_skill_library(progress_probe=None):
    """Library used during training: the held-out skills are withheld."""
    return [
        skill
        for skill in default_skill_library(progress_probe)
        if skill.skill_class not in HELDOUT_SKILL_CLASSES
    ]

def full_skill_library(progress_probe=None):
    """Evaluation library including skills never seen at training time."""
    return default_skill_library(progress_probe)

def restricted_skill_library(progress_probe=None):
    """A different subset again, to test new combinations of known features."""
    keep = {
        SkillClass.GOTO_MINERAL_SAFE,
        SkillClass.INSPECT_LEARNABLE,
        SkillClass.RETREAT_FROM_LAVA,
    }
    return [
        skill for skill in default_skill_library(progress_probe) if skill.skill_class in keep
    ]

@dataclass(frozen=True)
class EvaluationRegime:
    """One row of the Section 7.7 evaluation schedule."""

    name: str
    description: str
    layouts: Tuple[LayoutVariant, ...]
    motive: str = "training"          # "training" | "unseen" | profile name
    skills: str = "training"          # "training" | "full" | "restricted"
    intervention_step: Optional[int] = None
    intervention_profile: Optional[MotiveProfile] = None
    noisy_zone: bool = True

def default_regimes(max_steps: int) -> Tuple[EvaluationRegime, ...]:
    train_maps = LayoutVariant.train_variants()
    heldout_maps = LayoutVariant.heldout_variants()
    switch_step = max_steps // 3

    return (
        EvaluationRegime(
            "in_distribution",
            "Training maps and training motive distribution.",
            train_maps,
        ),
        EvaluationRegime(
            "heldout_maps",
            "Maps never seen during training; candidate set unchanged.",
            heldout_maps,
        ),
        EvaluationRegime(
            "unseen_motive",
            "Fresh motive vectors from inside the training support.",
            train_maps,
            motive="unseen",
        ),
        EvaluationRegime(
            "far_motive_safety",
            "Safety-dominant motive outside the dense training support.",
            train_maps,
            motive="safety",
        ),
        EvaluationRegime(
            "far_motive_curiosity",
            "Curiosity-dominant motive outside the dense training support.",
            train_maps,
            motive="curiosity",
        ),
        EvaluationRegime(
            "new_candidates",
            "Skills withheld during training are introduced at evaluation.",
            train_maps,
            skills="full",
        ),
        EvaluationRegime(
            "candidate_subset",
            "A restricted candidate subset, testing new feature combinations.",
            train_maps,
            skills="restricted",
        ),
        EvaluationRegime(
            "abrupt_switch_safety",
            "Abrupt mid-episode switch to a safety-dominant motive.",
            train_maps,
            intervention_step=switch_step,
            intervention_profile=SAFETY_PROFILE,
        ),
        EvaluationRegime(
            "abrupt_switch_curiosity",
            "Abrupt mid-episode switch to a curiosity-dominant motive.",
            train_maps,
            intervention_step=switch_step,
            intervention_profile=CURIOSITY_PROFILE,
        ),
        EvaluationRegime(
            "deterministic",
            "Noisy zone disabled: the deterministic environment variant.",
            train_maps,
            noisy_zone=False,
        ),
    )
