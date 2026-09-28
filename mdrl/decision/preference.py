"""Fixed map from (G, M) to outcome weights and a curiosity weight."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.config import (
    G_CURIO,
    G_ETHIC,
    G_HELP,
    G_IND,
    G_NOVEL,
    G_SELF,
    G_SOC,
    G_TRANS,
    M_APPROACH,
    M_AROUSAL,
    M_RESOLUTION,
    M_SECURING,
    M_THRESHOLD,
    M_VALENCE,
    NUM_GOALS,
    NUM_MODULATORS,
)
from core.state import MotivationalState
from core.state_profiles import create_reference_motivational_state
from mdrl.config import (
    EXTERNAL_EVALUATION_WEIGHTS,
    NUM_OUTCOMES,
    OUT_INFO,
    OUT_RESOURCE,
    OUT_SAFETY,
    OUT_TASK,
)

def _goal_matrix() -> np.ndarray:
    """W_G: rows are outcome components, columns are MetaMo goals."""
    matrix = np.zeros((NUM_OUTCOMES, NUM_GOALS), dtype=np.float32)

    matrix[OUT_TASK, G_HELP] = 2.0
    matrix[OUT_TASK, G_TRANS] = 0.6
    matrix[OUT_TASK, G_SOC] = 0.3
    matrix[OUT_TASK, G_IND] = -0.2

    matrix[OUT_SAFETY, G_IND] = 2.2
    matrix[OUT_SAFETY, G_ETHIC] = 1.6
    matrix[OUT_SAFETY, G_TRANS] = -0.8

    matrix[OUT_RESOURCE, G_SELF] = 1.8
    matrix[OUT_RESOURCE, G_IND] = 0.4

    matrix[OUT_INFO, G_CURIO] = 2.0
    matrix[OUT_INFO, G_NOVEL] = 1.6
    matrix[OUT_INFO, G_TRANS] = 0.8
    matrix[OUT_INFO, G_IND] = -0.4
    return matrix

def _modulator_matrix() -> np.ndarray:
    """W_M: rows are outcome components, columns are OpenPsi modulators."""
    matrix = np.zeros((NUM_OUTCOMES, NUM_MODULATORS), dtype=np.float32)

    matrix[OUT_TASK, M_RESOLUTION] = 0.8
    matrix[OUT_TASK, M_APPROACH] = 0.4

    matrix[OUT_SAFETY, M_THRESHOLD] = 1.0
    matrix[OUT_SAFETY, M_SECURING] = 1.2
    matrix[OUT_SAFETY, M_APPROACH] = -0.4

    matrix[OUT_RESOURCE, M_RESOLUTION] = 0.3
    matrix[OUT_RESOURCE, M_SECURING] = 0.5
    matrix[OUT_RESOURCE, M_VALENCE] = -0.3

    matrix[OUT_INFO, M_AROUSAL] = 1.0
    matrix[OUT_INFO, M_APPROACH] = 0.8
    matrix[OUT_INFO, M_SECURING] = -0.6
    return matrix

# Nominal weights. Without this anchor, safety dominates and the default policy stops moving.
NOMINAL_PREFERENCE = np.asarray(EXTERNAL_EVALUATION_WEIGHTS, dtype=np.float32)

# Low temperature collapses weights toward one objective.
PREFERENCE_TEMPERATURE = 1.5

@dataclass(frozen=True)
class CuriosityWeightProfile:
    """Logged coefficients of the curiosity-weight formula."""

    # Per-step curiosity scale. Near 1 it would outweigh a one-time mineral reward.
    beta_0: float = 0.15
    bias: float = -0.5
    transcendence: float = 2.0
    curiosity: float = 2.0
    approach: float = 1.0
    individuation: float = 2.0
    securing: float = 1.0

class MotivePreference:
    """Computes w_t and beta_t from the appraised motivational state."""

    def __init__(
        self,
        dynamic: bool = True,
        reference_state: Optional[MotivationalState] = None,
        profile: Optional[CuriosityWeightProfile] = None,
        temperature: float = PREFERENCE_TEMPERATURE,
    ):
        self.dynamic = dynamic
        self.reference_state = reference_state
        self.profile = profile or CuriosityWeightProfile()
        self.temperature = temperature
        self.goal_matrix = _goal_matrix()
        self.modulator_matrix = _modulator_matrix()
        self.bias = self._calibrate()

    def _calibrate(self) -> np.ndarray:
        """Offset logits so the nominal state yields NOMINAL_PREFERENCE for every episode."""
        anchor = create_reference_motivational_state()
        logits = self.goal_matrix @ anchor.G.astype(np.float32) + (
            self.modulator_matrix @ anchor.M.astype(np.float32)
        )
        return (np.log(NOMINAL_PREFERENCE) * self.temperature - logits).astype(np.float32)

    def _effective(self, state: MotivationalState) -> MotivationalState:
        """Freezing the preference is how ablation 7.5.2 isolates conditioning."""
        if self.dynamic or self.reference_state is None:
            return state
        return self.reference_state

    def weights(self, state: MotivationalState) -> np.ndarray:
        """Equation (17)."""
        effective = self._effective(state)
        logits = (
            self.goal_matrix @ effective.G.astype(np.float32)
            + self.modulator_matrix @ effective.M.astype(np.float32)
            + self.bias
        )
        logits = logits / max(self.temperature, 1e-6)
        logits = logits - float(np.max(logits))
        exponentiated = np.exp(logits)
        return (exponentiated / float(np.sum(exponentiated))).astype(np.float32)

    def beta(self, state: MotivationalState) -> float:
        """Equation (18)."""
        effective = self._effective(state)
        profile = self.profile
        argument = (
            profile.bias
            + profile.transcendence * float(effective.G[G_TRANS])
            + profile.curiosity * float(effective.G[G_CURIO])
            + profile.approach * float(effective.M[M_APPROACH])
            - profile.individuation * float(effective.G[G_IND])
            - profile.securing * float(effective.M[M_SECURING])
        )
        return float(profile.beta_0 / (1.0 + np.exp(-argument)))

    def active_goal(self, state: MotivationalState) -> np.ndarray:
        """One-hot over the dominant outcome, separate from the continuous weights."""
        weights = self.weights(state)
        active = np.zeros(NUM_OUTCOMES, dtype=np.float32)
        active[int(np.argmax(weights))] = 1.0
        return active

    def active_goal_name(self, state: MotivationalState) -> str:
        from mdrl.config import OUTCOME_COMPONENTS

        return OUTCOME_COMPONENTS[int(np.argmax(self.weights(state)))]
