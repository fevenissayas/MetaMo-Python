"""Shared containers. `Candidate` subclasses `core.state.Action` so MAGUS and DQN score the same objects."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np

from core.state import Action, MotivationalState, Stimulus
from mdrl.config import DESCRIPTOR_SCHEMA_VERSION, NUM_OUTCOMES

class CandidateKind(str, Enum):
    """Primitive moves are always available; skills and subgoals are proposed."""

    PRIMITIVE = "primitive"
    SKILL = "skill"
    SUBGOAL = "subgoal"

@dataclass
class Certificate:
    """Admission record for a skill. A schema or policy-hash mismatch rejects it."""

    skill_id: str
    admitted: bool = True
    schema_version: str = DESCRIPTOR_SCHEMA_VERSION
    policy_hash: str = "static-v1"
    # Motive-context support region: goal index -> (low, high).
    goal_support: Dict[int, tuple] = field(default_factory=dict)
    # Observation-level support conditions, evaluated by the adapter.
    support_notes: str = ""

    def is_valid(self, state: MotivationalState, schema_version: str, policy_hash: str) -> bool:
        """Evaluate Valid(chi_c, x~) from equation (11)."""
        if not self.admitted:
            return False
        if self.schema_version != schema_version or self.policy_hash != policy_hash:
            return False
        for goal_idx, (low, high) in self.goal_support.items():
            if not (low <= float(state.G[goal_idx]) <= high):
                return False
        return True

@dataclass
class Candidate(Action):
    """
    A candidate c = (id_c, phi(c), I_c, beta_c, mu_hat_c, chi_c) from equation (10).

    Inherits `id`, `goal_correlations`, `risk_estimate`, and `delta_g` from
    `Action` so the MetaMo bimonad and MAGUS decision monad accept it unchanged.
    """

    kind: CandidateKind = CandidateKind.PRIMITIVE
    descriptor: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    # mu_hat_c: predicted vector of motive-relevant effects, in outcome space.
    mu_hat: np.ndarray = field(default_factory=lambda: np.zeros(NUM_OUTCOMES, dtype=np.float32))
    certificate: Optional[Certificate] = None
    expected_duration: float = 1.0
    safety_cost: float = 0.0
    # I_c: initiation predicate, evaluated against the raw observation.
    initiation: Optional[Callable[[Dict[str, Any]], bool]] = None
    # beta_c: termination predicate for temporally extended skills.
    termination: Optional[Callable[[Dict[str, Any], int], bool]] = None
    # Primitive action index this candidate expands to, when it is a primitive.
    primitive_index: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.mu_hat.shape[0] != NUM_OUTCOMES:
            raise ValueError(f"mu_hat must have length {NUM_OUTCOMES}")

    @property
    def is_certified(self) -> bool:
        return self.certificate is not None and self.certificate.admitted

    def can_initiate(self, observation: Dict[str, Any]) -> bool:
        """Evaluate I_c(o_t) from equation (11)."""
        if self.initiation is None:
            return True
        return bool(self.initiation(observation))

    def should_terminate(self, observation: Dict[str, Any], elapsed: int) -> bool:
        """Evaluate beta_c for a temporally extended skill."""
        if self.termination is None:
            return True
        return bool(self.termination(observation, elapsed))

@dataclass
class DecisionContext:
    """
    The typed context recommended in Section 6.2.

    Carrying an explicit object avoids turning the appraisal feedback dictionary
    into an undocumented transport channel for observations and curiosity state.
    """

    observation: np.ndarray            # e_o(o_t) input features
    raw_observation: Dict[str, Any]
    state: MotivationalState           # x~_t, post-appraisal
    feedback: Stimulus                 # f_t
    active_goal: np.ndarray            # e_g(g*_t)
    curiosity_features: np.ndarray     # u_t
    weights: np.ndarray                # w_t, equation (17)
    beta: float                        # beta_t, equation (18)
    step: int = 0

    def context_vector(self, include_motive: bool) -> np.ndarray:
        """Assemble z_t. A dropped motive is zeroed so network width stays matched."""
        motive = np.concatenate([self.state.G, self.state.M]).astype(np.float32)
        if not include_motive:
            motive = np.zeros_like(motive)
        feedback = np.array(
            [
                self.feedback.novelty,
                self.feedback.conduciveness,
                self.feedback.risk,
                self.feedback.effort,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                self.observation.astype(np.float32),
                motive,
                self.active_goal.astype(np.float32),
                feedback,
                self.curiosity_features.astype(np.float32),
            ]
        )

@dataclass
class Transition:
    """Replay record: vector outcome plus the motive weights that produced the behaviour."""

    context: np.ndarray                # z_t
    descriptor: np.ndarray             # phi(c_t)
    goal_vector: np.ndarray            # G_t
    modulator_vector: np.ndarray       # M~_t
    weights: np.ndarray                # w_t
    beta: float                        # beta_t
    outcome: np.ndarray                # r^task_t in R^K
    lp_reward: float                   # r^LP_t
    safety_cost: float                 # c^safety_t
    next_context: np.ndarray           # z_{t+tau}
    next_descriptors: np.ndarray       # phi over C^+_{t+tau}
    next_weights: np.ndarray           # w_{t+tau}
    next_beta: float
    next_safety_costs: np.ndarray
    done: bool
    duration: int = 1                  # tau_t for semi-Markov returns
    certified: bool = True
    projection_magnitude: float = 0.0  # feeds the Section 8.4 penalty

@dataclass
class StepDiagnostics:
    """Per-step MetaMo stability record used by the Section 7.6 metrics."""

    raw_delta_g: np.ndarray
    accepted_delta_g: np.ndarray
    pre_projection_safe: bool
    post_projection_safe: bool
    projection_magnitude: float
    boundary_pressure: float
    contraction_ratio: Optional[float]
    lax_distributive_error: float
    self_model_drift: float
    blend_alpha: float
