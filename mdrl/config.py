"""Constants and `ConditionSpec`, the flags that define each experimental condition."""

from dataclasses import dataclass, replace
from typing import Tuple

# Outcome components stay separate so a new preference can reweight them.
OUTCOME_COMPONENTS: Tuple[str, ...] = ("task", "safety", "resource", "info")
NUM_OUTCOMES = len(OUTCOME_COMPONENTS)

OUT_TASK, OUT_SAFETY, OUT_RESOURCE, OUT_INFO = range(NUM_OUTCOMES)

# Fixed weights for cross-condition scores. Not the agent's own preference.
EXTERNAL_EVALUATION_WEIGHTS: Tuple[float, ...] = (0.40, 0.25, 0.20, 0.15)

# Unknown descriptor schemas fail closed.
DESCRIPTOR_SCHEMA_VERSION = "phi-v1"

LP_CLIP = 1.0
LP_WINDOW = 16
KAPPA_SAFETY_COST = 0.5

# Charged against the safety outcome so the learner cannot hide behind projection.
PROJECTION_PENALTY = 2.0

@dataclass(frozen=True)
class ConditionSpec:
    """One experimental combination, selected by the string flags on this class."""

    name: str
    label: str

    appraisal: str = "fixed"
    decision: str = "candidate_dqn"
    value_head: str = "vector"
    candidates: str = "primitive"
    certificate_gate: bool = True
    motive_in_context: bool = True
    dynamic_weights: bool = True
    curiosity: str = "lp_h"
    goal_update: str = "fixed"
    stabilizer: bool = True
    appraisal_signal: str = "none"
    training_stage: str = "joint"
    base_condition: str = ""
    notes: str = ""

    @property
    def uses_metamo(self) -> bool:
        return self.appraisal != "none"

    @property
    def uses_learned_decision(self) -> bool:
        return self.decision != "magus"

    @property
    def uses_skills(self) -> bool:
        return self.candidates == "primitive_plus_skills"

    @property
    def uses_curiosity(self) -> bool:
        return self.curiosity != "none"

    @property
    def uses_vector_values(self) -> bool:
        return self.value_head == "vector"

    def variant(self, name: str, label: str, **overrides) -> "ConditionSpec":
        """Derive an experimental variant by overriding selected flags."""
        return replace(self, name=name, label=label, **overrides)

    def describe(self) -> str:
        return (
            f"{self.label} | appraisal={self.appraisal} decision={self.decision} "
            f"value={self.value_head} candidates={self.candidates} "
            f"curiosity={self.curiosity} goal_update={self.goal_update} "
            f"appraisal_signal={self.appraisal_signal} stage={self.training_stage} "
            f"cert_gate={self.certificate_gate} motive_ctx={self.motive_in_context} "
            f"dyn_w={self.dynamic_weights} stabilizer={self.stabilizer}"
        )

@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters shared by every learned condition so ablations stay matched."""

    train_episodes: int = 300
    eval_episodes: int = 12
    max_steps: int = 120
    appraisal_train_episodes: int = 120
    appraisal_joint_finetune_episodes: int = 0

    gamma: float = 0.97
    learning_rate: float = 7e-4
    batch_size: int = 64
    buffer_size: int = 40_000
    min_buffer_size: int = 600
    target_update_period: int = 250
    grad_clip: float = 5.0

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.975

    hidden_dim: int = 128
    context_dim: int = 96
    candidate_dim: int = 64

    world_model_lr: float = 1e-3
    error_model_lr: float = 5e-4

    # ICM: forward-loss weight and reward scale.
    icm_feature_dim: int = 32
    icm_forward_loss_weight: float = 0.2
    icm_reward_scale: float = 1.0

    # Replay fraction rescaled under a new preference. Outcomes themselves stay fixed.
    relabel_fraction: float = 0.25
    # Fraction drawn from motive-stratified strata rather than uniformly.
    stratified_fraction: float = 0.5

    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
