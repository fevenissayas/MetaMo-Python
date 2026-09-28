"""Per-step record and the decision-context width shared by the agent."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from core.config import NUM_GOALS, NUM_MODULATORS
from mdrl.config import NUM_OUTCOMES
from mdrl.curiosity.module import CURIOSITY_FEATURE_DIM
from mdrl.envs.curious_gridworld import OBSERVATION_DIM

CONTEXT_DIM = (
    OBSERVATION_DIM + NUM_GOALS + NUM_MODULATORS + NUM_OUTCOMES + 4 + CURIOSITY_FEATURE_DIM
)

DIAGNOSTIC_PERIOD = 10  # how often the expensive stability probes run

@dataclass
class StepRecord:
    """Everything one decision step contributes to the Section 7.6 metrics."""

    candidate_id: str
    candidate_kind: str
    certified: bool
    duration: int
    outcome: np.ndarray
    lp_reward: float
    safety_cost: float
    weights: np.ndarray
    beta: float
    zone: str
    in_lava: bool
    lava_distance: int
    energy: float
    exploratory: bool
    # Gap between the chosen candidate's predicted effect and the best available one.
    descriptor_regret: float = 0.0
    events: List[str] = field(default_factory=list)
