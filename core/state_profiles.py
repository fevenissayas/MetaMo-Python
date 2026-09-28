"""Application-neutral constructors for named MetaMo motivational profiles."""

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
    NUM_GOALS,
    NUM_MODULATORS,
)
from core.state import MotivationalState

# Reference profile used by the use-case and as the MDRL preference anchor.
REFERENCE_GOAL_PROFILE = (
    0.65,  # individuation
    0.55,  # transcendence
    0.75,  # helpfulness
    0.50,  # curiosity
    0.45,  # novelty
    0.30,  # self-development
    0.85,  # ethics
    0.20,  # social engagement
)
REFERENCE_MODULATOR_LEVEL = 0.5

def create_reference_motivational_state() -> MotivationalState:
    """Return a fresh copy of the shared gridworld/MDRL reference profile."""
    goals = np.zeros(NUM_GOALS, dtype=float)
    goals[G_IND] = REFERENCE_GOAL_PROFILE[G_IND]
    goals[G_TRANS] = REFERENCE_GOAL_PROFILE[G_TRANS]
    goals[G_HELP] = REFERENCE_GOAL_PROFILE[G_HELP]
    goals[G_CURIO] = REFERENCE_GOAL_PROFILE[G_CURIO]
    goals[G_NOVEL] = REFERENCE_GOAL_PROFILE[G_NOVEL]
    goals[G_SELF] = REFERENCE_GOAL_PROFILE[G_SELF]
    goals[G_ETHIC] = REFERENCE_GOAL_PROFILE[G_ETHIC]
    goals[G_SOC] = REFERENCE_GOAL_PROFILE[G_SOC]
    modulators = np.full(
        NUM_MODULATORS, REFERENCE_MODULATOR_LEVEL, dtype=float
    )
    return MotivationalState(G=goals, M=modulators)
