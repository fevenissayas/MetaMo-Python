"""
Adapter layer that lets the use-case gridworld agent reason with the root MetaMo pseudo-bimonad.
"""

import numpy as np
from typing import List, Optional

from core.state import MotivationalState, Stimulus, Action
from core.config import (
    NUM_GOALS,
    G_IND,
    G_TRANS,
    G_HELP,
    G_CURIO,
    G_NOVEL,
    G_SELF,
    G_ETHIC,
    G_SOC,
    M_APPROACH,
    M_AROUSAL,
    M_THRESHOLD,
)
from openpsi.appraisal import OpenPsiAppraisal
from magus.decision import MagusDecision
from category.bimonad import MetaMoPseudoBimonad
from dynamics.stability import is_in_safe_region as canonical_is_in_safe_region

LAVA_CELLS = [(8, 8), (8, 9), (9, 8), (9, 9)]
ACTION_IDS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())


def in_safe_region(mot_state: MotivationalState) -> bool:
    """Delegate the use-case safety check to canonical MetaMo dynamics."""
    return bool(canonical_is_in_safe_region(mot_state))


def energy_drive(mot_state: MotivationalState) -> float:
    """Energy drive proxy for the gridworld dashboard."""
    return float(mot_state.M[M_APPROACH])


def safety_threshold(mot_state: MotivationalState) -> float:
    """Safety threshold proxy for the gridworld dashboard."""
    return float(mot_state.M[M_THRESHOLD])


def arousal(mot_state: MotivationalState) -> float:
    """Arousal proxy for the gridworld dashboard."""
    return float(mot_state.M[M_AROUSAL])


def _l1_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return the Manhattan distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _lava_cells(env_state: dict) -> tuple[tuple[int, int], ...]:
    """Extract the lava cell locations from an environment state."""
    return tuple(env_state.get("lava_cells", LAVA_CELLS))


def _distance_to_lava(pos: tuple[int, int], lava_cells: tuple[tuple[int, int], ...]) -> int:
    """Return the distance from a position to the nearest lava cell."""
    return min(_l1_distance(pos, lava) for lava in lava_cells)


def _is_lava_cell(pos: tuple[int, int], lava_cells: tuple[tuple[int, int], ...]) -> bool:
    """Return whether a position is occupied by lava."""
    return pos in lava_cells


def _project_move(env_state: dict, action: int) -> tuple[int, int]:
    """Project the agent's next position for a candidate action."""
    row, col = env_state["pos"]
    dr, dc = DELTAS[action]
    nr, nc = row + dr, col + dc
    if 0 <= nr < 10 and 0 <= nc < 10:
        return (nr, nc)
    return (row, col)



def build_stimulus(env_state: dict, mot_state: Optional[MotivationalState] = None) -> Stimulus:
    """
    Build the MetaMo appraisal stimulus from the current environment state.
    """
    distance = abs(env_state["dx_mineral"]) + abs(env_state["dy_mineral"])
    lava_dist = env_state.get("lava_distance", _distance_to_lava(env_state["pos"], _lava_cells(env_state)))
    risk = 1.0 if env_state["in_lava"] else float(np.clip(0.60 - lava_dist * 0.16, 0.0, 1.0))
    conduciveness = float(np.clip(1.0 - distance / 18.0, 0.0, 1.0))
    novelty = float(np.clip(0.30 + 0.70 * (1.0 - distance / 18.0), 0.0, 1.0))
    effort = float(np.clip(0.10 + 0.40 * (risk + distance / 18.0) / 2.0, 0.0, 1.0))

    return Stimulus(novelty=novelty, conduciveness=conduciveness, risk=risk, effort=effort)


def _make_local_candidate(env_state: dict, action: int) -> Action:
    """
    Construct a MetaMo action candidate describing one possible movement.
    """
    lava_cells = _lava_cells(env_state)
    next_pos = _project_move(env_state, action)
    dist_now = abs(env_state["dx_mineral"]) + abs(env_state["dy_mineral"])
    next_dist = abs(next_pos[0] - env_state["mineral_pos"][0]) + abs(next_pos[1] - env_state["mineral_pos"][1])
    toward_mineral = next_dist < dist_now
    next_lava_dist = _distance_to_lava(next_pos, lava_cells)
    current_lava_dist = env_state.get("lava_distance", _distance_to_lava(env_state["pos"], lava_cells))

    if _is_lava_cell(next_pos, lava_cells):
        risk_estimate = 1.0
    elif next_lava_dist <= 1:
        risk_estimate = 0.55
    elif next_lava_dist <= 2:
        risk_estimate = 0.25
    else:
        risk_estimate = 0.05

    goal_correlations = np.zeros(NUM_GOALS, dtype=float)
    goal_correlations[G_IND] = 0.25 + 0.25 * float(not _is_lava_cell(next_pos, lava_cells))
    goal_correlations[G_TRANS] = 0.35 + 0.25 * float(toward_mineral)
    goal_correlations[G_HELP] = 0.40 + 0.25 * float(toward_mineral)
    goal_correlations[G_CURIO] = 0.45 * float(toward_mineral) + 0.20
    goal_correlations[G_NOVEL] = 0.40 * float(toward_mineral) + 0.20
    goal_correlations[G_SELF] = 0.25 + 0.20 * float(not _is_lava_cell(next_pos, lava_cells))
    goal_correlations[G_ETHIC] = 0.30 + 0.40 * float(next_lava_dist >= current_lava_dist)
    goal_correlations[G_SOC] = 0.15 + 0.10 * float(toward_mineral)

    delta_g = np.zeros(NUM_GOALS, dtype=float)
    if toward_mineral:
        delta_g[G_CURIO] += 0.04
        delta_g[G_NOVEL] += 0.04
    if next_lava_dist > current_lava_dist:
        delta_g[G_IND] += 0.03
        delta_g[G_ETHIC] += 0.03
    if _is_lava_cell(next_pos, lava_cells):
        delta_g[G_IND] -= 0.05
        delta_g[G_ETHIC] += 0.05

    delta_g = np.clip(delta_g, -0.05, 0.05)
    return Action(
        id=ACTION_IDS[action],
        goal_correlations=goal_correlations,
        risk_estimate=float(risk_estimate),
        delta_g=delta_g,
    )


def build_candidates(env_state: dict, mot_state: Optional[MotivationalState] = None) -> List[Action]:
    """
    Generate motivational action candidates for all movement directions.
    """
    return [_make_local_candidate(env_state, action) for action in range(len(DELTAS))]


def build_consensus_states(
    env_state: dict,
    mot_state: MotivationalState,
    stimulus: Stimulus,
) -> tuple[MotivationalState, MotivationalState]:
    """
    Split the current motivation into safety and growth perspectives, then let
    consensus_transition merge them through the shared bimonad machinery.
    """
    risk = float(stimulus.risk)
    opportunity = float((stimulus.novelty + stimulus.conduciveness) / 2.0)
    near_lava = float(env_state["lava_distance"] <= 2)

    safety_state = mot_state.copy()
    safety_state.G[G_IND] = np.clip(safety_state.G[G_IND] + 0.10 * risk + 0.03 * near_lava, 0.0, 1.0)
    safety_state.G[G_ETHIC] = np.clip(safety_state.G[G_ETHIC] + 0.08 * risk + 0.02 * near_lava, 0.0, 1.0)
    safety_state.G[G_TRANS] = np.clip(safety_state.G[G_TRANS] - 0.04 * risk, 0.0, 1.0)

    growth_state = mot_state.copy()
    growth_state.G[G_TRANS] = np.clip(growth_state.G[G_TRANS] + 0.18 * opportunity, 0.0, 1.0)
    growth_state.G[G_CURIO] = np.clip(growth_state.G[G_CURIO] + 0.14 * opportunity, 0.0, 1.0)
    growth_state.G[G_NOVEL] = np.clip(growth_state.G[G_NOVEL] + 0.14 * opportunity, 0.0, 1.0)
    growth_state.G[G_IND] = np.clip(growth_state.G[G_IND] + 0.04 * risk, 0.0, 1.0)

    return safety_state, growth_state


def consensus_candidate_scores(
    mot_state: MotivationalState,
    stimulus: Stimulus,
    candidates: List[Action],
    env_state: Optional[dict] = None,
) -> np.ndarray:
    """Score actions from the same two-perspective consensus used for transition."""
    if env_state is None:
        state_a = mot_state
        state_b = mot_state
    else:
        state_a, state_b = build_consensus_states(
            env_state, mot_state, stimulus
        )
    return np.asarray(
        bimonad.consensus_scores(state_a, state_b, stimulus, candidates),
        dtype=float,
    )


def transition_for_action(
    env_state: dict,
    mot_state: MotivationalState,
    action_idx: int,
    stimulus: Optional[Stimulus] = None,
    candidates: Optional[List[Action]] = None,
) -> tuple[Action, MotivationalState, Stimulus, MotivationalState]:
    """Apply a selected grid action through bimonad.consensus_transition."""
    stimulus = stimulus or build_stimulus(env_state, mot_state)
    candidates = candidates or build_candidates(env_state, mot_state)
    selected = [candidates[action_idx]]
    safety_state, growth_state = build_consensus_states(env_state, mot_state, stimulus)
    transition = bimonad.complete_consensus_transition(
        previous_state=mot_state,
        state_a=safety_state,
        state_b=growth_state,
        stimulus=stimulus,
        candidates=selected,
    )
    return (
        transition.action,
        transition.next_state,
        stimulus,
        transition.projected_target,
    )


def choose_action(env_state: dict, mot_state: MotivationalState) -> tuple[Action, MotivationalState, Stimulus]:
    """
    Select an action using the MetaMo consensus decision process.
    """
    stimulus = build_stimulus(env_state, mot_state)
    candidates = build_candidates(env_state, mot_state)
    safety_state, growth_state = build_consensus_states(env_state, mot_state, stimulus)
    transition = bimonad.complete_consensus_transition(
        previous_state=mot_state,
        state_a=safety_state,
        state_b=growth_state,
        stimulus=stimulus,
        candidates=candidates,
    )
    return transition.action, transition.next_state, stimulus

def is_safe_motivational_state(mot_state: MotivationalState) -> bool:
    """Return whether the motivational state lies within the MetaMo safe region."""
    return in_safe_region(mot_state)

