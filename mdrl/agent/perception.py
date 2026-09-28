"""Stimulus, appraisal context, candidates, and selection."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from category.functors import AppraisalContext
from core.state import MotivationalState, Stimulus
from mdrl.curiosity.module import CURIOSITY_FEATURE_DIM
from mdrl.envs.curious_gridworld import GRID_SIZE, START_ENERGY, Zone
from mdrl.types import Candidate, CandidateKind, DecisionContext

class AgentPerception:
    def build_stimulus(self, observation: Dict[str, Any]) -> Stimulus:
        """Appraisal feedback: novelty, conduciveness, risk, and effort."""
        distance = abs(observation["dx_mineral"]) + abs(observation["dy_mineral"])
        lava_distance = int(observation.get("lava_distance", GRID_SIZE))
        risk = (
            1.0
            if observation["in_lava"]
            else float(np.clip(0.60 - lava_distance * 0.16, 0.0, 1.0))
        )
        conduciveness = float(np.clip(1.0 - distance / (2.0 * GRID_SIZE), 0.0, 1.0))
        structured = observation.get("zone", Zone.NONE) is not Zone.NONE
        model_error = float(self.curiosity.features()[0])
        novelty = float(np.clip(0.15 + 0.45 * structured + 0.40 * model_error, 0.0, 1.0))
        energy_fraction = float(observation.get("energy", START_ENERGY)) / START_ENERGY
        effort = float(
            np.clip(
                0.10
                + 0.40 * (risk + distance / (2.0 * GRID_SIZE)) / 2.0
                + 0.30 * (1.0 - energy_fraction),
                0.0,
                1.0,
            )
        )
        return Stimulus(
            novelty=novelty, conduciveness=conduciveness, risk=risk, effort=effort
        )

    def appraise(
        self,
        state: MotivationalState,
        observation: Dict[str, Any],
        stimulus: Stimulus,
        context: Optional[AppraisalContext] = None,
        learning: bool = False,
    ) -> MotivationalState:
        """Psi followed by boundary-sensitive caution."""
        if not self.condition.uses_metamo:
            # No motivational layer at all: the state is inert.
            return state.copy()
        context = context or self.build_appraisal_context(observation, learning)
        return self.metamo.appraise(state, stimulus, context)

    def build_appraisal_context(
        self, observation: Dict[str, Any], learning: bool = False
    ) -> AppraisalContext:
        """Build the explicit, dependency-neutral context consumed by Psi."""
        energy_fraction = float(observation.get("energy", START_ENERGY)) / START_ENERGY
        appraisal_curiosity = self.active_appraisal_curiosity()
        return AppraisalContext(
            observation=np.asarray(observation["features"], dtype=np.float32).copy(),
            curiosity_features=(
                np.zeros(CURIOSITY_FEATURE_DIM, dtype=np.float32)
                if appraisal_curiosity is None
                else appraisal_curiosity.features(energy_fraction).copy()
            ),
            learning=learning,
            metadata={"step": self._step_index},
        )

    def build_context(
        self,
        observation: Dict[str, Any],
        appraised: MotivationalState,
        stimulus: Stimulus,
    ) -> DecisionContext:
        energy_fraction = float(observation.get("energy", START_ENERGY)) / START_ENERGY
        return DecisionContext(
            observation=observation["features"],
            raw_observation=observation,
            state=appraised,
            feedback=stimulus,
            active_goal=self.preference.active_goal(appraised),
            curiosity_features=self.curiosity.features(energy_fraction),
            weights=self.preference.weights(appraised),
            beta=self.preference.beta(appraised) if self.condition.uses_curiosity else 0.0,
            step=self._step_index,
        )

    def candidates_for(
        self, observation: Dict[str, Any], appraised: MotivationalState
    ) -> List[Candidate]:
        candidates = self.adapter.valid_candidates(observation, appraised)
        if self.condition.decision == "fixed_output_dqn":
            # A fixed output head cannot represent a variable candidate set.
            candidates = [c for c in candidates if c.kind is CandidateKind.PRIMITIVE]
        return candidates

    def select(
        self, context: DecisionContext, candidates: Sequence[Candidate]
    ) -> Tuple[Candidate, np.ndarray, bool]:
        """Epsilon-greedy over C^+_t, equation (20)."""
        if self.condition.decision == "magus":
            if self.rng.random() < self.epsilon:
                index = int(self.rng.integers(len(candidates)))
                chosen = candidates[index]
                return chosen, chosen.delta_g.copy(), True
            decision = self.metamo.decide(
                context.state, list(candidates), context
            )
            return decision.action, decision.proposed_delta_g, False

        self.decision.epsilon = self.epsilon
        decision = self.metamo.decide(
            context.state, list(candidates), context
        )
        return (
            decision.action,
            decision.proposed_delta_g,
            self.decision.last_exploratory,
        )

