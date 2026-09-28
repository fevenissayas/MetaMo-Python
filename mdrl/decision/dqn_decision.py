"""Learned decision monad: the network picks a candidate, and MetaMo applies the goal-change rule."""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

from category.functors import DecisionMonad
from core.state import Action, MotivationalState
from mdrl.config import KAPPA_SAFETY_COST
from mdrl.types import Candidate, DecisionContext

class DQNDecisionMonad(DecisionMonad):
    """Scores candidates with a learned network and proposes Gamma_0's Delta G."""

    def __init__(
        self,
        network,
        include_motive_in_context: bool = True,
        vector_values: bool = True,
        kappa: float = KAPPA_SAFETY_COST,
        goal_residual_scale: float = 0.0,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.network = network
        self.include_motive_in_context = include_motive_in_context
        self.vector_values = vector_values
        self.kappa = kappa
        # alpha_G in equation (31); zero disables the Stage 5 residual head.
        self.goal_residual_scale = goal_residual_scale
        self.device = device or torch.device("cpu")
        self.rng = rng or np.random.default_rng(0)

        self.epsilon = 1.0
        self.last_scores: Optional[np.ndarray] = None
        self.last_q_task: Optional[np.ndarray] = None
        self.last_q_lp: Optional[np.ndarray] = None
        self.last_exploratory: bool = False

    def unit(self, state: MotivationalState) -> MotivationalState:
        """The monadic unit: inject the state without altering it."""
        return state

    def decide(
        self,
        state: MotivationalState,
        candidates: List[Action],
    ) -> Tuple[Action, np.ndarray]:
        """Reject context-free use: this learned monad needs neural context."""
        return self.decide_with_context(state, candidates, context=None)

    def decide_with_context(
        self,
        state: MotivationalState,
        candidates: List[Action],
        context: Optional[Any] = None,
    ) -> Tuple[Action, np.ndarray]:
        """Canonical contextual entry point for exploratory selection."""
        return self.select_action(state, candidates, context)

    def select_action(
        self,
        state: MotivationalState,
        candidates: List[Action],
        context: Optional[Any] = None,
    ) -> Tuple[Action, np.ndarray]:
        """Select one candidate; unlike evaluation, this may consume RNG."""
        if not candidates:
            raise ValueError("Must provide at least one candidate action to the decision monad.")
        context = self._require_context(context)

        scores, q_task, q_lp = self.evaluate(context, candidates)
        self.last_scores = scores.copy()
        self.last_q_task = q_task.copy()
        self.last_q_lp = q_lp.copy()
        if self.rng.random() < self.epsilon:
            index = int(self.rng.integers(len(candidates)))
            self.last_exploratory = True
        else:
            index = int(np.argmax(scores))
            self.last_exploratory = False

        chosen = candidates[index]
        return chosen, self.goal_update_for_candidate(
            state, chosen, index, context
        )

    def score_candidate(self, state: MotivationalState, candidate: Action) -> float:
        """Reject the legacy context-free scoring path."""
        raise RuntimeError("DQN candidate evaluation requires a DecisionContext")

    def evaluate_candidates(
        self,
        state: MotivationalState,
        candidates: List[Action],
        context: Optional[Any] = None,
    ) -> np.ndarray:
        """Deterministically score candidates without RNG or telemetry mutation."""
        context = self._require_context(context)
        scores, _, _ = self.evaluate(context, candidates)
        return scores

    @staticmethod
    def _require_context(context: Optional[Any]) -> DecisionContext:
        if not isinstance(context, DecisionContext):
            raise RuntimeError("DQNDecisionMonad requires an explicit DecisionContext")
        return context

    def score_all(
        self, context: DecisionContext, candidates: Sequence[Action]
    ) -> np.ndarray:
        """Backward-compatible alias for deterministic explicit evaluation."""
        scores, _, _ = self.evaluate(context, candidates)
        return scores

    def evaluate(
        self, context: DecisionContext, candidates: Sequence[Action]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return scores and value components without selection side effects."""
        q_task, q_lp = self.q_values(context, candidates)
        costs = np.array(
            [self._cost(candidate) for candidate in candidates], dtype=np.float32
        )

        if self.vector_values:
            task_term = q_task @ context.weights.astype(np.float32)
            scores = task_term + context.beta * q_lp
        else:
            # Weights were already applied in the scalar target.
            scores = q_task[:, 0]

        scores = scores - self.kappa * costs
        return scores, q_task, q_lp

    def q_values(
        self, context: DecisionContext, candidates: Sequence[Action]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (Q_task [N, K], Q_LP [N]) as numpy arrays."""
        context_tensor = torch.as_tensor(
            context.context_vector(self.include_motive_in_context), device=self.device
        ).unsqueeze(0)
        descriptors = self._descriptor_tensor(candidates)

        was_training = self.network.training
        self.network.eval()
        try:
            with torch.no_grad():
                q_task, q_lp = self.network(context_tensor, descriptors)
        finally:
            self.network.train(was_training)
        return q_task.squeeze(0).cpu().numpy(), q_lp.squeeze(0).cpu().numpy()

    def _descriptor_tensor(self, candidates: Sequence[Action]) -> torch.Tensor:
        descriptors = np.stack(
            [
                candidate.descriptor
                if isinstance(candidate, Candidate)
                else np.zeros(self.network.candidate_tower[0].in_features, dtype=np.float32)
                for candidate in candidates
            ]
        )
        return torch.as_tensor(descriptors, device=self.device).unsqueeze(0)

    @staticmethod
    def _cost(candidate: Action) -> float:
        """cost(z, c): the declared safety cost plus the candidate's own risk."""
        declared = getattr(candidate, "safety_cost", 0.0)
        return float(declared + 0.5 * candidate.risk_estimate)

    def goal_update_for_candidate(
        self,
        state: MotivationalState,
        candidate: Action,
        index: int,
        context: Optional[Any] = None,
    ) -> np.ndarray:
        context = self._require_context(context)
        return self.propose_goal_update(context, candidate, index)

    def propose_goal_update(
        self, context: DecisionContext, candidate: Action, index: int
    ) -> np.ndarray:
        """
        Gamma_0(x~, c, f) from equation (29), optionally plus the Stage 5
        residual of equation (31). The residual is bounded and must still pass
        through the MetaMo stabilizer.
        """
        base = np.asarray(candidate.delta_g, dtype=float).copy()
        if self.goal_residual_scale <= 0.0 or self.network.goal_residual_head is None:
            return base

        context_tensor = torch.as_tensor(
            context.context_vector(self.include_motive_in_context), device=self.device
        ).unsqueeze(0)
        descriptors = self._descriptor_tensor([candidate])
        self.network.eval()
        with torch.no_grad():
            residual = self.network.goal_residual(context_tensor, descriptors)
        return base + self.goal_residual_scale * residual.squeeze(0).squeeze(0).cpu().numpy()

class FixedOutputDecisionMonad(DQNDecisionMonad):
    """
    Conventional DQN decision head for conditions B0, B1, B3, and B4.

    Values live at fixed action indices, so the candidate set is restricted to
    primitives and any skill proposed by the adapter is simply invisible.
    """

    def q_values(
        self, context: DecisionContext, candidates: Sequence[Action]
    ) -> Tuple[np.ndarray, np.ndarray]:
        context_tensor = torch.as_tensor(
            context.context_vector(self.include_motive_in_context), device=self.device
        ).unsqueeze(0)
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.no_grad():
                q_task, q_lp = self.network(context_tensor)
        finally:
            self.network.train(was_training)
        q_task = q_task.squeeze(0).cpu().numpy()
        q_lp = q_lp.squeeze(0).cpu().numpy()

        indices = [
            candidate.primitive_index if getattr(candidate, "primitive_index", None) is not None else 0
            for candidate in candidates
        ]
        return q_task[indices], q_lp[indices]

    def propose_goal_update(
        self, context: DecisionContext, candidate: Action, index: int
    ) -> np.ndarray:
        return np.asarray(candidate.delta_g, dtype=float).copy()
