from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from core.state import MotivationalState, Stimulus, Action
from core.config import (
    G_ETHIC,
    LAX_DISTRIBUTIVE_DELTA,
    G_IND,
    G_HELP,
    G_NOVEL,
    G_SELF,
    G_SOC,
    G_TRANS,
    G_CURIO,
    C_CONTRACT,
    EPSILON,
    M_APPROACH,
    M_AROUSAL,
    M_RESOLUTION,
    M_SECURING,
    M_THRESHOLD,
    M_VALENCE,
    )
from category.functors import AppraisalComonad, AppraisalContext, DecisionMonad
from dynamics.stability import (
    StabilizationPolicy,
    StabilizationResult,
    is_in_safe_region,
    is_in_boundary_band,
    project_to_safe_region,
    raise_boundary_caution,
    MetaMoStabilizationPolicy,
    stabilize_goal_update,
)
from dynamics.coherence import blend_states

DecisionContextFactory = Callable[[MotivationalState], Any]

@dataclass(frozen=True)
class DecisionResult:
    """One selected candidate and its proposed goal update."""

    action: Action
    proposed_delta_g: np.ndarray

@dataclass(frozen=True)
class TransitionValidation:
    """Side-effect-free runtime validation of one MetaMo transition."""

    lax_distributive_error: float
    lax_distributive_passed: bool
    contraction_ratio: Optional[float]
    contractivity_passed: bool
    safety_passed: bool
    fallback_reasons: Tuple[str, ...] = ()

@dataclass(frozen=True)
class RuntimeValidationPolicy:
    """Controls validation and conservative fallback explicitly."""

    enabled: bool = True
    fallback_on_failure: bool = True

@dataclass(frozen=True)
class MetaMoTransitionResult:
    """Structured result of appraisal, decision, stabilization, and validation."""

    previous_state: MotivationalState
    appraised_state: MotivationalState
    decision: DecisionResult
    stabilization: StabilizationResult
    next_state: MotivationalState
    validation: TransitionValidation

@dataclass(frozen=True)
class ConsensusTransitionResult:
    """Structured two-perspective consensus transition."""

    action: Action
    perspective_a_target: MotivationalState
    perspective_b_target: MotivationalState
    merged_target: MotivationalState
    projected_target: MotivationalState
    next_state: MotivationalState

class MetaMoPseudoBimonad:
    """
    Represents the composite appraisal-then-decision operator F = D \circ \Psi.
    This forms a pseudo-bimonad on the motivational state space X = G \times M.
    """
    def __init__(
        self,
        appraisal: AppraisalComonad,
        decision: DecisionMonad,
        stabilization_policy: Optional[StabilizationPolicy] = None,
        validation_policy: Optional[RuntimeValidationPolicy] = None,
    ):
        self.appraisal = appraisal
        self.decision = decision
        # Default is no blending. Callers opt in with MetaMoStabilizationPolicy(blend=True).
        self.stabilization_policy = stabilization_policy or MetaMoStabilizationPolicy(
            blend=False
        )
        self.validation_policy = validation_policy or RuntimeValidationPolicy()

    def appraise(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        context: Optional[AppraisalContext] = None,
    ) -> MotivationalState:
        """Apply Psi and canonical boundary-sensitive caution."""
        appraised = self.appraisal.appraise_with_context(state, stimulus, context)
        if not self.stabilization_policy.enabled:
            return appraised
        return raise_boundary_caution(appraised)

    def decide(
        self,
        appraised_state: MotivationalState,
        candidates: List[Action],
        context: Optional[Any] = None,
    ) -> DecisionResult:
        """Select one candidate, allowing the policy to consume RNG."""
        action, proposed_delta_g = self.decision.decide_with_context(
            appraised_state, candidates, context
        )
        return DecisionResult(action, np.asarray(proposed_delta_g, dtype=float))

    def evaluate_decision(
        self,
        decision_state: MotivationalState,
        candidates: List[Action],
        context: Optional[Any] = None,
    ) -> DecisionResult:
        """Greedily evaluate a decision without exploration or RNG mutation."""
        scores = self.decision.evaluate_candidates(
            decision_state, candidates, context
        )
        if len(scores) != len(candidates) or not candidates:
            raise ValueError("candidate evaluation must return one score per candidate")
        index = int(np.argmax(scores))
        action = candidates[index]
        proposed_delta_g = self.decision.goal_update_for_candidate(
            decision_state, action, index, context
        )
        return DecisionResult(action, np.asarray(proposed_delta_g, dtype=float))

    def stabilize(
        self,
        previous_state: MotivationalState,
        appraised_state: MotivationalState,
        proposed_delta_g: np.ndarray,
        policy: Optional[StabilizationPolicy] = None,
    ) -> StabilizationResult:
        """Apply the canonical stabilization operation configured for this cycle."""
        return stabilize_goal_update(
            previous_state=previous_state,
            appraised_state=appraised_state,
            proposed_delta_g=proposed_delta_g,
            policy=policy or self.stabilization_policy,
        )

    def _compute_transition(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
        decision_context: Optional[Any] = None,
    ) -> Tuple[Action, MotivationalState]:
        """Compatibility helper: compute a transition without validation."""
        appraised_state = self.appraise(state, stimulus, appraisal_context)
        decision = self.decide(appraised_state, candidates, decision_context)
        stabilization = self.stabilize(
            state, appraised_state, decision.proposed_delta_g
        )
        return decision.action, stabilization.final_state

    def _state_from_delta(self, decision_state: MotivationalState, proposed_delta_g: np.ndarray) -> MotivationalState:
        """
        Apply a proposed goal update inside the same stabilization path used by the main transition.
        """
        stabilization = self.stabilize(
            previous_state=decision_state,
            appraised_state=decision_state,
            proposed_delta_g=proposed_delta_g,
        )
        return stabilization.final_state

    def _decision_context(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        appraisal_context: Optional[AppraisalContext] = None,
    ) -> MotivationalState:
        """
        Build the post-appraisal state that the decision monad should score.
        """
        return self.appraise(state, stimulus, appraisal_context)

    def _local_reference_state(self, state: MotivationalState, next_state: MotivationalState) -> MotivationalState:
        """
        Build a nearby state to probe local contractivity without depending on another subsystem.
        """
        delta_G = next_state.G - state.G
        delta_M = next_state.M - state.M

        probe_G = np.where(np.abs(delta_G) > 1e-6, np.sign(delta_G) * 0.01, 0.01)
        probe_M = np.where(np.abs(delta_M) > 1e-6, np.sign(delta_M) * 0.01, 0.01)

        return MotivationalState(
            G=np.clip(state.G + probe_G, 0.0, 1.0),
            M=np.clip(state.M + probe_M, 0.0, 1.0),
        )

    def consensus_action(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
    ) -> Action:
        """
        Select a shared action by combining the two subsystem evaluations over the same candidate set.
        """
        scores = self.consensus_scores(
            state_a, state_b, stimulus, candidates, appraisal_context
        )
        return candidates[int(np.argmax(scores))]

    def consensus_scores(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
    ) -> np.ndarray:
        """Return mean perspective scores with a disagreement penalty."""
        context_a = self.appraise(state_a, stimulus, appraisal_context)
        context_b = self.appraise(state_b, stimulus, appraisal_context)
        scores_a = self.decision.evaluate_candidates(context_a, candidates)
        scores_b = self.decision.evaluate_candidates(context_b, candidates)
        return (scores_a + scores_b) / 2.0 - 0.25 * np.abs(scores_a - scores_b)

    def consensus_transition(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
    ) -> Tuple[Action, MotivationalState]:
        """
        Build a coupled consensus action and consensus target state from the same shared candidate set.
        """
        action, _, _, merged_target = self._consensus_targets(
            state_a, state_b, stimulus, candidates, appraisal_context
        )
        return action, merged_target

    def _consensus_targets(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
    ) -> Tuple[
        Action, MotivationalState, MotivationalState, MotivationalState
    ]:
        """Build the two stabilized perspective targets and merge them."""
        action = self.consensus_action(
            state_a, state_b, stimulus, candidates, appraisal_context
        )
        context_a = self.appraise(state_a, stimulus, appraisal_context)
        context_b = self.appraise(state_b, stimulus, appraisal_context)
        target_a = self._state_from_delta(context_a, action.delta_g)
        target_b = self._state_from_delta(context_b, action.delta_g)
        return action, target_a, target_b, self.parallel_merge(target_a, target_b)

    def complete_consensus_transition(
        self,
        previous_state: MotivationalState,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
    ) -> ConsensusTransitionResult:
        """Project and blend a consensus target. ``consensus_transition`` still returns the unprojected merge."""
        action, target_a, target_b, merged_target = self._consensus_targets(
            state_a, state_b, stimulus, candidates, appraisal_context
        )
        projected_target = project_to_safe_region(merged_target)
        next_state = blend_states(previous_state, projected_target)
        return ConsensusTransitionResult(
            action=action,
            perspective_a_target=target_a,
            perspective_b_target=target_b,
            merged_target=merged_target,
            projected_target=projected_target,
            next_state=next_state,
        )

    def _apply_conservative_fallback(self, current_state: MotivationalState, next_state: MotivationalState) -> MotivationalState:
        """
        Shrink the transition toward the current state when runtime checks fail.
        """
        fallback_state = MotivationalState(
            G=((current_state.G * 0.5) + (next_state.G * 0.5)),
            M=((current_state.M * 0.5) + (next_state.M * 0.5)),
        )
        return project_to_safe_region(fallback_state)

    @staticmethod
    def _context_for(
        state: MotivationalState,
        factory: Optional[DecisionContextFactory],
    ) -> Optional[Any]:
        return None if factory is None else factory(state)

    def _deterministic_transition(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
        decision_context_factory: Optional[DecisionContextFactory] = None,
    ) -> MotivationalState:
        appraised = self.appraise(state, stimulus, appraisal_context)
        context = self._context_for(appraised, decision_context_factory)
        decision = self.evaluate_decision(appraised, candidates, context)
        return self.stabilize(
            state, appraised, decision.proposed_delta_g
        ).final_state

    def lax_distributive_error(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
        decision_context_factory: Optional[DecisionContextFactory] = None,
    ) -> float:
        """Measure the two deterministic appraisal/decision orderings."""
        decision_state_1 = self.appraise(state, stimulus, appraisal_context)
        context_1 = self._context_for(decision_state_1, decision_context_factory)
        decision_1 = self.evaluate_decision(
            decision_state_1, candidates, context_1
        )
        final_state_1 = self.stabilize(
            decision_state_1,
            decision_state_1,
            decision_1.proposed_delta_g,
        ).final_state

        context_2 = self._context_for(state, decision_context_factory)
        decision_2 = self.evaluate_decision(state, candidates, context_2)
        decided_state_2 = self.stabilize(
            state, state, decision_2.proposed_delta_g
        ).final_state
        final_state_2 = self.appraise(
            decided_state_2, stimulus, appraisal_context
        )
        return float(final_state_1.distance_to(final_state_2))

    def contractivity_ratio(
        self,
        state: MotivationalState,
        reference_state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
        decision_context_factory: Optional[DecisionContextFactory] = None,
    ) -> Optional[float]:
        """Return deterministic d(F(x), F(y)) / d(x, y) near the boundary."""
        if not (
            is_in_boundary_band(state) or is_in_boundary_band(reference_state)
        ):
            return None
        initial_distance = state.distance_to(reference_state)
        if initial_distance <= 1e-12:
            return None
        next_state = self._deterministic_transition(
            state,
            stimulus,
            candidates,
            appraisal_context,
            decision_context_factory,
        )
        next_reference = self._deterministic_transition(
            reference_state,
            stimulus,
            candidates,
            appraisal_context,
            decision_context_factory,
        )
        return float(next_state.distance_to(next_reference) / initial_distance)

    def complete_transition(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraised_state: MotivationalState,
        decision: DecisionResult,
        appraisal_context: Optional[AppraisalContext] = None,
        decision_context_factory: Optional[DecisionContextFactory] = None,
        validation_policy: Optional[RuntimeValidationPolicy] = None,
    ) -> MetaMoTransitionResult:
        """Stabilize and validate an already appraised and selected cycle."""
        policy = validation_policy or self.validation_policy
        stabilization = self.stabilize(
            state, appraised_state, decision.proposed_delta_g
        )
        next_state = stabilization.final_state

        lax_error = 0.0
        lax_passed = True
        contraction_ratio = None
        contractivity_passed = True
        safety_passed = is_in_safe_region(next_state)
        fallback_reasons = []

        if policy.enabled:
            lax_error = self.lax_distributive_error(
                state,
                stimulus,
                candidates,
                appraisal_context,
                decision_context_factory,
            )
            lax_passed = lax_error <= LAX_DISTRIBUTIVE_DELTA
            if not lax_passed:
                fallback_reasons.append("lax_distributive")
                if policy.fallback_on_failure:
                    next_state = self._apply_conservative_fallback(state, next_state)

            reference_state = self._local_reference_state(
                state, stabilization.final_state
            )
            contraction_ratio = self.contractivity_ratio(
                state,
                reference_state,
                stimulus,
                candidates,
                appraisal_context,
                decision_context_factory,
            )
            if contraction_ratio is not None:
                initial_distance = state.distance_to(reference_state)
                contractivity_passed = (
                    contraction_ratio * initial_distance
                    <= C_CONTRACT * initial_distance + EPSILON
                )
            if not contractivity_passed:
                fallback_reasons.append("contractivity")
                if policy.fallback_on_failure:
                    next_state = self._apply_conservative_fallback(state, next_state)

            safety_passed = is_in_safe_region(next_state)
            if not safety_passed:
                fallback_reasons.append("safe_region")
                if policy.fallback_on_failure:
                    next_state = self._apply_conservative_fallback(state, next_state)

        validation = TransitionValidation(
            lax_distributive_error=lax_error,
            lax_distributive_passed=lax_passed,
            contraction_ratio=contraction_ratio,
            contractivity_passed=contractivity_passed,
            safety_passed=safety_passed,
            fallback_reasons=tuple(fallback_reasons),
        )
        return MetaMoTransitionResult(
            previous_state=state,
            appraised_state=appraised_state,
            decision=decision,
            stabilization=stabilization,
            next_state=next_state,
            validation=validation,
        )

    def transition(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
        decision_context: Optional[Any] = None,
        decision_context_factory: Optional[DecisionContextFactory] = None,
        validation_policy: Optional[RuntimeValidationPolicy] = None,
    ) -> MetaMoTransitionResult:
        """Execute a complete MetaMo cycle and expose every intermediate."""
        appraised = self.appraise(state, stimulus, appraisal_context)
        decision = self.decide(appraised, candidates, decision_context)
        return self.complete_transition(
            state=state,
            stimulus=stimulus,
            candidates=candidates,
            appraised_state=appraised,
            decision=decision,
            appraisal_context=appraisal_context,
            decision_context_factory=decision_context_factory,
            validation_policy=validation_policy,
        )

    def step(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
    ) -> Tuple[Action, MotivationalState]:
        """Compatibility wrapper returning the original tuple API."""
        result = self.transition(
            state,
            stimulus,
            candidates,
            appraisal_context=appraisal_context,
        )
        return result.decision.action, result.next_state

    def check_lax_distributive_law(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
        appraisal_context: Optional[AppraisalContext] = None,
    ) -> bool:
        """Backward-compatible Boolean wrapper around the measured error."""
        return (
            self.lax_distributive_error(
                state, stimulus, candidates, appraisal_context
            )
            <= LAX_DISTRIBUTIVE_DELTA
        )
    
    def parallel_merge(self, state_a: MotivationalState, state_b: MotivationalState, coherence_correction: float = 0.05) -> MotivationalState:
        """Merge two motivational states, conservatively on safety and damped on exploration."""
        weight_a = state_a.G[G_IND]
        weight_b = state_b.G[G_IND]
        total_weight = weight_a + weight_b + 1e-9

        base_G = ((state_a.G * weight_a) + (state_b.G * weight_b)) / total_weight
        base_M = ((state_a.M * weight_a) + (state_b.M * weight_b)) / total_weight

        disagreement_G = np.abs(state_a.G - state_b.G)
        disagreement_M = np.abs(state_a.M - state_b.M)

        consensus_G = base_G.copy()
        consensus_M = base_M.copy()

        # Safety dimensions keep the stronger caution signal.
        safety_goal_idx = np.array([G_IND, G_HELP, G_ETHIC])
        consensus_G[safety_goal_idx] = np.maximum(state_a.G[safety_goal_idx], state_b.G[safety_goal_idx])

        # Exploratory dimensions keep the weaker signal unless both agree.
        exploratory_goal_idx = np.array([G_TRANS, G_CURIO, G_NOVEL, G_SELF])
        consensus_G[exploratory_goal_idx] = np.minimum(state_a.G[exploratory_goal_idx], state_b.G[exploratory_goal_idx])

        # Social engagement cannot exceed either subsystem.
        consensus_G[G_SOC] = min(base_G[G_SOC], state_a.G[G_SOC], state_b.G[G_SOC])

        # Caution modulators keep the higher warning.
        caution_mod_idx = np.array([M_THRESHOLD, M_SECURING])
        consensus_M[caution_mod_idx] = np.maximum(state_a.M[caution_mod_idx], state_b.M[caution_mod_idx])

        # Exploratory modulators keep the lower value.
        exploratory_mod_idx = np.array([M_AROUSAL, M_APPROACH])
        consensus_M[exploratory_mod_idx] = np.minimum(state_a.M[exploratory_mod_idx], state_b.M[exploratory_mod_idx])

        # Valence and resolution stay near the weighted average.
        shared_mod_idx = np.array([M_VALENCE, M_RESOLUTION])
        consensus_M[shared_mod_idx] = (
            (state_a.M[shared_mod_idx] + state_b.M[shared_mod_idx]) / 2.0
        )

        goal_correction_scale = np.ones_like(base_G)
        goal_correction_scale[safety_goal_idx] = 1.5
        goal_correction_scale[exploratory_goal_idx] = 1.0
        goal_correction_scale[G_SOC] = 0.8

        mod_correction_scale = np.ones_like(base_M)
        mod_correction_scale[caution_mod_idx] = 1.5
        mod_correction_scale[exploratory_mod_idx] = 1.0
        mod_correction_scale[shared_mod_idx] = 0.8

        goal_correction = np.clip(coherence_correction * disagreement_G * goal_correction_scale, 0.0, 1.0)
        mod_correction = np.clip(coherence_correction * disagreement_M * mod_correction_scale, 0.0, 1.0)

        merged_G = base_G + goal_correction * (consensus_G - base_G)
        merged_M = base_M + mod_correction * (consensus_M - base_M)

        return MotivationalState(G=merged_G, M=merged_M)
