"""One decision step, motivational update, and counterfactual scoring."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from category.bimonad import DecisionResult, RuntimeValidationPolicy
from category.functors import AppraisalContext
from core.config import NUM_GOALS
from core.state import MotivationalState, Stimulus
from mdrl.agent.records import DIAGNOSTIC_PERIOD, StepRecord
from mdrl.config import NUM_OUTCOMES, OUT_SAFETY, PROJECTION_PENALTY
from mdrl.envs.curious_gridworld import MotiveIntervention, Zone
from mdrl.types import Candidate, DecisionContext

class AgentStep:
    def step(self, env, observation: Dict[str, Any], learn: bool = True) -> Tuple[Dict[str, Any], StepRecord, bool]:
        learn = bool(learn and self.learning_enabled)
        if (
            not learn
            and self.appraisal_curiosity is not None
            and self._appraisal_curiosity_rollout is None
        ):
            # Evaluation without set_learning_mode still uses a frozen online copy.
            self._start_appraisal_curiosity_rollout()
        stimulus = self.build_stimulus(observation)
        appraisal_context = self.build_appraisal_context(observation, learning=learn)
        appraised = self.appraise(
            self.state, observation, stimulus, context=appraisal_context
        )
        candidates = self.candidates_for(observation, appraised)
        context = self.build_context(observation, appraised, stimulus)
        chosen, raw_delta_g, exploratory = self.select(context, candidates)

        zone_name = observation.get("zone", Zone.NONE).name.lower()

        def step_hook(previous, action, nxt, outcome, info) -> float:
            descriptor = chosen.descriptor
            policy_reward = self.curiosity.update_and_measure_progress(
                previous["features"],
                descriptor,
                nxt["features"],
                zone=previous.get("zone", Zone.NONE).name.lower(),
                action=action,
                learn=learn and self.policy_curiosity_learning_enabled,
            )
            appraisal_curiosity = self.active_appraisal_curiosity()
            if appraisal_curiosity is not None:
                appraisal_curiosity.update_and_measure_progress(
                    previous["features"],
                    descriptor,
                    nxt["features"],
                    zone=previous.get("zone", Zone.NONE).name.lower(),
                    action=action,
                    learn=(
                        self.appraisal_curiosity_learning_enabled
                        or appraisal_curiosity is self._appraisal_curiosity_rollout
                    ),
                )
            return policy_reward

        result = self.adapter.execute(
            chosen,
            env,
            observation,
            gamma=self.training.gamma,
            step_hook=(
                step_hook
                if self.condition.uses_curiosity
                or self.active_appraisal_curiosity() is not None
                else None
            ),
        )

        next_observation = result.observation
        next_stimulus = self.build_stimulus(next_observation)

        next_state, diagnostics = self._advance_motivation(
            observation,
            appraised,
            chosen,
            raw_delta_g,
            stimulus,
            candidates,
            appraisal_context,
        )

        # Charge the safety outcome for how far the stabilizer had to project.
        outcome = result.discounted_outcome.copy()
        outcome[OUT_SAFETY] -= PROJECTION_PENALTY * diagnostics.projection_magnitude

        next_appraised = self.appraise(next_state, next_observation, next_stimulus)
        next_context = self.build_context(next_observation, next_appraised, next_stimulus)
        next_candidates = self.candidates_for(next_observation, next_appraised)

        if learn and self.decision_learning_enabled and self.replay is not None:
            self._store(
                context,
                chosen,
                outcome,
                result,
                next_context,
                next_candidates,
                diagnostics.projection_magnitude,
            )
            self._learn()

        if learn and self.appraisal_learning_enabled:
            self.appraisal.train_step(
                appraisal_context,
                self.state,
                result.total_outcome,
                stimulus=stimulus,
            )

        self.state = next_state
        self.motivational_trace.append(self.state.copy())
        self._step_index += 1

        record = StepRecord(
            candidate_id=chosen.id,
            candidate_kind=chosen.kind.value,
            certified=chosen.is_certified,
            duration=result.duration,
            outcome=result.total_outcome,
            lp_reward=result.lp_reward,
            safety_cost=result.safety_cost,
            weights=context.weights.copy(),
            beta=context.beta,
            zone=zone_name,
            in_lava=bool(next_observation["in_lava"]),
            lava_distance=int(next_observation["lava_distance"]),
            energy=float(next_observation["energy"]),
            exploratory=exploratory,
            descriptor_regret=self._descriptor_regret(context.weights, chosen, candidates),
            events=result.events,
        )
        self.episode_records.append(record)
        return next_observation, record, result.done

    @staticmethod
    def _descriptor_regret(
        weights: np.ndarray, chosen: Candidate, candidates: Sequence[Candidate]
    ) -> float:
        """Scalarized gap between the best declared effect vector and the chosen one."""
        if not candidates:
            return 0.0
        values = [float(weights @ candidate.mu_hat) for candidate in candidates]
        return float(max(values) - float(weights @ chosen.mu_hat))

    def _advance_motivation(
        self,
        observation: Dict[str, Any],
        appraised: MotivationalState,
        chosen: Candidate,
        raw_delta_g: np.ndarray,
        stimulus: Stimulus,
        candidates: Sequence[Candidate],
        appraisal_context: AppraisalContext,
    ):
        if self.condition.goal_update == "none" or not self.condition.uses_metamo:
            from mdrl.types import StepDiagnostics

            inert = StepDiagnostics(
                raw_delta_g=np.zeros(NUM_GOALS),
                accepted_delta_g=np.zeros(NUM_GOALS),
                pre_projection_safe=True,
                post_projection_safe=True,
                projection_magnitude=0.0,
                boundary_pressure=0.0,
                contraction_ratio=None,
                lax_distributive_error=0.0,
                self_model_drift=0.0,
                blend_alpha=0.0,
            )
            return self.state.copy(), inert

        run_probes = self._step_index % DIAGNOSTIC_PERIOD == 0

        def context_factory(decision_state: MotivationalState) -> DecisionContext:
            return self.build_context(observation, decision_state, stimulus)

        transition = self.metamo.complete_transition(
            state=self.state,
            stimulus=stimulus,
            candidates=list(candidates),
            appraised_state=appraised,
            decision=DecisionResult(chosen, np.asarray(raw_delta_g, dtype=float)),
            appraisal_context=appraisal_context,
            decision_context_factory=context_factory,
            validation_policy=RuntimeValidationPolicy(
                enabled=run_probes, fallback_on_failure=False
            ),
        )
        diagnostics = self.stabilizer.record(
            transition.stabilization,
            contraction_ratio=transition.validation.contraction_ratio,
            lax_distributive_error=transition.validation.lax_distributive_error,
        )
        return transition.next_state, diagnostics

    def score_under_state(
        self,
        observation: Dict[str, Any],
        motivational_state: MotivationalState,
        candidates: Optional[Sequence[Candidate]] = None,
    ) -> Dict[str, Any]:
        """
        Greedy evaluation of the same observation under an arbitrary
        motivational state. This is the primitive behind PSR (equation 42) and
        motivational consistency (equation 43).
        """
        stimulus = self.build_stimulus(observation)
        appraised = self.appraise(motivational_state, observation, stimulus)
        if candidates is None:
            candidates = self.candidates_for(observation, appraised)
        context = self.build_context(observation, appraised, stimulus)

        if self.condition.decision == "magus":
            scores = self.decision.evaluate_candidates(
                appraised, list(candidates)
            )
            q_task = np.zeros((len(candidates), NUM_OUTCOMES), dtype=np.float32)
            q_lp = np.zeros(len(candidates), dtype=np.float32)
        else:
            scores, q_task, q_lp = self.decision.evaluate(context, candidates)

        index = int(np.argmax(scores))
        return {
            "index": index,
            "candidate": candidates[index],
            "candidates": list(candidates),
            "scores": np.asarray(scores),
            "q_task": np.asarray(q_task),
            "q_lp": np.asarray(q_lp),
            "weights": context.weights,
            "beta": context.beta,
        }

    def apply_intervention(self, intervention: MotiveIntervention) -> None:
        """Apply a scheduled motive change to the agent's own state."""
        if not self.condition.uses_metamo:
            return
        state = self.state.copy()
        for index, delta in intervention.goal_deltas.items():
            state.G[index] = float(np.clip(state.G[index] + delta, 0.0, 1.0))
        for index, delta in intervention.modulator_deltas.items():
            state.M[index] = float(np.clip(state.M[index] + delta, 0.0, 1.0))
        self.state = state

