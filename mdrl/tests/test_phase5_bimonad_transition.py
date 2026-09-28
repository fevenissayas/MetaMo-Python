"""Tests for the phased, structured MetaMo pseudo-bimonad transition API."""

import numpy as np

from category.bimonad import (
    MetaMoPseudoBimonad,
    MetaMoTransitionResult,
    RuntimeValidationPolicy,
)
from category.functors import DecisionMonad
from core.config import NUM_GOALS, NUM_MODULATORS
from core.state import Action, MotivationalState, Stimulus
from dynamics.stability import MetaMoStabilizationPolicy
from mdrl.agent import MetaMoDRLAgent
from mdrl.bench.conditions import BY_NAME
from mdrl.config import TrainingConfig
from mdrl.envs.curious_gridworld import CuriousGridWorld, LayoutVariant
from openpsi.appraisal import OpenPsiAppraisal

class TrackingDecision(DecisionMonad):
    def __init__(self):
        self.selection_calls = 0
        self.evaluation_calls = 0

    def unit(self, state):
        return state

    def score_candidate(self, state, candidate):
        self.evaluation_calls += 1
        return float(candidate.goal_correlations @ state.G)

    def decide(self, state, candidates):
        self.selection_calls += 1
        scores = np.asarray(
            [candidate.goal_correlations @ state.G for candidate in candidates]
        )
        chosen = candidates[int(np.argmax(scores))]
        return chosen, chosen.delta_g.copy()

def state():
    return MotivationalState(
        G=np.array([0.34, 0.72, 0.55, 0.48, 0.42, 0.31, 0.62, 0.27]),
        M=np.array([0.45, 0.58, 0.52, 0.41, 0.36, 0.39]),
    )

def feedback():
    return Stimulus(novelty=0.73, conduciveness=0.61, risk=0.42, effort=0.28)

def candidates():
    first = Action(
        id="cautious",
        goal_correlations=np.array([0.8, 0.1, 0.4, 0.1, 0.1, 0.2, 0.8, 0.2]),
        risk_estimate=0.1,
        delta_g=np.array([0.02, -0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0]),
    )
    second = Action(
        id="explore",
        goal_correlations=np.array([0.1, 0.8, 0.2, 0.9, 0.8, 0.6, 0.1, 0.2]),
        risk_estimate=0.5,
        delta_g=np.array([-0.08, 0.08, 0.0, 0.06, 0.05, 0.02, 0.0, 0.0]),
    )
    return [first, second]

def test_public_phases_equal_unvalidated_structured_transition():
    decision_monad = TrackingDecision()
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), decision_monad)
    initial = state()
    options = candidates()

    appraised = bimonad.appraise(initial, feedback())
    decision = bimonad.decide(appraised, options)
    stabilization = bimonad.stabilize(
        initial, appraised, decision.proposed_delta_g
    )
    result = bimonad.transition(
        initial,
        feedback(),
        options,
        validation_policy=RuntimeValidationPolicy(enabled=False),
    )

    assert isinstance(result, MetaMoTransitionResult)
    assert result.decision.action.id == decision.action.id
    np.testing.assert_allclose(result.appraised_state.G, appraised.G)
    np.testing.assert_allclose(result.appraised_state.M, appraised.M)
    np.testing.assert_allclose(
        result.stabilization.final_state.G, stabilization.final_state.G
    )
    np.testing.assert_allclose(result.next_state.G, result.stabilization.final_state.G)
    assert result.validation.fallback_reasons == ()

def test_runtime_validation_never_reselects_an_action():
    decision = TrackingDecision()
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), decision)
    bimonad.transition(state(), feedback(), candidates())
    assert decision.selection_calls == 1
    assert decision.evaluation_calls >= 4

def test_fallback_behavior_is_an_explicit_policy():
    without_fallback = MetaMoPseudoBimonad(
        OpenPsiAppraisal(),
        TrackingDecision(),
        validation_policy=RuntimeValidationPolicy(
            enabled=True, fallback_on_failure=False
        ),
    ).transition(state(), feedback(), candidates())
    assert without_fallback.validation.fallback_reasons
    np.testing.assert_allclose(
        without_fallback.next_state.G,
        without_fallback.stabilization.final_state.G,
    )

    with_fallback = MetaMoPseudoBimonad(
        OpenPsiAppraisal(), TrackingDecision()
    ).transition(state(), feedback(), candidates())
    assert with_fallback.validation.fallback_reasons
    assert not np.allclose(
        with_fallback.next_state.G,
        with_fallback.stabilization.final_state.G,
    )

def test_step_preserves_the_original_action_state_tuple_api():
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), TrackingDecision())
    action, next_state = bimonad.step(state(), feedback(), candidates())
    assert isinstance(action, Action)
    assert isinstance(next_state, MotivationalState)

def test_context_factory_supports_dqn_validation_without_ambient_state():
    training = TrainingConfig(
        train_episodes=1,
        eval_episodes=1,
        max_steps=20,
        min_buffer_size=32,
        batch_size=16,
    )
    agent = MetaMoDRLAgent(BY_NAME["P1"], training, seed=12)
    env = CuriousGridWorld(
        seed=12,
        layout=LayoutVariant.train_variants()[0],
        max_steps=20,
    )
    observation = env.reset()
    agent.reset_episode()
    stimulus = agent.build_stimulus(observation)
    appraisal_context = agent.build_appraisal_context(observation)

    bimonad = MetaMoPseudoBimonad(
        agent.appraisal,
        agent.decision,
        stabilization_policy=MetaMoStabilizationPolicy(blend=True),
        validation_policy=RuntimeValidationPolicy(
            enabled=True, fallback_on_failure=False
        ),
    )
    appraised = bimonad.appraise(agent.state, stimulus, appraisal_context)
    options = agent.candidates_for(observation, appraised)
    main_context = agent.build_context(observation, appraised, stimulus)
    context_states = []

    def context_factory(decision_state):
        context_states.append(decision_state)
        return agent.build_context(observation, decision_state, stimulus)

    result = bimonad.transition(
        agent.state,
        stimulus,
        options,
        appraisal_context=appraisal_context,
        decision_context=main_context,
        decision_context_factory=context_factory,
    )

    assert result.decision.action in options
    assert context_states
    assert not hasattr(agent.decision, "context")
    assert result.validation.lax_distributive_error >= 0.0
