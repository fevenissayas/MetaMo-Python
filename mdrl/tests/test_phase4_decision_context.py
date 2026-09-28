"""Tests for explicit decision context and side-effect-free policy evaluation."""

import copy

import numpy as np
import pytest

from category.functors import DecisionMonad
from core.config import NUM_GOALS
from core.state import Action, MotivationalState
from mdrl.agent import MetaMoDRLAgent
from mdrl.bench.conditions import BY_NAME
from mdrl.config import TrainingConfig
from mdrl.envs.curious_gridworld import CuriousGridWorld, LayoutVariant

class LegacyDecision(DecisionMonad):
    """A context-free decision monad written against the original interface."""

    def unit(self, state):
        return state

    def score_candidate(self, state, candidate):
        return float(candidate.goal_correlations @ state.G)

    def decide(self, state, candidates):
        scores = self.evaluate_candidates(state, candidates)
        chosen = candidates[int(np.argmax(scores))]
        return chosen, chosen.delta_g.copy()

def state():
    return MotivationalState(
        G=np.linspace(0.2, 0.9, NUM_GOALS),
        M=np.full(6, 0.5),
    )

def actions():
    first = Action(
        id="first",
        goal_correlations=np.linspace(0.0, 0.7, NUM_GOALS),
        risk_estimate=0.0,
        delta_g=np.zeros(NUM_GOALS),
    )
    second = Action(
        id="second",
        goal_correlations=np.linspace(0.7, 0.0, NUM_GOALS),
        risk_estimate=0.0,
        delta_g=np.full(NUM_GOALS, 0.01),
    )
    return [first, second]

def tiny_agent(seed=0):
    training = TrainingConfig(
        train_episodes=1,
        eval_episodes=1,
        max_steps=20,
        min_buffer_size=32,
        batch_size=16,
    )
    return MetaMoDRLAgent(BY_NAME["P1"], training, seed=seed)

def decision_fixture(seed=0):
    agent = tiny_agent(seed)
    env = CuriousGridWorld(
        seed=seed,
        layout=LayoutVariant.train_variants()[0],
        max_steps=20,
    )
    observation = env.reset()
    agent.reset_episode()
    feedback = agent.build_stimulus(observation)
    appraised = agent.appraise(agent.state, observation, feedback)
    candidates = agent.candidates_for(observation, appraised)
    context = agent.build_context(observation, appraised, feedback)
    return agent, observation, feedback, appraised, candidates, context

def test_legacy_decision_ignores_context_through_default_method():
    decision = LegacyDecision()
    candidates = actions()
    direct = decision.decide(state(), candidates)
    contextual = decision.decide_with_context(
        state(), candidates, context={"integration": "ignored"}
    )
    assert contextual[0].id == direct[0].id
    np.testing.assert_array_equal(contextual[1], direct[1])

def test_dqn_rejects_missing_decision_context():
    agent, _, _, appraised, candidates, _ = decision_fixture()
    assert not hasattr(agent.decision, "context")
    with pytest.raises(RuntimeError, match="explicit DecisionContext"):
        agent.decision.decide(appraised, list(candidates))
    with pytest.raises(RuntimeError, match="explicit DecisionContext"):
        agent.decision.evaluate_candidates(appraised, list(candidates))

def test_deterministic_evaluation_has_no_selection_side_effects():
    agent, _, _, appraised, candidates, context = decision_fixture(seed=4)
    decision = agent.decision
    decision.network.train()
    rng_before = copy.deepcopy(decision.rng.bit_generator.state)

    first = decision.evaluate_candidates(appraised, list(candidates), context)
    second = decision.evaluate_candidates(appraised, list(candidates), context)

    np.testing.assert_allclose(first, second)
    assert decision.rng.bit_generator.state == rng_before
    assert decision.network.training
    assert decision.last_scores is None
    assert decision.last_q_task is None
    assert decision.last_q_lp is None
    assert not decision.last_exploratory

def test_explicit_selection_records_the_selected_evaluation():
    agent, _, _, appraised, candidates, context = decision_fixture(seed=5)
    decision = agent.decision
    decision.epsilon = 0.0
    expected, q_task, q_lp = decision.evaluate(context, candidates)

    chosen, delta_g = decision.decide_with_context(
        appraised, list(candidates), context
    )

    assert chosen in candidates
    assert delta_g.shape == (NUM_GOALS,)
    np.testing.assert_allclose(decision.last_scores, expected)
    np.testing.assert_allclose(decision.last_q_task, q_task)
    np.testing.assert_allclose(decision.last_q_lp, q_lp)
    assert not decision.last_exploratory

def test_stability_probes_do_not_overwrite_last_selection_telemetry():
    agent, observation, feedback, appraised, candidates, context = decision_fixture(seed=7)
    agent.set_epsilon(0.0)
    agent.select(context, candidates)
    scores_before = agent.decision.last_scores.copy()
    q_task_before = agent.decision.last_q_task.copy()
    q_lp_before = agent.decision.last_q_lp.copy()
    exploratory_before = agent.decision.last_exploratory
    rng_before = copy.deepcopy(agent.rng.bit_generator.state)

    appraisal_context = agent.build_appraisal_context(observation)
    context_factory = lambda decision_state: agent.build_context(
        observation, decision_state, feedback
    )
    reference = MotivationalState(
        G=np.clip(agent.state.G + 0.01, 0.0, 1.0),
        M=np.clip(agent.state.M + 0.01, 0.0, 1.0),
    )
    agent.metamo.contractivity_ratio(
        agent.state,
        reference,
        feedback,
        list(candidates),
        appraisal_context,
        context_factory,
    )
    agent.metamo.lax_distributive_error(
        agent.state,
        feedback,
        list(candidates),
        appraisal_context,
        context_factory,
    )

    np.testing.assert_array_equal(agent.decision.last_scores, scores_before)
    np.testing.assert_array_equal(agent.decision.last_q_task, q_task_before)
    np.testing.assert_array_equal(agent.decision.last_q_lp, q_lp_before)
    assert agent.decision.last_exploratory == exploratory_before
    assert agent.rng.bit_generator.state == rng_before
