"""Tests for explicit, backward-compatible appraisal context transport."""

import numpy as np
import pytest
import torch

from category.bimonad import MetaMoPseudoBimonad
from category.functors import AppraisalComonad, AppraisalContext, DecisionMonad
from core.config import NUM_GOALS, NUM_MODULATORS
from core.state import Action, MotivationalState, Stimulus
from mdrl.agent import MetaMoDRLAgent
from mdrl.appraisal.residual import ResidualAppraisal
from mdrl.bench.conditions import BY_NAME
from mdrl.config import TrainingConfig
from mdrl.envs.curious_gridworld import OBSERVATION_DIM, CuriousGridWorld, LayoutVariant
from openpsi.appraisal import OpenPsiAppraisal

def motivational_state():
    return MotivationalState(
        G=np.array([0.65, 0.55, 0.75, 0.50, 0.45, 0.30, 0.85, 0.20]),
        M=np.full(NUM_MODULATORS, 0.5),
    )

def stimulus():
    return Stimulus(novelty=0.6, conduciveness=0.7, risk=0.2, effort=0.3)

class LegacyAppraisal(AppraisalComonad):
    """An existing implementation with no awareness of context."""

    def extract(self, state):
        return state

    def appraise(self, state, feedback):
        result = state.copy()
        result.M[0] += feedback.conduciveness
        return result

class FixedDecision(DecisionMonad):
    def unit(self, state):
        return state

    def decide(self, state, candidates):
        return candidates[0], np.zeros(NUM_GOALS)

    def score_candidate(self, state, candidate):
        return 0.0

class CapturingAppraisal(LegacyAppraisal):
    def __init__(self):
        self.contexts = []

    def appraise_with_context(self, state, feedback, context=None):
        self.contexts.append(context)
        return super().appraise_with_context(state, feedback, context)

def action():
    return Action(
        id="stay",
        goal_correlations=np.zeros(NUM_GOALS),
        risk_estimate=0.0,
        delta_g=np.zeros(NUM_GOALS),
    )

def test_legacy_appraiser_ignores_context_through_default_method():
    appraiser = LegacyAppraisal()
    state = motivational_state()
    context = AppraisalContext(
        observation=np.arange(5),
        curiosity_features=np.ones(4),
        learning=True,
    )
    direct = appraiser.appraise(state, stimulus())
    contextual = appraiser.appraise_with_context(state, stimulus(), context)
    np.testing.assert_allclose(contextual.G, direct.G)
    np.testing.assert_allclose(contextual.M, direct.M)

def test_bimonad_threads_context_through_appraisal_and_validation():
    appraiser = CapturingAppraisal()
    context = AppraisalContext(observation=np.arange(3), metadata={"source": "test"})
    bimonad = MetaMoPseudoBimonad(appraiser, FixedDecision())
    bimonad.step(motivational_state(), stimulus(), [action()], context)
    assert len(appraiser.contexts) >= 3
    assert all(seen is context for seen in appraiser.contexts)

def test_residual_appraisal_requires_explicit_observation_context():
    torch.manual_seed(0)
    residual = ResidualAppraisal(
        base=OpenPsiAppraisal(),
        observation_dim=OBSERVATION_DIM,
        enabled=True,
    )
    # Make the residual deterministic and nonzero independently of observation.
    for parameter in residual.network.parameters():
        parameter.data.zero_()
    residual.network.residual_head[0].bias.data.fill_(0.5)

    state = motivational_state()
    fixed = residual.appraise(state, stimulus())
    contextual = residual.appraise_with_context(
        state,
        stimulus(),
        AppraisalContext(observation=np.zeros(OBSERVATION_DIM, dtype=np.float32)),
    )
    expected_base = OpenPsiAppraisal().appraise(state, stimulus())
    np.testing.assert_allclose(fixed.M, expected_base.M)
    assert not np.allclose(contextual.M, expected_base.M)
    assert not hasattr(residual, "observation")

def test_residual_training_rejects_missing_observation():
    residual = ResidualAppraisal(
        base=OpenPsiAppraisal(),
        observation_dim=OBSERVATION_DIM,
        enabled=True,
    )
    with pytest.raises(ValueError, match="requires an observation"):
        residual.train_step(
            AppraisalContext(learning=True),
            motivational_state(),
            np.zeros(4, dtype=np.float32),
        )

def test_agent_reuses_pre_appraisal_inputs_for_inference_and_training(monkeypatch):
    training = TrainingConfig(
        train_episodes=1,
        eval_episodes=1,
        max_steps=20,
        min_buffer_size=32,
        batch_size=16,
    )
    agent = MetaMoDRLAgent(BY_NAME["P2"], training, seed=9)
    env = CuriousGridWorld(
        seed=9,
        layout=LayoutVariant.train_variants()[0],
        max_steps=20,
    )
    observation = env.reset()
    agent.reset_episode()
    initial_state = agent.state.copy()
    seen = {}

    original_appraise = agent.appraisal.appraise_with_context

    def capture_appraise(state, feedback, context=None):
        if "inference_state" not in seen:
            seen["inference_state"] = state.copy()
            seen["inference_context"] = context
        return original_appraise(state, feedback, context)

    def capture_train(context, state, realized_outcome, stimulus=None):
        seen["training_state"] = state.copy()
        seen["training_context"] = context
        seen["training_stimulus"] = stimulus
        return None

    monkeypatch.setattr(agent.appraisal, "appraise_with_context", capture_appraise)
    monkeypatch.setattr(agent.appraisal, "train_step", capture_train)
    agent.step(env, observation, learn=True)

    np.testing.assert_allclose(seen["inference_state"].G, initial_state.G)
    np.testing.assert_allclose(seen["inference_state"].M, initial_state.M)
    np.testing.assert_allclose(seen["training_state"].G, initial_state.G)
    np.testing.assert_allclose(seen["training_state"].M, initial_state.M)
    assert seen["training_context"] is seen["inference_context"]
    np.testing.assert_array_equal(
        seen["training_context"].observation, observation["features"]
    )
    assert seen["training_context"].learning
    assert seen["training_stimulus"] is not None
