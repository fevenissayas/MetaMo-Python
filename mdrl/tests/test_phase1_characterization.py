"""Characterization of the pre-refactor transition paths, before ownership is consolidated."""

import copy

import numpy as np
import pytest
import torch

from category.bimonad import MetaMoPseudoBimonad
from category.functors import DecisionMonad
from core.config import NUM_GOALS
from core.state import Action, MotivationalState, Stimulus
from dynamics.coherence import blend_states, calculate_blend_factor
from dynamics.stability import (
    apply_homeostatic_damping,
    is_in_safe_region,
    project_to_safe_region,
    raise_boundary_caution,
)
from mdrl.agent import MetaMoDRLAgent
from mdrl.bench.conditions import BY_NAME
from mdrl.config import TrainingConfig
from mdrl.envs.curious_gridworld import CuriousGridWorld, LayoutVariant
from mdrl.stabilizer import MetaMoStabilizer
from openpsi.appraisal import OpenPsiAppraisal
from usecase.metamo.core import transition_for_action
from usecase.metamo.state import create_initial_motivational_state

GOLDEN_STATE = MotivationalState(
    G=np.array([0.34, 0.72, 0.55, 0.48, 0.42, 0.31, 0.62, 0.27]),
    M=np.array([0.45, 0.58, 0.52, 0.41, 0.36, 0.39]),
)
GOLDEN_STIMULUS = Stimulus(
    novelty=0.73,
    conduciveness=0.61,
    risk=0.42,
    effort=0.28,
)
GOLDEN_DELTA = np.array([-0.18, 0.12, 0.08, 0.06, 0.05, 0.04, 0.09, 0.03])

class FixedDecision(DecisionMonad):
    """Deterministic decision used to isolate the canonical transition path."""

    def __init__(self, delta_g: np.ndarray):
        self.delta_g = delta_g
        self.calls = 0
        self.evaluation_calls = 0

    def unit(self, state: MotivationalState) -> MotivationalState:
        return state

    def decide(self, state, candidates):
        self.calls += 1
        return candidates[0], self.delta_g.copy()

    def score_candidate(self, state, candidate):
        self.evaluation_calls += 1
        return 0.0

    def goal_update_for_candidate(self, state, candidate, index, context=None):
        return self.delta_g.copy()

def fixed_action() -> Action:
    return Action(
        id="fixed",
        goal_correlations=np.zeros(NUM_GOALS),
        risk_estimate=0.0,
        delta_g=GOLDEN_DELTA.copy(),
    )

def test_openpsi_appraisal_golden_output():
    """Lock down the fixed appraiser used by both applications."""
    appraised = OpenPsiAppraisal().appraise(GOLDEN_STATE, GOLDEN_STIMULUS)
    np.testing.assert_allclose(appraised.G, GOLDEN_STATE.G, atol=1e-12)
    np.testing.assert_allclose(
        appraised.M,
        [
            0.7234020208715899,
            0.9924097309897464,
            0.7202745667481365,
            0.8468362842349139,
            0.6055122712681896,
            0.5634759447195261,
        ],
        atol=1e-12,
    )

def test_canonical_bimonad_compute_transition_matches_shared_equations():
    """The root bimonad currently owns caution, damping, and projection."""
    decision = FixedDecision(GOLDEN_DELTA)
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), decision)
    action, actual = bimonad._compute_transition(
        GOLDEN_STATE, GOLDEN_STIMULUS, [fixed_action()]
    )

    appraised = OpenPsiAppraisal().appraise(GOLDEN_STATE, GOLDEN_STIMULUS)
    decision_state = raise_boundary_caution(appraised)
    damped = apply_homeostatic_damping(decision_state, GOLDEN_DELTA)
    target = MotivationalState(
        G=np.clip(decision_state.G + damped, 0.0, 1.0),
        M=decision_state.M.copy(),
    )
    expected = project_to_safe_region(target)

    assert action.id == "fixed"
    np.testing.assert_allclose(actual.G, expected.G, atol=1e-12)
    np.testing.assert_allclose(actual.M, expected.M, atol=1e-12)
    assert is_in_safe_region(actual)

def test_mdrl_stabilizer_is_canonical_target_plus_blending():
    """Document the deliberate orchestration difference to consolidate later."""
    appraised = raise_boundary_caution(
        OpenPsiAppraisal().appraise(GOLDEN_STATE, GOLDEN_STIMULUS)
    )
    canonical_target = MetaMoPseudoBimonad(
        OpenPsiAppraisal(), FixedDecision(GOLDEN_DELTA)
    )._state_from_delta(appraised, GOLDEN_DELTA)
    expected = blend_states(GOLDEN_STATE, canonical_target)

    actual, diagnostics = MetaMoStabilizer().update(
        GOLDEN_STATE, appraised, GOLDEN_DELTA
    )
    np.testing.assert_allclose(actual.G, expected.G, atol=1e-12)
    np.testing.assert_allclose(actual.M, expected.M, atol=1e-12)
    assert diagnostics.blend_alpha == pytest.approx(
        calculate_blend_factor(GOLDEN_STATE)
    )
    assert diagnostics.pre_projection_safe is np.False_
    assert diagnostics.post_projection_safe is np.True_
    assert diagnostics.projection_magnitude == pytest.approx(0.10328)

def test_mdrl_stabilizer_golden_final_state():
    """Protect the historical benchmark transition while it is migrated."""
    appraised = raise_boundary_caution(
        OpenPsiAppraisal().appraise(GOLDEN_STATE, GOLDEN_STIMULUS)
    )
    actual, _ = MetaMoStabilizer().update(GOLDEN_STATE, appraised, GOLDEN_DELTA)
    np.testing.assert_allclose(
        actual.G,
        [
            0.33652,
            0.72831024,
            0.55554016,
            0.48415512,
            0.4234626,
            0.31277008,
            0.62623268,
            0.27207756,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.M,
        [
            0.4737859758158284,
            0.6158796465961079,
            0.5374238873070879,
            0.4480047567284375,
            0.4031095676003325,
            0.42684240719059874,
        ],
        atol=1e-12,
    )

def test_original_usecase_transition_golden_output():
    """Lock down the existing tabular-gridworld MetaMo adapter."""
    env_state = {
        "pos": (4, 4),
        "mineral_pos": (2, 7),
        "dx_mineral": -2,
        "dy_mineral": 3,
        "in_lava": False,
        "lava_distance": 3,
        "lava_cells": ((8, 8), (8, 9), (9, 8), (9, 9)),
    }
    action, next_state, stimulus, target = transition_for_action(
        env_state, create_initial_motivational_state(), action_idx=3
    )
    assert action.id == "RIGHT"
    assert stimulus == Stimulus(
        novelty=0.8055555555555556,
        conduciveness=0.7222222222222222,
        risk=0.12,
        effort=0.17955555555555558,
    )
    np.testing.assert_allclose(
        next_state.G,
        [
            0.6522572196263752,
            0.5538466171501343,
            0.7499999999679964,
            0.5054583541220987,
            0.4554583541242208,
            0.29999999998719856,
            0.8520461766644668,
            0.19999999999146575,
        ],
        atol=1e-12,
    )
    expected = blend_states(create_initial_motivational_state(), target)
    np.testing.assert_allclose(next_state.G, expected.G, atol=1e-12)
    np.testing.assert_allclose(next_state.M, expected.M, atol=1e-12)

def test_root_step_selects_once_and_validates_by_deterministic_evaluation():
    """Phase 5 replacement for the repeated-selection migration sentinel."""
    decision = FixedDecision(np.zeros(NUM_GOALS))
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), decision)
    bimonad.step(GOLDEN_STATE, GOLDEN_STIMULUS, [fixed_action()])
    assert decision.calls == 1
    assert decision.evaluation_calls >= 2

def test_mdrl_stability_probes_do_not_consume_rng_or_train_networks():
    """Current deterministic probes may update telemetry, but not policy state."""
    training = TrainingConfig(
        train_episodes=1,
        eval_episodes=1,
        max_steps=20,
        min_buffer_size=32,
        batch_size=16,
    )
    agent = MetaMoDRLAgent(BY_NAME["P1"], training, seed=17)
    env = CuriousGridWorld(
        seed=17,
        layout=LayoutVariant.train_variants()[0],
        max_steps=20,
    )
    observation = env.reset()
    agent.reset_episode()
    stimulus = agent.build_stimulus(observation)
    appraised = agent.appraise(agent.state, observation, stimulus)
    candidates = agent.candidates_for(observation, appraised)
    context = agent.build_context(observation, appraised, stimulus)

    rng_before = copy.deepcopy(agent.rng.bit_generator.state)
    epsilon_before = agent.epsilon
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in agent.network.named_parameters()
    }

    appraisal_context = agent.build_appraisal_context(observation)
    context_factory = lambda decision_state: agent.build_context(
        observation, decision_state, stimulus
    )
    reference = MotivationalState(
        G=np.clip(agent.state.G + 0.01, 0.0, 1.0),
        M=np.clip(agent.state.M + 0.01, 0.0, 1.0),
    )
    agent.metamo.contractivity_ratio(
        agent.state,
        reference,
        stimulus,
        list(candidates),
        appraisal_context,
        context_factory,
    )
    agent.metamo.lax_distributive_error(
        agent.state,
        stimulus,
        list(candidates),
        appraisal_context,
        context_factory,
    )

    assert agent.rng.bit_generator.state == rng_before
    assert agent.epsilon == epsilon_before
    for name, parameter in agent.network.named_parameters():
        torch.testing.assert_close(parameter, parameters_before[name])
