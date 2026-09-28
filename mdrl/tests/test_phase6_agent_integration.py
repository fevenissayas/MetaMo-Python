"""Tests that MDRL is a thin integration over the canonical MetaMo cycle."""

import numpy as np

from dynamics.stability import MetaMoStabilizationPolicy, NoStabilizationPolicy
from mdrl.agent import MetaMoDRLAgent
from mdrl.bench.conditions import BY_NAME
from mdrl.config import TrainingConfig
from mdrl.envs.curious_gridworld import CuriousGridWorld, LayoutVariant

def agent(name="P1", seed=0):
    training = TrainingConfig(
        train_episodes=1,
        eval_episodes=1,
        max_steps=20,
        min_buffer_size=32,
        batch_size=16,
    )
    return MetaMoDRLAgent(BY_NAME[name], training, seed=seed)

def environment(seed=0):
    return CuriousGridWorld(
        seed=seed,
        layout=LayoutVariant.train_variants()[0],
        max_steps=20,
    )

def test_agent_bimonad_owns_the_same_appraisal_and_decision_instances():
    instance = agent("P1")
    assert instance.metamo.appraisal is instance.appraisal
    assert instance.metamo.decision is instance.decision
    assert isinstance(
        instance.metamo.stabilization_policy, MetaMoStabilizationPolicy
    )
    assert instance.metamo.stabilization_policy.blend

def test_no_stabilizer_ablation_uses_explicit_canonical_policy():
    instance = agent("A8-no-stabilizer")
    assert isinstance(instance.metamo.stabilization_policy, NoStabilizationPolicy)

def test_one_agent_step_delegates_to_canonical_phases(monkeypatch):
    instance = agent("P1", seed=3)
    env = environment(seed=3)
    observation = env.reset()
    instance.reset_episode()
    calls = {"appraise": 0, "decide": 0, "complete": 0}
    captured = {}

    original_appraise = instance.metamo.appraise
    original_decide = instance.metamo.decide
    original_complete = instance.metamo.complete_transition

    def appraise(*args, **kwargs):
        calls["appraise"] += 1
        return original_appraise(*args, **kwargs)

    def decide(*args, **kwargs):
        calls["decide"] += 1
        return original_decide(*args, **kwargs)

    def complete(*args, **kwargs):
        calls["complete"] += 1
        result = original_complete(*args, **kwargs)
        captured["transition"] = result
        return result

    monkeypatch.setattr(instance.metamo, "appraise", appraise)
    monkeypatch.setattr(instance.metamo, "decide", decide)
    monkeypatch.setattr(instance.metamo, "complete_transition", complete)

    instance.step(env, observation, learn=False)

    assert calls["appraise"] >= 1
    assert calls["decide"] == 1
    assert calls["complete"] == 1
    np.testing.assert_allclose(
        instance.state.G, captured["transition"].next_state.G
    )
    np.testing.assert_allclose(
        instance.state.M, captured["transition"].next_state.M
    )
    assert len(instance.stabilizer.history) == 1

def test_agent_no_longer_contains_parallel_stability_pipeline():
    instance = agent("P1")
    assert not hasattr(instance, "_apply_pipeline")
    assert not hasattr(instance, "_contraction_ratio")
    assert not hasattr(instance, "_lax_distributive_error")
    assert not hasattr(instance, "_greedy_delta_g")
