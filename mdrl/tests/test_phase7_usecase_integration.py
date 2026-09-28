"""Tests that the original gridworld delegates its MetaMo transition."""

import inspect

import numpy as np

from dynamics.coherence import blend_states
from dynamics.stability import project_to_safe_region
from usecase.metamo import core as usecase_core
from usecase.metamo.state import create_initial_motivational_state

def environment_state():
    return {
        "pos": (4, 4),
        "mineral_pos": (2, 7),
        "dx_mineral": -2,
        "dy_mineral": 3,
        "in_lava": False,
        "lava_distance": 3,
        "lava_cells": ((8, 8), (8, 9), (9, 8), (9, 9)),
    }

def test_usecase_adapter_has_no_local_projection_or_blending_calls():
    source = inspect.getsource(usecase_core)
    assert "project_to_safe_region" not in source
    assert "blend_states" not in source

def test_canonical_complete_consensus_preserves_historical_composition():
    current = create_initial_motivational_state()
    env_state = environment_state()
    stimulus = usecase_core.build_stimulus(env_state, current)
    options = usecase_core.build_candidates(env_state, current)
    safety, growth = usecase_core.build_consensus_states(
        env_state, current, stimulus
    )

    action, merged = usecase_core.bimonad.consensus_transition(
        safety, growth, stimulus, [options[3]]
    )
    expected_target = project_to_safe_region(merged)
    expected_next = blend_states(current, expected_target)

    result = usecase_core.bimonad.complete_consensus_transition(
        current, safety, growth, stimulus, [options[3]]
    )
    assert result.action.id == action.id
    np.testing.assert_allclose(result.merged_target.G, merged.G, atol=1e-12)
    np.testing.assert_allclose(
        result.projected_target.G, expected_target.G, atol=1e-12
    )
    np.testing.assert_allclose(result.next_state.G, expected_next.G, atol=1e-12)
    np.testing.assert_allclose(result.next_state.M, expected_next.M, atol=1e-12)

def test_transition_for_action_delegates_to_complete_consensus(monkeypatch):
    calls = []
    original = usecase_core.bimonad.complete_consensus_transition

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        usecase_core.bimonad, "complete_consensus_transition", capture
    )
    usecase_core.transition_for_action(
        environment_state(), create_initial_motivational_state(), action_idx=3
    )
    assert len(calls) == 1

def test_choose_action_delegates_to_complete_consensus(monkeypatch):
    calls = []
    original = usecase_core.bimonad.complete_consensus_transition

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        usecase_core.bimonad, "complete_consensus_transition", capture
    )
    action, next_state, stimulus = usecase_core.choose_action(
        environment_state(), create_initial_motivational_state()
    )
    assert len(calls) == 1
    assert action.id in usecase_core.ACTION_IDS
    assert np.isfinite(next_state.G).all()
    assert 0.0 <= stimulus.risk <= 1.0
