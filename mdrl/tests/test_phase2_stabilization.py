"""Tests for the canonical Phase 2 goal-update stabilization operation."""

import numpy as np
import pytest

from core.state import MotivationalState
from dynamics.coherence import blend_states
from dynamics.stability import (
    MetaMoStabilizationPolicy,
    NoStabilizationPolicy,
    apply_homeostatic_damping,
    is_in_safe_region,
    project_to_safe_region,
    stabilize_goal_update,
)

def states():
    previous = MotivationalState(
        G=np.array([0.34, 0.72, 0.55, 0.48, 0.42, 0.31, 0.62, 0.27]),
        M=np.array([0.45, 0.58, 0.52, 0.41, 0.36, 0.39]),
    )
    appraised = MotivationalState(
        G=previous.G.copy(),
        M=np.array(
            [
                0.7234020208715899,
                0.9924097309897464,
                0.7202745667481365,
                0.8468362842349139,
                0.7555122712681895,
                0.713475944719526,
            ]
        ),
    )
    delta = np.array([-0.18, 0.12, 0.08, 0.06, 0.05, 0.04, 0.09, 0.03])
    return previous, appraised, delta

def test_normal_policy_exposes_every_intermediate_without_mutating_inputs():
    previous, appraised, delta = states()
    previous_before = previous.copy()
    appraised_before = appraised.copy()
    delta_before = delta.copy()

    result = stabilize_goal_update(previous, appraised, delta)

    expected_raw = MotivationalState(
        G=np.clip(appraised.G + delta, 0.0, 1.0),
        M=appraised.M.copy(),
    )
    expected_damped = apply_homeostatic_damping(appraised, delta)
    expected_target = MotivationalState(
        G=np.clip(appraised.G + expected_damped, 0.0, 1.0),
        M=appraised.M.copy(),
    )
    expected_projected = project_to_safe_region(expected_target)
    expected_final = blend_states(previous, expected_projected)

    np.testing.assert_allclose(result.raw_state.G, expected_raw.G)
    np.testing.assert_allclose(result.damped_delta_g, expected_damped)
    np.testing.assert_allclose(result.target_state.G, expected_target.G)
    np.testing.assert_allclose(result.projected_state.G, expected_projected.G)
    np.testing.assert_allclose(result.final_state.G, expected_final.G)
    np.testing.assert_allclose(result.final_state.M, expected_final.M)
    assert result.projection_magnitude == pytest.approx(
        np.linalg.norm(expected_target.G - expected_projected.G)
    )
    assert not result.pre_projection_safe
    assert result.post_projection_safe

    np.testing.assert_array_equal(previous.G, previous_before.G)
    np.testing.assert_array_equal(previous.M, previous_before.M)
    np.testing.assert_array_equal(appraised.G, appraised_before.G)
    np.testing.assert_array_equal(appraised.M, appraised_before.M)
    np.testing.assert_array_equal(delta, delta_before)

def test_non_blending_policy_preserves_historical_root_bimonad_semantics():
    previous, appraised, delta = states()
    result = stabilize_goal_update(
        previous,
        appraised,
        delta,
        policy=MetaMoStabilizationPolicy(blend=False),
    )
    np.testing.assert_allclose(result.final_state.G, result.projected_state.G)
    np.testing.assert_allclose(result.final_state.M, result.projected_state.M)
    assert result.blend_alpha == 1.0
    assert is_in_safe_region(result.final_state)

def test_no_stabilization_policy_applies_only_the_clipped_raw_proposal():
    previous, appraised, delta = states()
    result = stabilize_goal_update(
        previous,
        appraised,
        delta,
        policy=NoStabilizationPolicy(),
    )
    np.testing.assert_allclose(
        result.final_state.G,
        np.clip(appraised.G + delta, 0.0, 1.0),
    )
    np.testing.assert_allclose(result.final_state.M, appraised.M)
    np.testing.assert_allclose(result.damped_delta_g, delta)
    assert result.final_state is result.raw_state
    assert result.projection_magnitude == 0.0
    assert result.blend_alpha == 1.0

def test_projection_magnitude_ignores_the_projectors_caution_nudge():
    """Only displacement in goal space counts as a projection rescue."""
    previous, appraised, _ = states()
    result = stabilize_goal_update(
        previous,
        appraised,
        np.zeros_like(previous.G),
        policy=MetaMoStabilizationPolicy(blend=False),
    )
    assert result.projection_magnitude == pytest.approx(0.0, abs=1e-12)
    assert not np.array_equal(result.target_state.M, result.projected_state.M)
