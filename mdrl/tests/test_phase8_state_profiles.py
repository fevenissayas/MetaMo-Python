"""Tests for application-neutral MetaMo state-profile construction."""

from pathlib import Path

import numpy as np

from core.state_profiles import (
    REFERENCE_GOAL_PROFILE,
    REFERENCE_MODULATOR_LEVEL,
    create_reference_motivational_state,
)
from usecase.metamo.state import create_initial_motivational_state

def test_reference_profile_preserves_historical_values():
    state = create_reference_motivational_state()
    np.testing.assert_array_equal(
        state.G,
        np.array([0.65, 0.55, 0.75, 0.50, 0.45, 0.30, 0.85, 0.20]),
    )
    np.testing.assert_array_equal(state.G, np.asarray(REFERENCE_GOAL_PROFILE))
    np.testing.assert_array_equal(
        state.M, np.full(state.M.shape, REFERENCE_MODULATOR_LEVEL)
    )

def test_profile_factory_returns_independent_state_arrays():
    first = create_reference_motivational_state()
    second = create_reference_motivational_state()
    first.G[0] = 0.0
    first.M[0] = 1.0
    assert second.G[0] == 0.65
    assert second.M[0] == 0.5
    assert not np.shares_memory(first.G, second.G)
    assert not np.shares_memory(first.M, second.M)

def test_historical_usecase_constructor_is_a_compatible_reexport():
    canonical = create_reference_motivational_state()
    compatibility = create_initial_motivational_state()
    np.testing.assert_array_equal(compatibility.G, canonical.G)
    np.testing.assert_array_equal(compatibility.M, canonical.M)
    assert compatibility is not canonical
    assert not np.shares_memory(compatibility.G, canonical.G)

def test_mdrl_production_code_does_not_import_usecase():
    mdrl_root = Path(__file__).resolve().parents[1]
    violations = []
    for path in mdrl_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text()
        if "from usecase" in source or "import usecase" in source:
            violations.append(str(path.relative_to(mdrl_root)))
    assert not violations, f"MDRL production imports usecase modules: {violations}"

def test_original_agent_imports_the_core_profile_directly():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "usecase/agents/metamo_agent.py").read_text()
    assert "from core.state_profiles import" in source
    assert "from metamo.state import" not in source
