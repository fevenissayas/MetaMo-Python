"""Compatibility exports for the historical use-case state constructor."""

from core.state import MotivationalState
from core.state_profiles import create_reference_motivational_state


def create_initial_motivational_state() -> MotivationalState:
    """Return the canonical reference profile under the original API name."""
    return create_reference_motivational_state()


__all__ = ["create_initial_motivational_state", "create_reference_motivational_state"]
