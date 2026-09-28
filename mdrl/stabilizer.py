"""Benchmark history for the canonical MetaMo stabilizer. Transition math stays in ``dynamics.stability``."""

from typing import Callable, List, Optional

import numpy as np

from core.state import MotivationalState, Stimulus
from dynamics.stability import (
    is_in_boundary_band,
    MetaMoStabilizationPolicy,
    NoStabilizationPolicy,
    raise_boundary_caution,
    stabilize_goal_update,
    StabilizationResult,
)
from mdrl.types import StepDiagnostics

class MetaMoStabilizer:
    """Compatibility adapter and recorder for canonical stabilization results."""

    def __init__(self, enabled: bool = True, blend: bool = True):
        self.enabled = enabled
        self.blend = blend
        self.history: List[StepDiagnostics] = []

    def reset(self) -> None:
        self.history = []

    def appraisal_context(self, appraised: MotivationalState) -> MotivationalState:
        """Boundary-sensitive caution applied before the decision is scored."""
        if not self.enabled:
            return appraised
        return raise_boundary_caution(appraised)

    def update(
        self,
        state: MotivationalState,
        appraised: MotivationalState,
        raw_delta_g: np.ndarray,
        contraction_probe: Optional[Callable[[], Optional[float]]] = None,
        lax_distributive_error: float = 0.0,
    ) -> tuple:
        """
        Equation (30): x_{t+1} = H(x_t, x~_t, Delta G_hat) in X_safe.

        Returns (next_state, diagnostics).
        """
        policy = (
            MetaMoStabilizationPolicy(blend=self.blend)
            if self.enabled
            else NoStabilizationPolicy()
        )
        result = stabilize_goal_update(
            previous_state=state,
            appraised_state=appraised,
            proposed_delta_g=raw_delta_g,
            policy=policy,
        )
        contraction_ratio = contraction_probe() if contraction_probe is not None else None
        diagnostics = self.record(
            result,
            contraction_ratio=contraction_ratio,
            lax_distributive_error=lax_distributive_error,
        )
        return result.final_state, diagnostics

    def record(
        self,
        result: StabilizationResult,
        contraction_ratio: Optional[float] = None,
        lax_distributive_error: float = 0.0,
    ) -> StepDiagnostics:
        """Record diagnostics from a transition stabilized by the bimonad."""
        diagnostics = StepDiagnostics(
            raw_delta_g=result.raw_delta_g,
            accepted_delta_g=result.accepted_delta_g,
            pre_projection_safe=result.pre_projection_safe,
            post_projection_safe=result.post_projection_safe,
            projection_magnitude=result.projection_magnitude,
            boundary_pressure=result.boundary_pressure,
            contraction_ratio=contraction_ratio,
            lax_distributive_error=lax_distributive_error,
            self_model_drift=result.self_model_drift,
            blend_alpha=result.blend_alpha,
        )
        self.history.append(diagnostics)
        return diagnostics

    def summary(self) -> dict:
        """Aggregate the Section 7.6 safety and stability block."""
        if not self.history:
            return {}
        ratios = [
            record.contraction_ratio
            for record in self.history
            if record.contraction_ratio is not None
        ]
        return {
            "pre_projection_violation_rate": float(
                np.mean([not record.pre_projection_safe for record in self.history])
            ),
            "post_projection_violation_rate": float(
                np.mean([not record.post_projection_safe for record in self.history])
            ),
            "projection_frequency": float(
                np.mean([record.projection_magnitude > 1e-9 for record in self.history])
            ),
            "projection_magnitude": float(
                np.mean([record.projection_magnitude for record in self.history])
            ),
            "boundary_pressure": float(
                np.mean([record.boundary_pressure for record in self.history])
            ),
            "contraction_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "contractivity_pass_rate": (
                float(np.mean([ratio <= 1.0 for ratio in ratios])) if ratios else float("nan")
            ),
            "lax_distributive_error": float(
                np.mean([record.lax_distributive_error for record in self.history])
            ),
            "self_model_drift": float(
                np.mean([record.self_model_drift for record in self.history])
            ),
            "raw_vs_accepted_delta": float(
                np.mean(
                    [
                        float(np.linalg.norm(record.raw_delta_g - record.accepted_delta_g))
                        for record in self.history
                    ]
                )
            ),
        }

def boundary_band_rate(states: List[MotivationalState]) -> float:
    if not states:
        return 0.0
    return float(np.mean([is_in_boundary_band(state) for state in states]))
