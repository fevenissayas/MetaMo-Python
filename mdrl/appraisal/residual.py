"""Bounded learned residual on the fixed appraiser, projected and logged apart from the base update."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from category.functors import AppraisalComonad, AppraisalContext
from core.config import (
    M_APPROACH,
    M_SECURING,
    M_THRESHOLD,
    NUM_GOALS,
    NUM_MODULATORS,
)
from core.state import MotivationalState, Stimulus
from mdrl.config import NUM_OUTCOMES

class AppraisalResidualNet(nn.Module):
    """rho_eta(o, x, h): a bounded correction to the modulator update."""

    def __init__(
        self, observation_dim: int, curiosity_dim: int = 0, hidden_dim: int = 64
    ):
        super().__init__()
        self.curiosity_dim = curiosity_dim
        input_dim = observation_dim + NUM_GOALS + NUM_MODULATORS + curiosity_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.residual_head = nn.Sequential(nn.Linear(hidden_dim, NUM_MODULATORS), nn.Tanh())
        # Outcome loss trains the residual only through the corrected appraisal.
        outcome_input = NUM_GOALS + NUM_MODULATORS
        self.outcome_head = nn.Sequential(
            nn.Linear(outcome_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_OUTCOMES),
        )

    def residual(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        curiosity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if curiosity is None:
            curiosity = torch.zeros(
                observation.shape[0], self.curiosity_dim, device=observation.device
            )
        hidden = self.trunk(torch.cat([observation, state, curiosity], dim=-1))
        return self.residual_head(hidden)

    def predict_outcome(
        self, observation: torch.Tensor, corrected_state: torch.Tensor
    ) -> torch.Tensor:
        """Predict from the corrected appraisal so loss trains the residual."""
        del observation  # retained in the signature for compatibility
        return self.outcome_head(corrected_state)

class ResidualAppraisal(AppraisalComonad):
    """Fixed OpenPsi appraisal plus an optional bounded residual from ``AppraisalContext``."""

    def __init__(
        self,
        base: AppraisalComonad,
        observation_dim: int,
        enabled: bool = False,
        alpha_m: float = 0.15,
        learning_rate: float = 3e-4,
        lambda_pred: float = 1.0,
        lambda_small: float = 0.5,
        lambda_smooth: float = 0.25,
        lambda_safe: float = 1.0,
        curiosity_dim: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.base = base
        self.enabled = enabled
        self.alpha_m = alpha_m
        self.device = device or torch.device("cpu")
        self.lambda_pred = lambda_pred
        self.lambda_small = lambda_small
        self.lambda_smooth = lambda_smooth
        self.lambda_safe = lambda_safe

        self.network: Optional[AppraisalResidualNet] = None
        self.optimizer = None
        if enabled:
            self.network = AppraisalResidualNet(
                observation_dim, curiosity_dim=curiosity_dim
            ).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self._previous_observation: Optional[np.ndarray] = None
        self._previous_state: Optional[MotivationalState] = None
        self._previous_curiosity: Optional[np.ndarray] = None
        self._last_residual: Optional[np.ndarray] = None
        self.residual_log: list = []

    def extract(self, state: MotivationalState) -> MotivationalState:
        return self.base.extract(state)

    def appraise(self, state: MotivationalState, stimulus: Stimulus) -> MotivationalState:
        """Context-free compatibility path used by fixed MetaMo callers."""
        return self.appraise_with_context(state, stimulus, context=None)

    def appraise_with_context(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        context: Optional[AppraisalContext] = None,
    ) -> MotivationalState:
        appraised = self.base.appraise_with_context(state, stimulus, context)
        observation = None if context is None else context.observation
        if not self.enabled or observation is None:
            self._last_residual = np.zeros(NUM_MODULATORS, dtype=np.float32)
            return appraised

        curiosity = None if context is None else context.curiosity_features
        residual = self._residual(observation, state, curiosity)
        self._last_residual = residual
        self.residual_log.append(float(np.linalg.norm(residual)))
        corrected = np.clip(appraised.M + self.alpha_m * residual, 0.0, 1.0)
        return MotivationalState(G=appraised.G.copy(), M=corrected)

    def _residual(
        self,
        observation: np.ndarray,
        state: MotivationalState,
        curiosity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            residual = self.network.residual(
                *self._tensors(observation, state, curiosity)
            )
        return residual.squeeze(0).cpu().numpy()

    def _tensors(
        self,
        observation: np.ndarray,
        state: MotivationalState,
        curiosity: Optional[np.ndarray] = None,
    ):
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        state_tensor = torch.as_tensor(
            np.concatenate([state.G, state.M]).astype(np.float32), device=self.device
        ).unsqueeze(0)
        curiosity_dim = 0 if self.network is None else self.network.curiosity_dim
        values = (
            np.zeros(curiosity_dim, dtype=np.float32)
            if curiosity is None
            else np.asarray(curiosity, dtype=np.float32)
        )
        if values.shape != (curiosity_dim,):
            raise ValueError(
                f"appraisal curiosity must have shape {(curiosity_dim,)}, got {values.shape}"
            )
        curiosity_tensor = torch.as_tensor(values, device=self.device).unsqueeze(0)
        return observation_tensor, state_tensor, curiosity_tensor

    def train_step(
        self,
        context: AppraisalContext,
        state: MotivationalState,
        realized_outcome: np.ndarray,
        stimulus: Optional[Stimulus] = None,
    ) -> Optional[Dict[str, float]]:
        """One step of the equation (34) objective."""
        if not self.enabled:
            return None
        if context.observation is None:
            raise ValueError("Residual appraisal training requires an observation")

        observation = np.asarray(context.observation, dtype=np.float32)

        self.network.train()
        observation_tensor, state_tensor, curiosity_tensor = self._tensors(
            observation, state, context.curiosity_features
        )
        residual = self.network.residual(
            observation_tensor, state_tensor, curiosity_tensor
        )

        base_appraised = (
            self.base.appraise_with_context(state, stimulus, context)
            if stimulus is not None
            else state
        )
        base_modulators = torch.as_tensor(
            base_appraised.M.astype(np.float32), device=self.device
        ).unsqueeze(0)
        corrected_modulators = torch.clamp(
            base_modulators + self.alpha_m * residual, 0.0, 1.0
        )
        corrected_state = torch.cat(
            [state_tensor[:, :NUM_GOALS], corrected_modulators], dim=-1
        )
        predicted_outcome = self.network.predict_outcome(
            observation_tensor, corrected_state
        )

        target = torch.as_tensor(
            np.asarray(realized_outcome, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        outcome_loss = nn.functional.mse_loss(predicted_outcome, target)
        small_loss = residual.pow(2).sum()

        if self._previous_observation is not None and self._previous_state is not None:
            previous_tensor, previous_state_tensor, previous_curiosity_tensor = self._tensors(
                self._previous_observation,
                self._previous_state,
                self._previous_curiosity,
            )
            with torch.no_grad():
                previous_residual = self.network.residual(
                    previous_tensor,
                    previous_state_tensor,
                    previous_curiosity_tensor,
                )
            smooth_loss = (residual - previous_residual).pow(2).sum()
        else:
            smooth_loss = torch.zeros((), device=self.device)

        # Under risk, caution should meet the risk and approach should stay inside the safe margin.
        risk = torch.tensor(
            0.0 if stimulus is None else float(stimulus.risk),
            dtype=corrected_modulators.dtype,
            device=self.device,
        )
        caution = 0.5 * (
            corrected_modulators[:, M_THRESHOLD]
            + corrected_modulators[:, M_SECURING]
        )
        caution_gap = torch.relu(risk - caution)
        approach_gap = torch.relu(
            corrected_modulators[:, M_APPROACH] - (1.0 - risk)
        )
        safe_loss = (caution_gap.pow(2) + approach_gap.pow(2)).mean()

        loss = (
            self.lambda_pred * outcome_loss
            + self.lambda_small * small_loss
            + self.lambda_smooth * smooth_loss
            + self.lambda_safe * safe_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._previous_observation = np.asarray(observation, dtype=np.float32).copy()
        self._previous_state = state.copy()
        self._previous_curiosity = (
            None
            if context.curiosity_features is None
            else np.asarray(context.curiosity_features, dtype=np.float32).copy()
        )
        return {
            "appraisal_loss": float(loss.item()),
            "appraisal_outcome_loss": float(outcome_loss.item()),
            "appraisal_safety_loss": float(safe_loss.item()),
            "appraisal_smoothness_loss": float(smooth_loss.item()),
            "appraisal_residual_norm": float(residual.detach().norm().item()),
        }

    def reset_episode(self) -> None:
        self._previous_observation = None
        self._previous_state = None
        self._previous_curiosity = None

    def summary(self) -> Dict[str, float]:
        if not self.residual_log:
            return {"appraisal_residual_norm": 0.0}
        return {"appraisal_residual_norm": float(np.mean(self.residual_log))}
