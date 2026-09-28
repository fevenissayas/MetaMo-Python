"""Intrinsic reward for every curiosity mode in `VALID_MODES`."""

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import torch

from mdrl.config import LP_CLIP, LP_WINDOW
from mdrl.curiosity.models import (
    ErrorModel,
    IntrinsicCuriosityModule,
    RandomNetworkDistillation,
    WorldModel,
    WorldModelEnsemble,
)

CURIOSITY_FEATURE_DIM = 4

VALID_MODES = (
    "none",
    "lp_h",
    "lp_f",
    "lp_window",
    "raw_error",
    "icm",
    "rnd",
    "disagreement",
)

# Slow decay of the normalizer's high-water mark.
SCALE_DECAY = 0.99995

# Floor so a silent module cannot turn near-zero noise into a full reward.
SCALE_FLOOR = 1e-3

class CuriosityModule:
    """Adapter over learning-progress, ICM, and novelty estimators."""

    def __init__(
        self,
        observation_dim: int,
        descriptor_dim: int,
        num_actions: int = 4,
        mode: str = "lp_h",
        world_model_lr: float = 1e-3,
        error_model_lr: float = 5e-4,
        icm_feature_dim: int = 32,
        icm_forward_loss_weight: float = 0.2,
        icm_reward_scale: float = 1.0,
        window: int = LP_WINDOW,
        normalize: bool = True,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"unknown curiosity mode {mode!r}; expected one of {VALID_MODES}")
        self.mode = mode
        self.window = window
        self.normalize = normalize
        self.device = device or torch.device("cpu")
        torch.manual_seed(seed)

        self.world_model: Optional[WorldModel] = None
        self.error_model: Optional[ErrorModel] = None
        self.icm: Optional[IntrinsicCuriosityModule] = None
        self.rnd: Optional[RandomNetworkDistillation] = None
        self.ensemble: Optional[WorldModelEnsemble] = None

        if mode in ("lp_h", "lp_f", "lp_window", "raw_error"):
            self.world_model = WorldModel(
                observation_dim, descriptor_dim, learning_rate=world_model_lr
            ).to(self.device)
        if mode == "lp_h":
            self.error_model = ErrorModel(
                observation_dim, descriptor_dim, learning_rate=error_model_lr
            ).to(self.device)
        if mode == "rnd":
            self.rnd = RandomNetworkDistillation(observation_dim).to(self.device)
        if mode == "icm":
            self.icm = IntrinsicCuriosityModule(
                observation_dim=observation_dim,
                num_actions=num_actions,
                feature_dim=icm_feature_dim,
                learning_rate=world_model_lr,
                forward_loss_weight=icm_forward_loss_weight,
                reward_scale=icm_reward_scale,
            ).to(self.device)
        if mode == "disagreement":
            self.ensemble = WorldModelEnsemble(observation_dim, descriptor_dim).to(self.device)

        self._error_history: Deque[float] = deque(maxlen=2 * window)
        self._running_scale = 0.0
        self._scale_peak = 0.0
        self._last_error = 0.0
        self._last_progress = 0.0
        self._progress_ema = 0.0
        self._steps_since_progress = 0
        self.total_steps = 0
        self.learning_enabled = True

        # Per-zone bookkeeping for the curiosity-quality metrics of Section 7.6.
        self.zone_errors: Dict[str, list] = {}
        self.zone_progress: Dict[str, list] = {}

    def reset_episode(self) -> None:
        self._error_history.clear()
        self._last_error = 0.0
        self._last_progress = 0.0
        self._progress_ema = 0.0
        self._steps_since_progress = 0
        self.total_steps = 0
        self.zone_errors = {}
        self.zone_progress = {}

    def set_learning(self, enabled: bool) -> None:
        """Enable or freeze estimator and normalizer updates."""
        self.learning_enabled = bool(enabled)

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self.device).unsqueeze(0)

    def _normalized(self, raw: float, update: bool = True) -> float:
        """Scale by a decaying high-water mark so progress fades as the world model converges."""
        if not self.normalize:
            return float(np.clip(raw, -LP_CLIP, LP_CLIP))
        magnitude = abs(float(raw))
        if update:
            self._running_scale = 0.98 * self._running_scale + 0.02 * magnitude
            self._scale_peak = max(self._scale_peak * SCALE_DECAY, self._running_scale)
        scale = max(self._scale_peak, SCALE_FLOOR)
        return float(np.clip(raw / scale, -LP_CLIP, LP_CLIP))

    def update_and_measure_progress(
        self,
        observation: np.ndarray,
        descriptor: np.ndarray,
        next_observation: np.ndarray,
        zone: Optional[str] = None,
        action: Optional[int] = None,
        learn: bool = True,
    ) -> float:
        """
        Update the curiosity machinery on one primitive transition and return
        the selected estimator's intrinsic reward.
        """
        learn = bool(learn and self.learning_enabled)
        self.total_steps += 1

        if self.mode == "none":
            self._last_progress = 0.0
            return 0.0

        obs_tensor = self._tensor(observation)
        descriptor_tensor = self._tensor(descriptor)
        next_tensor = self._tensor(next_observation)

        if self.mode == "icm":
            if action is None:
                raise ValueError("Pathak ICM requires the executed primitive action")
            raw = (
                self.icm.update_and_score(obs_tensor, int(action), next_tensor)
                if learn
                else self.icm.score(obs_tensor, int(action), next_tensor)
            )
            self._last_error = raw
            progress = self._normalized(raw, update=learn)
        elif self.mode == "rnd":
            raw = (
                self.rnd.update_and_score(next_tensor)
                if learn
                else self.rnd.score(next_tensor)
            )
            self._last_error = raw
            progress = self._normalized(raw, update=learn)
        elif self.mode == "disagreement":
            raw = (
                self.ensemble.update_and_score(obs_tensor, descriptor_tensor, next_tensor)
                if learn
                else self.ensemble.score(obs_tensor, descriptor_tensor)
            )
            self._last_error = raw
            progress = self._normalized(raw, update=learn)
        else:
            error_before = self.world_model.error(obs_tensor, descriptor_tensor, next_tensor)
            self._last_error = error_before

            if not learn and self.mode in ("lp_h", "lp_f", "lp_window"):
                raw = 0.0
            elif self.mode == "lp_h":
                predicted_before = self.error_model.predict(obs_tensor, descriptor_tensor)
                self.world_model.update(obs_tensor, descriptor_tensor, next_tensor)
                actual = self.world_model.error(obs_tensor, descriptor_tensor, next_tensor)
                self.error_model.update(
                    obs_tensor,
                    descriptor_tensor,
                    torch.tensor([[min(actual, 1.0)]], device=self.device),
                )
                predicted_after = self.error_model.predict(obs_tensor, descriptor_tensor)
                raw = predicted_before - predicted_after
            elif self.mode == "lp_f":
                self.world_model.update(obs_tensor, descriptor_tensor, next_tensor)
                error_after = self.world_model.error(obs_tensor, descriptor_tensor, next_tensor)
                raw = error_before - error_after
            elif self.mode == "lp_window":
                self.world_model.update(obs_tensor, descriptor_tensor, next_tensor)
                self._error_history.append(error_before)
                raw = self._windowed_progress()
            else:  # raw_error
                if learn:
                    self.world_model.update(
                        obs_tensor, descriptor_tensor, next_tensor
                    )
                raw = error_before

            progress = self._normalized(raw, update=learn)

        self._last_progress = progress
        self._progress_ema = 0.95 * self._progress_ema + 0.05 * progress
        if abs(progress) > 0.05:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        if zone is not None:
            self.zone_errors.setdefault(zone, []).append(self._last_error)
            self.zone_progress.setdefault(zone, []).append(progress)

        return progress

    def _windowed_progress(self) -> float:
        """Equation (26): mean error over an old window minus a new one."""
        if len(self._error_history) < 2 * self.window:
            return 0.0
        history = list(self._error_history)
        old = history[: self.window]
        new = history[self.window :]
        return float(np.mean(old) - np.mean(new))

    def features(self, energy_fraction: float = 1.0) -> np.ndarray:
        """
        u_t from equation (12): world-model error, recent learning progress, and
        remaining budget, so the decision learner can condition on how much
        there is left to learn.
        """
        return np.array(
            [
                float(np.clip(self._last_error, 0.0, 2.0)) / 2.0,
                float(np.clip(self._last_progress, -1.0, 1.0)),
                float(np.clip(self._progress_ema, -1.0, 1.0)),
                float(np.clip(self._steps_since_progress / 25.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def progress_probe(self) -> float:
        """Recent learning progress, used by skill termination predicates."""
        return float(self._progress_ema)

    def zone_report(self) -> Dict[str, Dict[str, float]]:
        report = {}
        for zone in set(self.zone_errors) | set(self.zone_progress):
            errors = self.zone_errors.get(zone, [])
            progress = self.zone_progress.get(zone, [])
            report[zone] = {
                "mean_error": float(np.mean(errors)) if errors else 0.0,
                "final_error": float(np.mean(errors[-20:])) if errors else 0.0,
                "mean_progress": float(np.mean(progress)) if progress else 0.0,
                "steps": float(len(errors)),
            }
        return report

    def diagnostics(self) -> Dict[str, float]:
        """Estimator-specific diagnostics for implementation validation."""
        if self.icm is not None:
            return self.icm.diagnostics()
        return {}
