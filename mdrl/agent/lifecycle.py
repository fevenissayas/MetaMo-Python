"""Episode reset, training scope, and decision checkpoints."""

import copy
import hashlib
from typing import Any, Dict, Optional

import torch

from core.state import MotivationalState
from mdrl.curiosity.module import CuriosityModule

class AgentLifecycle:
    def reset_episode(self) -> None:
        self.state = self.initial_state.copy()
        self.stabilizer.reset()
        self.curiosity.reset_episode()
        if self.appraisal_curiosity is not None:
            self.appraisal_curiosity.reset_episode()
            if self.appraisal_curiosity_learning_enabled:
                self._appraisal_curiosity_rollout = None
            else:
                self._start_appraisal_curiosity_rollout()
        self.appraisal.reset_episode()
        self.episode_records = []
        self.motivational_trace = []
        self._step_index = 0

    def set_learning_mode(self, enabled: bool) -> None:
        """Enable training or freeze every learned auxiliary component."""
        self.learning_enabled = bool(enabled)
        self.decision_learning_enabled = bool(enabled)
        self.appraisal_learning_enabled = bool(
            enabled and self.condition.appraisal == "residual"
        )
        self.policy_curiosity_learning_enabled = bool(enabled)
        self.appraisal_curiosity_learning_enabled = bool(
            enabled and self.appraisal_curiosity is not None
        )
        self.curiosity.set_learning(enabled)
        if self.appraisal_curiosity is not None:
            self.appraisal_curiosity.set_learning(enabled)
        self._appraisal_curiosity_rollout = None
        modules = [self.network, self.target_network, self.appraisal.network]
        for module in modules:
            if module is not None:
                module.train(enabled)

    def set_training_scope(self, scope: str) -> None:
        """Select which learned components may update during an episode."""
        if scope not in ("joint", "appraisal", "frozen"):
            raise ValueError(f"unknown training scope {scope!r}")
        active = scope != "frozen"
        self.learning_enabled = active
        self.decision_learning_enabled = scope == "joint"
        self.appraisal_learning_enabled = (
            active and self.condition.appraisal == "residual"
        )
        self.policy_curiosity_learning_enabled = scope == "joint"
        # Staged appraisal trains the residual only; its curiosity signal stays frozen.
        self.appraisal_curiosity_learning_enabled = (
            scope == "joint" and self.appraisal_curiosity is not None
        )
        self.curiosity.set_learning(self.policy_curiosity_learning_enabled)
        if self.appraisal_curiosity is not None:
            self.appraisal_curiosity.set_learning(
                self.appraisal_curiosity_learning_enabled
            )
        self._appraisal_curiosity_rollout = None
        if self.network is not None:
            for parameter in self.network.parameters():
                parameter.requires_grad_(self.decision_learning_enabled)
        if self.appraisal.network is not None:
            for parameter in self.appraisal.network.parameters():
                parameter.requires_grad_(self.appraisal_learning_enabled)

    def active_appraisal_curiosity(self) -> Optional[CuriosityModule]:
        """Return the trainable module or its episode-local frozen rollout."""
        return self._appraisal_curiosity_rollout or self.appraisal_curiosity

    def _start_appraisal_curiosity_rollout(self) -> None:
        """Create an online episode copy without mutating the frozen artifact."""
        if self.appraisal_curiosity is None:
            return
        # Episode-local copy of the pretrained signal. It is not checkpointed.
        self._appraisal_curiosity_rollout = copy.deepcopy(
            self.appraisal_curiosity
        )
        self._appraisal_curiosity_rollout.set_learning(True)
        self._appraisal_curiosity_rollout.reset_episode()

    def learning_state_hash(self) -> str:
        """Hash parameters, optimizer state, and learned normalizer state."""
        digest = hashlib.sha256()

        def update(value, prefix=""):
            digest.update(prefix.encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            elif isinstance(value, dict):
                for key in sorted(value, key=lambda item: str(item)):
                    update(value[key], f"{prefix}/{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    update(item, f"{prefix}/{index}")
            else:
                digest.update(repr(value).encode("utf-8"))

        named_modules = {
            "decision": self.network,
            "target": self.target_network,
            "appraisal": self.appraisal.network,
            "world_model": self.curiosity.world_model,
            "error_model": self.curiosity.error_model,
            "icm": self.curiosity.icm,
            "rnd": self.curiosity.rnd,
            "ensemble": self.curiosity.ensemble,
        }
        if self.appraisal_curiosity is not None:
            named_modules.update(
                {
                    "appraisal_world_model": self.appraisal_curiosity.world_model,
                    "appraisal_error_model": self.appraisal_curiosity.error_model,
                    "appraisal_icm": self.appraisal_curiosity.icm,
                    "appraisal_rnd": self.appraisal_curiosity.rnd,
                    "appraisal_ensemble": self.appraisal_curiosity.ensemble,
                }
            )
        for name, module in named_modules.items():
            if module is not None:
                update(module.state_dict(), name)
        optimizers = {
            "decision_optimizer": self.optimizer,
            "appraisal_optimizer": self.appraisal.optimizer,
        }
        for name, module in named_modules.items():
            optimizer = getattr(module, "optimizer", None) if module is not None else None
            if optimizer is not None:
                optimizers[f"{name}_optimizer"] = optimizer
        for name, optimizer in optimizers.items():
            if optimizer is not None:
                update(optimizer.state_dict(), name)
        update(
            {
                "running_scale": self.curiosity._running_scale,
                "scale_peak": self.curiosity._scale_peak,
                "appraisal_running_scale": (
                    self.appraisal_curiosity._running_scale
                    if self.appraisal_curiosity is not None
                    else 0.0
                ),
                "appraisal_scale_peak": (
                    self.appraisal_curiosity._scale_peak
                    if self.appraisal_curiosity is not None
                    else 0.0
                ),
            },
            "curiosity_normalizer",
        )
        return digest.hexdigest()

    @staticmethod
    def _curiosity_checkpoint(module: CuriosityModule) -> Dict[str, Any]:
        models = {}
        for name in ("world_model", "error_model", "icm", "rnd", "ensemble"):
            model = getattr(module, name)
            if model is not None:
                models[name] = model.state_dict()
        return {
            "mode": module.mode,
            "models": models,
            "running_scale": module._running_scale,
            "scale_peak": module._scale_peak,
        }

    @staticmethod
    def _load_curiosity_checkpoint(
        module: CuriosityModule, checkpoint: Dict[str, Any]
    ) -> None:
        if checkpoint.get("mode") != module.mode:
            raise ValueError(
                f"curiosity checkpoint mode {checkpoint.get('mode')!r} "
                f"does not match {module.mode!r}"
            )
        for name, state_dict in checkpoint.get("models", {}).items():
            model = getattr(module, name)
            if model is not None:
                model.load_state_dict(state_dict)
        module._running_scale = float(checkpoint.get("running_scale", 0.0))
        module._scale_peak = float(checkpoint.get("scale_peak", 0.0))

    def decision_checkpoint(self) -> Dict[str, Any]:
        """Return the decision-stage state reused by staged appraisal runs."""
        if self.network is None or self.target_network is None:
            raise RuntimeError("only learned decision agents have checkpoints")
        checkpoint = {
            "condition": self.condition.name,
            "seed": self.seed,
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "curiosity": self._curiosity_checkpoint(self.curiosity),
            "epsilon": float(self.epsilon),
            "learn_steps": int(self.learn_steps),
            "gradient_steps": int(self.gradient_steps),
            "state_hash": self.learning_state_hash(),
            "decision_hash": self.decision_state_hash(),
        }
        if self.appraisal_curiosity is not None:
            checkpoint["appraisal_curiosity"] = self._curiosity_checkpoint(
                self.appraisal_curiosity
            )
        return checkpoint

    def decision_state_hash(self) -> str:
        """Hash every artifact frozen during staged appraisal training."""
        digest = hashlib.sha256()

        def add_tensor(name: str, tensor: torch.Tensor) -> None:
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(value.numpy().tobytes())

        for prefix, module in (
            ("decision", self.network),
            ("target", self.target_network),
            ("world", self.curiosity.world_model),
            ("error", self.curiosity.error_model),
            ("icm", self.curiosity.icm),
            ("rnd", self.curiosity.rnd),
            ("ensemble", self.curiosity.ensemble),
        ):
            if module is not None:
                for name, tensor in sorted(module.state_dict().items()):
                    add_tensor(f"{prefix}/{name}", tensor)
        if self.appraisal_curiosity is not None:
            for prefix, module in (
                ("appraisal-world", self.appraisal_curiosity.world_model),
                ("appraisal-error", self.appraisal_curiosity.error_model),
                ("appraisal-icm", self.appraisal_curiosity.icm),
                ("appraisal-rnd", self.appraisal_curiosity.rnd),
                ("appraisal-ensemble", self.appraisal_curiosity.ensemble),
            ):
                if module is not None:
                    for name, tensor in sorted(module.state_dict().items()):
                        add_tensor(f"{prefix}/{name}", tensor)
        digest.update(repr(self.curiosity._running_scale).encode("ascii"))
        digest.update(repr(self.curiosity._scale_peak).encode("ascii"))
        if self.appraisal_curiosity is not None:
            digest.update(
                repr(self.appraisal_curiosity._running_scale).encode("ascii")
            )
            digest.update(
                repr(self.appraisal_curiosity._scale_peak).encode("ascii")
            )
        return digest.hexdigest()

    def load_decision_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load a base policy without replacing appraisal-specific modules."""
        if self.network is None or self.target_network is None:
            raise RuntimeError("only learned decision agents accept checkpoints")
        self.network.load_state_dict(checkpoint["network"], strict=True)
        self.target_network.load_state_dict(
            checkpoint["target_network"], strict=True
        )
        self._load_curiosity_checkpoint(self.curiosity, checkpoint["curiosity"])
        appraisal_checkpoint = checkpoint.get("appraisal_curiosity")
        if appraisal_checkpoint is not None:
            if self.appraisal_curiosity is None:
                raise ValueError(
                    "checkpoint contains an appraisal signal but agent does not"
                )
            self._load_curiosity_checkpoint(
                self.appraisal_curiosity, appraisal_checkpoint
            )
        self.epsilon = float(checkpoint["epsilon"])
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))

    def decay_epsilon(self) -> None:
        if self.condition.decision == "magus":
            return
        self.epsilon = max(
            self.training.epsilon_min, self.epsilon * self.training.epsilon_decay
        )

    def set_epsilon(self, value: float) -> None:
        if self.condition.decision != "magus":
            self.epsilon = value

    def set_initial_motive(self, state: MotivationalState) -> None:
        """Set the next episode's starting motive without moving the preference weight map."""
        self.initial_state = state.copy()

