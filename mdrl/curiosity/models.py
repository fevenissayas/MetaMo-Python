"""Curiosity networks: world model F, error model H, plus RND and ensemble ablations."""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def _mlp(input_dim: int, hidden: Sequence[int], output_dim: int, final: str = "none") -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.append(nn.Linear(previous, width))
        layers.append(nn.ReLU())
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    if final == "sigmoid":
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

class IntrinsicCuriosityModule(nn.Module):
    """Pathak ICM: inverse-model features, with forward prediction error as the intrinsic reward."""

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        feature_dim: int = 32,
        hidden: Sequence[int] = (64, 64),
        learning_rate: float = 1e-3,
        forward_loss_weight: float = 0.2,
        reward_scale: float = 1.0,
    ):
        super().__init__()
        if num_actions < 2:
            raise ValueError("ICM requires at least two discrete actions")
        if not 0.0 <= forward_loss_weight <= 1.0:
            raise ValueError("forward_loss_weight must be in [0, 1]")

        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.forward_loss_weight = forward_loss_weight
        self.reward_scale = reward_scale

        self.encoder = _mlp(observation_dim, hidden, feature_dim)
        self.inverse_model = _mlp(2 * feature_dim, hidden, num_actions)
        self.forward_model = _mlp(feature_dim + num_actions, hidden, feature_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.last_inverse_loss = 0.0
        self.last_forward_loss = 0.0
        self.last_inverse_accuracy = 0.0

    def _action_tensor(self, action: int, batch_size: int, device: torch.device) -> torch.Tensor:
        action = int(action)
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action must be in [0, {self.num_actions}), got {action}")
        return torch.full((batch_size,), action, dtype=torch.long, device=device)

    def losses(
        self,
        observation: torch.Tensor,
        action: int,
        next_observation: torch.Tensor,
    ) -> tuple:
        """Return the paper's inverse loss, forward loss, and reward error."""
        phi = self.encoder(observation)
        next_phi = self.encoder(next_observation)
        actions = self._action_tensor(action, observation.shape[0], observation.device)
        one_hot = F.one_hot(actions, num_classes=self.num_actions).to(phi.dtype)

        inverse_logits = self.inverse_model(torch.cat([phi, next_phi], dim=-1))
        inverse_loss = F.cross_entropy(inverse_logits, actions)

        predicted_next_phi = self.forward_model(torch.cat([phi, one_hot], dim=-1))
        # Half squared error in feature space.
        forward_error = 0.5 * (predicted_next_phi - next_phi).pow(2).sum(dim=-1)
        forward_loss = forward_error.mean()
        return inverse_loss, forward_loss, forward_error, inverse_logits

    def update_and_score(
        self,
        observation: torch.Tensor,
        action: int,
        next_observation: torch.Tensor,
    ) -> float:
        """Train ICM on one transition and return its pre-update reward."""
        self.train()
        inverse_loss, forward_loss, forward_error, inverse_logits = self.losses(
            observation, action, next_observation
        )
        # Score before the update so this transition is not already learned.
        intrinsic_reward = self.reward_scale * float(forward_error.detach().mean().item())
        weight = self.forward_loss_weight
        loss = (1.0 - weight) * inverse_loss + weight * forward_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        actions = self._action_tensor(action, observation.shape[0], observation.device)
        self.last_inverse_loss = float(inverse_loss.detach().item())
        self.last_forward_loss = float(forward_loss.detach().item())
        self.last_inverse_accuracy = float(
            (inverse_logits.detach().argmax(dim=-1) == actions).float().mean().item()
        )
        return intrinsic_reward

    @torch.no_grad()
    def score(
        self,
        observation: torch.Tensor,
        action: int,
        next_observation: torch.Tensor,
    ) -> float:
        """Return intrinsic reward without changing parameters or diagnostics."""
        self.eval()
        _, _, forward_error, _ = self.losses(
            observation, action, next_observation
        )
        return self.reward_scale * float(forward_error.mean().item())

    def diagnostics(self) -> Dict[str, float]:
        return {
            "icm_inverse_loss": self.last_inverse_loss,
            "icm_forward_loss": self.last_forward_loss,
            "icm_inverse_accuracy": self.last_inverse_accuracy,
        }

class WorldModel(nn.Module):
    """F_omega: (e_o(o_t), phi(c_t)) -> predicted e_o(o_{t+1})."""

    def __init__(
        self,
        observation_dim: int,
        descriptor_dim: int,
        hidden: Sequence[int] = (96, 96),
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.network = _mlp(observation_dim + descriptor_dim, hidden, observation_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, observation: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observation, descriptor], dim=-1))

    @torch.no_grad()
    def error(
        self, observation: torch.Tensor, descriptor: torch.Tensor, next_observation: torch.Tensor
    ) -> float:
        self.eval()
        return float(self.loss_fn(self.forward(observation, descriptor), next_observation).item())

    def update(
        self, observation: torch.Tensor, descriptor: torch.Tensor, next_observation: torch.Tensor
    ) -> float:
        self.train()
        loss = self.loss_fn(self.forward(observation, descriptor), next_observation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

class ErrorModel(nn.Module):
    """
    H_nu: predicts the world model's error for a context/candidate pair.

    Trained more slowly than F so that its prediction reflects durable
    competence rather than the most recent gradient step.
    """

    def __init__(
        self,
        observation_dim: int,
        descriptor_dim: int,
        hidden: Sequence[int] = (48, 48),
        learning_rate: float = 5e-4,
    ):
        super().__init__()
        self.network = _mlp(observation_dim + descriptor_dim, hidden, 1, final="sigmoid")
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, observation: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observation, descriptor], dim=-1))

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, descriptor: torch.Tensor) -> float:
        self.eval()
        return float(self.forward(observation, descriptor).item())

    def update(
        self, observation: torch.Tensor, descriptor: torch.Tensor, actual_error: torch.Tensor
    ) -> float:
        self.train()
        loss = self.loss_fn(self.forward(observation, descriptor), actual_error)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

class RandomNetworkDistillation(nn.Module):
    """Ablation baseline: novelty as distillation error against a frozen target."""

    def __init__(
        self,
        observation_dim: int,
        embedding_dim: int = 32,
        hidden: Sequence[int] = (64, 64),
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.target = _mlp(observation_dim, hidden, embedding_dim)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.predictor = _mlp(observation_dim, hidden, embedding_dim)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_and_score(self, observation: torch.Tensor) -> float:
        with torch.no_grad():
            target = self.target(observation)
        loss = self.loss_fn(self.predictor(observation), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def score(self, observation: torch.Tensor) -> float:
        return float(self.loss_fn(self.predictor(observation), self.target(observation)).item())

class WorldModelEnsemble(nn.Module):
    """Ablation baseline: intrinsic reward as ensemble disagreement."""

    def __init__(
        self,
        observation_dim: int,
        descriptor_dim: int,
        size: int = 3,
        hidden: Sequence[int] = (64, 64),
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.members = nn.ModuleList(
            [
                _mlp(observation_dim + descriptor_dim, hidden, observation_dim)
                for _ in range(size)
            ]
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_and_score(
        self, observation: torch.Tensor, descriptor: torch.Tensor, next_observation: torch.Tensor
    ) -> float:
        features = torch.cat([observation, descriptor], dim=-1)
        predictions = torch.stack([member(features) for member in self.members])
        disagreement = float(predictions.var(dim=0).mean().item())

        loss = sum(self.loss_fn(prediction, next_observation) for prediction in predictions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return disagreement

    @torch.no_grad()
    def score(self, observation: torch.Tensor, descriptor: torch.Tensor) -> float:
        features = torch.cat([observation, descriptor], dim=-1)
        predictions = torch.stack([member(features) for member in self.members])
        return float(predictions.var(dim=0).mean().item())
