"""Descriptor-conditioned Q network, plus a fixed-output baseline with matched capacity."""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

def _tower(input_dim: int, hidden: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    previous = input_dim
    for width in hidden:
        layers.append(nn.Linear(previous, width))
        layers.append(nn.ReLU())
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class CandidateQNetwork(nn.Module):
    """Candidate-conditioned scorer with optional vector-valued task heads."""

    def __init__(
        self,
        context_dim: int,
        descriptor_dim: int,
        num_outcomes: int,
        hidden_dim: int = 128,
        context_out: int = 96,
        candidate_out: int = 64,
        vector_values: bool = True,
        goal_residual_dim: int = 0,
    ):
        super().__init__()
        self.vector_values = vector_values
        self.num_outcomes = num_outcomes
        self.task_out = num_outcomes if vector_values else 1
        self.goal_residual_dim = goal_residual_dim

        self.context_tower = _tower(context_dim, (hidden_dim, hidden_dim), context_out)
        self.candidate_tower = _tower(descriptor_dim, (hidden_dim // 2,), candidate_out)

        self.projection = nn.Linear(context_out, candidate_out)
        joint_dim = context_out + 2 * candidate_out

        self.task_head = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.task_out),
        )
        self.lp_head = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Optional goal residual, still passed through the stabilizer.
        self.goal_residual_head: Optional[nn.Module] = None
        if goal_residual_dim > 0:
            self.goal_residual_head = nn.Sequential(
                nn.Linear(joint_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, goal_residual_dim),
                nn.Tanh(),
            )

    def _joint(self, context: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        """Build [h^x, h^a, h^x * h^a] for every candidate in the set."""
        h_context = self.context_tower(context)                    # [B, C]
        h_candidate = self.candidate_tower(descriptors)            # [B, N, A]
        num_candidates = h_candidate.shape[1]

        expanded = h_context.unsqueeze(1).expand(-1, num_candidates, -1)
        projected = self.projection(h_context).unsqueeze(1).expand(-1, num_candidates, -1)
        return torch.cat([expanded, h_candidate, projected * h_candidate], dim=-1)

    def forward(
        self, context: torch.Tensor, descriptors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (Q_task, Q_LP) with shapes [B, N, K] and [B, N].

        In scalar mode K is 1 and Q_LP is identically zero, because the
        intrinsic reward has already been folded into the scalar training target.
        """
        joint = self._joint(context, descriptors)
        q_task = self.task_head(joint)
        if self.vector_values:
            q_lp = self.lp_head(joint).squeeze(-1)
        else:
            q_lp = torch.zeros(q_task.shape[:2], device=q_task.device)
        return q_task, q_lp

    def goal_residual(self, context: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        if self.goal_residual_head is None:
            raise RuntimeError("network was built without a goal-residual head")
        return self.goal_residual_head(self._joint(context, descriptors))

class FixedOutputQNetwork(nn.Module):
    """
    Conventional DQN head: one value per primitive action index.

    Cannot represent a candidate set that changes size or meaning, which is the
    limitation condition B5 and P1 are meant to expose (Section 8.2).
    """

    def __init__(
        self,
        context_dim: int,
        num_actions: int,
        num_outcomes: int,
        hidden_dim: int = 128,
        context_out: int = 96,
        vector_values: bool = True,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.vector_values = vector_values
        self.task_out = num_outcomes if vector_values else 1

        self.context_tower = _tower(context_dim, (hidden_dim, hidden_dim), context_out)
        self.task_head = nn.Sequential(
            nn.Linear(context_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * self.task_out),
        )
        self.lp_head = nn.Sequential(
            nn.Linear(context_out, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(
        self, context: torch.Tensor, descriptors: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        hidden = self.context_tower(context)
        q_task = self.task_head(hidden).view(batch, self.num_actions, self.task_out)
        if self.vector_values:
            q_lp = self.lp_head(hidden)
        else:
            q_lp = torch.zeros(batch, self.num_actions, device=context.device)
        return q_task, q_lp
