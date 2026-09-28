"""Replay of vector outcomes and motive context, scalarized at training time."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from mdrl.config import NUM_OUTCOMES
from mdrl.types import Transition

@dataclass
class ReplayBatch:
    """A minibatch, already padded over variable next-state candidate sets."""

    context: torch.Tensor            # [B, Z]
    descriptor: torch.Tensor         # [B, 1, D]
    outcome: torch.Tensor            # [B, K]
    lp_reward: torch.Tensor          # [B]
    safety_cost: torch.Tensor        # [B]
    projection_magnitude: torch.Tensor  # [B]
    next_context: torch.Tensor       # [B, Z]
    next_descriptors: torch.Tensor   # [B, N, D]
    next_mask: torch.Tensor          # [B, N] boolean
    next_costs: torch.Tensor         # [B, N]
    weights: torch.Tensor            # [B, K]
    beta: torch.Tensor               # [B]
    next_weights: torch.Tensor       # [B, K]
    next_beta: torch.Tensor          # [B]
    done: torch.Tensor               # [B]
    duration: torch.Tensor           # [B]
    relabeled: torch.Tensor          # [B] boolean, for diagnostics

class MotiveStratifiedReplay:
    """Ring buffer with optional motive-region stratification and relabeling."""

    def __init__(
        self,
        capacity: int = 40_000,
        num_strata: int = NUM_OUTCOMES,
        rng: Optional[np.random.Generator] = None,
        device: Optional[torch.device] = None,
    ):
        self.capacity = capacity
        self.num_strata = num_strata
        self.rng = rng or np.random.default_rng(0)
        self.device = device or torch.device("cpu")
        self._storage: List[Transition] = []
        self._position = 0
        self._strata: List[List[int]] = [[] for _ in range(num_strata)]

    def __len__(self) -> int:
        return len(self._storage)

    def is_ready(self, minimum: int) -> bool:
        return len(self._storage) >= minimum

    @staticmethod
    def stratum_of(weights: np.ndarray) -> int:
        """Motive region, taken as the dominant outcome component."""
        return int(np.argmax(weights))

    def push(self, transition: Transition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
            index = len(self._storage) - 1
        else:
            index = self._position
            evicted = self._storage[index]
            self._strata[self.stratum_of(evicted.weights)].remove(index)
            self._storage[index] = transition
            self._position = (self._position + 1) % self.capacity
        self._strata[self.stratum_of(transition.weights)].append(index)

    def _sample_indices(self, batch_size: int, stratified_fraction: float) -> np.ndarray:
        """Sample part of the batch uniformly and part evenly across motive strata."""
        total = len(self._storage)
        stratified_count = int(round(batch_size * float(np.clip(stratified_fraction, 0.0, 1.0))))
        uniform_count = batch_size - stratified_count

        indices = list(self.rng.integers(0, total, size=uniform_count))

        populated = [stratum for stratum in self._strata if stratum]
        if populated and stratified_count > 0:
            per_stratum = max(1, stratified_count // len(populated))
            for stratum in populated:
                take = min(per_stratum, len(stratum))
                chosen = self.rng.choice(len(stratum), size=take, replace=take > len(stratum))
                indices.extend(stratum[int(position)] for position in np.atleast_1d(chosen))

        indices = indices[:batch_size]
        while len(indices) < batch_size:
            indices.append(int(self.rng.integers(0, total)))
        return np.asarray(indices, dtype=np.int64)

    def _sample_preference(self) -> np.ndarray:
        """A fresh weight vector for preference relabeling."""
        return self.rng.dirichlet(np.ones(NUM_OUTCOMES)).astype(np.float32)

    def sample(
        self,
        batch_size: int,
        stratified_fraction: float = 0.5,
        relabel_fraction: float = 0.0,
    ) -> ReplayBatch:
        indices = self._sample_indices(batch_size, stratified_fraction)
        items = [self._storage[int(index)] for index in indices]

        max_candidates = max(item.next_descriptors.shape[0] for item in items)
        descriptor_dim = items[0].descriptor.shape[0]

        next_descriptors = np.zeros(
            (len(items), max_candidates, descriptor_dim), dtype=np.float32
        )
        next_mask = np.zeros((len(items), max_candidates), dtype=bool)
        next_costs = np.zeros((len(items), max_candidates), dtype=np.float32)
        for row, item in enumerate(items):
            count = item.next_descriptors.shape[0]
            next_descriptors[row, :count] = item.next_descriptors
            next_mask[row, :count] = True
            next_costs[row, :count] = item.next_safety_costs

        weights = np.stack([item.weights for item in items]).astype(np.float32)
        next_weights = np.stack([item.next_weights for item in items]).astype(np.float32)
        betas = np.array([item.beta for item in items], dtype=np.float32)
        next_betas = np.array([item.next_beta for item in items], dtype=np.float32)

        relabeled = np.zeros(len(items), dtype=bool)
        if relabel_fraction > 0.0:
            draw = self.rng.random(len(items)) < relabel_fraction
            for row in np.flatnonzero(draw):
                preference = self._sample_preference()
                weights[row] = preference
                next_weights[row] = preference
                relabeled[row] = True

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

        return ReplayBatch(
            context=tensor(np.stack([item.context for item in items])),
            descriptor=tensor(np.stack([item.descriptor for item in items])).unsqueeze(1),
            outcome=tensor(np.stack([item.outcome for item in items])),
            lp_reward=tensor([item.lp_reward for item in items]),
            safety_cost=tensor([item.safety_cost for item in items]),
            projection_magnitude=tensor([item.projection_magnitude for item in items]),
            next_context=tensor(np.stack([item.next_context for item in items])),
            next_descriptors=tensor(next_descriptors),
            next_mask=tensor(next_mask, dtype=torch.bool),
            next_costs=tensor(next_costs),
            weights=tensor(weights),
            beta=tensor(betas),
            next_weights=tensor(next_weights),
            next_beta=tensor(next_betas),
            done=tensor([float(item.done) for item in items]),
            duration=tensor([float(item.duration) for item in items]),
            relabeled=tensor(relabeled, dtype=torch.bool),
        )

    def stratum_counts(self) -> Sequence[int]:
        return [len(stratum) for stratum in self._strata]

    def records(self) -> Sequence[Transition]:
        """Read-only view of stored transitions, for diagnostics and tests."""
        return tuple(self._storage)
