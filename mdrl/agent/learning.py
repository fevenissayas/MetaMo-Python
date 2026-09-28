"""Replay storage and the Double-DQN update."""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from mdrl.config import KAPPA_SAFETY_COST
from mdrl.types import Candidate, DecisionContext, Transition

class AgentLearning:
    def _store(
        self,
        context: DecisionContext,
        chosen: Candidate,
        outcome: np.ndarray,
        result,
        next_context: DecisionContext,
        next_candidates: Sequence[Candidate],
        projection_magnitude: float,
    ) -> None:
        include = self.condition.motive_in_context
        next_descriptors = np.stack([c.descriptor for c in next_candidates]).astype(np.float32)
        next_costs = np.array(
            [c.safety_cost + 0.5 * c.risk_estimate for c in next_candidates], dtype=np.float32
        )
        self.replay.push(
            Transition(
                context=context.context_vector(include),
                descriptor=chosen.descriptor,
                goal_vector=context.state.G.copy(),
                modulator_vector=context.state.M.copy(),
                weights=context.weights.copy(),
                beta=context.beta,
                outcome=outcome.astype(np.float32),
                lp_reward=float(result.lp_reward),
                safety_cost=float(result.safety_cost),
                next_context=next_context.context_vector(include),
                next_descriptors=next_descriptors,
                next_weights=next_context.weights.copy(),
                next_beta=next_context.beta,
                next_safety_costs=next_costs,
                done=bool(result.done),
                duration=int(result.duration),
                certified=bool(chosen.is_certified),
                projection_magnitude=float(projection_magnitude),
            )
        )

    def _learn(self) -> Optional[float]:
        if self.replay is None or not self.replay.is_ready(self.training.min_buffer_size):
            return None

        batch = self.replay.sample(
            self.training.batch_size,
            stratified_fraction=self.training.stratified_fraction,
            relabel_fraction=self.training.relabel_fraction,
        )

        self.network.eval()
        with torch.no_grad():
            online_task, online_lp = self._forward(self.network, batch.next_context, batch.next_descriptors)
            if self.condition.uses_vector_values:
                selection = (
                    torch.einsum("bnk,bk->bn", online_task, batch.next_weights)
                    + batch.next_beta.unsqueeze(1) * online_lp
                )
            else:
                selection = online_task[..., 0]
            selection = selection - KAPPA_SAFETY_COST * batch.next_costs
            selection = selection.masked_fill(~batch.next_mask, -1e9)
            best = selection.argmax(dim=1)

            target_task, target_lp = self._forward(
                self.target_network, batch.next_context, batch.next_descriptors
            )
            index = best.view(-1, 1, 1).expand(-1, 1, target_task.shape[-1])
            next_task = target_task.gather(1, index).squeeze(1)
            next_lp = target_lp.gather(1, best.view(-1, 1)).squeeze(1)

            # Discount by the option's duration.
            discount = torch.pow(
                torch.tensor(self.training.gamma, device=self.device), batch.duration
            )
            not_done = 1.0 - batch.done

            if self.condition.uses_vector_values:
                task_target = batch.outcome + discount.unsqueeze(1) * not_done.unsqueeze(1) * next_task
                lp_target = batch.lp_reward + discount * not_done * next_lp
            else:
                scalar_reward = (
                    (batch.outcome * batch.weights).sum(dim=1)
                    + batch.beta * batch.lp_reward
                    - KAPPA_SAFETY_COST * batch.safety_cost
                )
                task_target = (scalar_reward + discount * not_done * next_task[:, 0]).unsqueeze(1)
                lp_target = None

        self.network.train()
        predicted_task, predicted_lp = self._forward(
            self.network, batch.context, batch.descriptor, single=True
        )
        loss = nn.functional.smooth_l1_loss(predicted_task, task_target)
        if lp_target is not None:
            loss = loss + 0.5 * nn.functional.smooth_l1_loss(predicted_lp, lp_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.training.grad_clip)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.training.target_update_period == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.losses.append(float(loss.item()))
        return float(loss.item())

    def _forward(self, network, context, descriptors, single: bool = False):
        """
        Run a network over a batch.

        A fixed-output network ignores descriptors, so the stored primitive
        index is recovered from the descriptor's direction block.
        """
        if self.condition.decision == "candidate_dqn":
            q_task, q_lp = network(context, descriptors)
        else:
            q_task, q_lp = network(context)
            if single:
                indices = self._primitive_indices(descriptors.squeeze(1))
                gather_index = indices.view(-1, 1, 1).expand(-1, 1, q_task.shape[-1])
                q_task = q_task.gather(1, gather_index)
                q_lp = q_lp.gather(1, indices.view(-1, 1))
        if single:
            return q_task.squeeze(1), q_lp.squeeze(1)
        return q_task, q_lp

    @staticmethod
    def _primitive_indices(descriptors: torch.Tensor) -> torch.Tensor:
        from mdrl.candidates.descriptors import BLOCK_OFFSETS

        start, end = BLOCK_OFFSETS["direction"]
        return descriptors[:, start:end].argmax(dim=1)

