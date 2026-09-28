"""Construct the components selected by a condition."""

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from category.bimonad import MetaMoPseudoBimonad, RuntimeValidationPolicy
from core.state import MotivationalState
from core.state_profiles import create_reference_motivational_state
from dynamics.stability import MetaMoStabilizationPolicy, NoStabilizationPolicy
from magus.decision import MagusDecision
from mdrl.agent.records import CONTEXT_DIM, StepRecord
from mdrl.appraisal.residual import ResidualAppraisal
from mdrl.candidates.adapter import SubRepAdapter
from mdrl.candidates.descriptors import DESCRIPTOR_DIM
from mdrl.candidates.skills import default_skill_library
from mdrl.config import KAPPA_SAFETY_COST, NUM_OUTCOMES, ConditionSpec, TrainingConfig
from mdrl.curiosity.module import CURIOSITY_FEATURE_DIM, CuriosityModule
from mdrl.decision.dqn_decision import DQNDecisionMonad, FixedOutputDecisionMonad
from mdrl.decision.preference import MotivePreference
from mdrl.envs.curious_gridworld import NUM_ACTIONS, OBSERVATION_DIM
from mdrl.nets.candidate_q import CandidateQNetwork, FixedOutputQNetwork
from mdrl.replay.buffer import MotiveStratifiedReplay
from mdrl.stabilizer import MetaMoStabilizer
from openpsi.appraisal import OpenPsiAppraisal

class AgentBuild:
    def __init__(
        self,
        condition: ConditionSpec,
        training: Optional[TrainingConfig] = None,
        seed: int = 0,
        skill_library: Optional[Sequence] = None,
        device: Optional[torch.device] = None,
    ):
        self.condition = condition
        self.training = training or TrainingConfig()
        self.seed = seed
        self.device = device or torch.device("cpu")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.curiosity = CuriosityModule(
            observation_dim=OBSERVATION_DIM,
            descriptor_dim=DESCRIPTOR_DIM,
            num_actions=NUM_ACTIONS,
            mode=condition.curiosity,
            world_model_lr=self.training.world_model_lr,
            error_model_lr=self.training.error_model_lr,
            icm_feature_dim=self.training.icm_feature_dim,
            icm_forward_loss_weight=self.training.icm_forward_loss_weight,
            icm_reward_scale=self.training.icm_reward_scale,
            device=self.device,
            seed=seed,
        )

        self.appraisal_curiosity: Optional[CuriosityModule] = None
        self._appraisal_curiosity_rollout: Optional[CuriosityModule] = None
        if condition.appraisal_signal != "none":
            if condition.appraisal_signal not in ("lp_h", "icm"):
                raise ValueError(
                    f"unsupported appraisal signal {condition.appraisal_signal!r}"
                )
            self.appraisal_curiosity = CuriosityModule(
                observation_dim=OBSERVATION_DIM,
                descriptor_dim=DESCRIPTOR_DIM,
                num_actions=NUM_ACTIONS,
                mode=condition.appraisal_signal,
                world_model_lr=self.training.world_model_lr,
                error_model_lr=self.training.error_model_lr,
                icm_feature_dim=self.training.icm_feature_dim,
                icm_forward_loss_weight=self.training.icm_forward_loss_weight,
                icm_reward_scale=self.training.icm_reward_scale,
                device=self.device,
                seed=seed + 100_003,
            )

        # Own seed so residual init does not depend on the curiosity module's RNG use.
        torch.manual_seed(seed + 200_003)
        self.appraisal = ResidualAppraisal(
            base=OpenPsiAppraisal(),
            observation_dim=OBSERVATION_DIM,
            enabled=condition.appraisal == "residual",
            curiosity_dim=CURIOSITY_FEATURE_DIM,
            device=self.device,
        )

        library = skill_library if skill_library is not None else default_skill_library(
            progress_probe=self.curiosity.progress_probe
        )
        self.adapter = SubRepAdapter(
            skill_library=library,
            use_skills=condition.uses_skills,
            certificate_gate=condition.certificate_gate,
        )

        self.initial_state = create_reference_motivational_state()
        self.preference = MotivePreference(
            dynamic=condition.dynamic_weights and condition.uses_metamo,
            reference_state=self.initial_state,
        )

        self.stabilizer = MetaMoStabilizer(enabled=condition.stabilizer)
        self.network: Optional[nn.Module] = None
        self.target_network: Optional[nn.Module] = None
        self.optimizer = None
        self.replay: Optional[MotiveStratifiedReplay] = None

        if condition.decision == "magus":
            self.decision = MagusDecision()
            self.epsilon = 0.02  # breaks deterministic oscillation only
        else:
            # Re-seed so matched conditions share one decision-network initialization.
            torch.manual_seed(seed)
            self._build_networks()
            monad_cls = (
                DQNDecisionMonad
                if condition.decision == "candidate_dqn"
                else FixedOutputDecisionMonad
            )
            self.decision = monad_cls(
                network=self.network,
                include_motive_in_context=condition.motive_in_context,
                vector_values=condition.uses_vector_values,
                kappa=KAPPA_SAFETY_COST,
                goal_residual_scale=0.15 if condition.goal_update == "residual" else 0.0,
                device=self.device,
                rng=self.rng,
            )
            self.epsilon = self.training.epsilon_start
            self.replay = MotiveStratifiedReplay(
                capacity=self.training.buffer_size, rng=self.rng, device=self.device
            )

        stabilization_policy = (
            MetaMoStabilizationPolicy(blend=True)
            if condition.stabilizer
            else NoStabilizationPolicy()
        )
        self.metamo = MetaMoPseudoBimonad(
            appraisal=self.appraisal,
            decision=self.decision,
            stabilization_policy=stabilization_policy,
            # Record law violations without applying a second fallback.
            validation_policy=RuntimeValidationPolicy(
                enabled=True, fallback_on_failure=False
            ),
        )

        self.state = self.initial_state.copy()
        self.learn_steps = 0
        self.gradient_steps = 0
        self.losses: List[float] = []
        self.episode_records: List[StepRecord] = []
        self.motivational_trace: List[MotivationalState] = []
        self._step_index = 0
        self.learning_enabled = True
        self.decision_learning_enabled = True
        self.appraisal_learning_enabled = condition.appraisal == "residual"
        self.policy_curiosity_learning_enabled = True
        self.appraisal_curiosity_learning_enabled = (
            self.appraisal_curiosity is not None
        )

    def _build_networks(self) -> None:
        condition = self.condition
        shared = dict(
            num_outcomes=NUM_OUTCOMES,
            hidden_dim=self.training.hidden_dim,
            context_out=self.training.context_dim,
            vector_values=condition.uses_vector_values,
        )
        if condition.decision == "candidate_dqn":
            self.network = CandidateQNetwork(
                context_dim=CONTEXT_DIM,
                descriptor_dim=DESCRIPTOR_DIM,
                candidate_out=self.training.candidate_dim,
                goal_residual_dim=NUM_GOALS if condition.goal_update == "residual" else 0,
                **shared,
            ).to(self.device)
            self.target_network = CandidateQNetwork(
                context_dim=CONTEXT_DIM,
                descriptor_dim=DESCRIPTOR_DIM,
                candidate_out=self.training.candidate_dim,
                goal_residual_dim=NUM_GOALS if condition.goal_update == "residual" else 0,
                **shared,
            ).to(self.device)
        else:
            self.network = FixedOutputQNetwork(
                context_dim=CONTEXT_DIM, num_actions=NUM_ACTIONS, **shared
            ).to(self.device)
            self.target_network = FixedOutputQNetwork(
                context_dim=CONTEXT_DIM, num_actions=NUM_ACTIONS, **shared
            ).to(self.device)

        self.target_network.load_state_dict(self.network.state_dict())
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.training.learning_rate
        )

