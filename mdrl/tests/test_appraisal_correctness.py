"""Gradient-level checks for constrained residual appraisal."""

import numpy as np
import torch

from category.functors import AppraisalContext
from core.state import Stimulus
from core.state_profiles import create_reference_motivational_state
from mdrl.appraisal.residual import ResidualAppraisal
from mdrl.config import NUM_OUTCOMES
from mdrl.envs.curious_gridworld import OBSERVATION_DIM
from openpsi.appraisal import OpenPsiAppraisal

def residual_gradient(appraisal):
    return sum(
        float(parameter.grad.abs().sum().item())
        for parameter in appraisal.network.residual_head.parameters()
        if parameter.grad is not None
    )

def test_outcome_loss_backpropagates_through_corrected_appraisal():
    torch.manual_seed(31)
    appraisal = ResidualAppraisal(
        OpenPsiAppraisal(),
        OBSERVATION_DIM,
        enabled=True,
        lambda_pred=1.0,
        lambda_small=0.0,
        lambda_smooth=0.0,
        lambda_safe=0.0,
    )
    context = AppraisalContext(
        observation=np.linspace(-1.0, 1.0, OBSERVATION_DIM).astype(np.float32),
        learning=True,
    )
    appraisal.train_step(
        context,
        create_reference_motivational_state(),
        np.ones(NUM_OUTCOMES, dtype=np.float32),
        stimulus=Stimulus(0.7, 0.4, 0.6, 0.5),
    )
    assert residual_gradient(appraisal) > 0.0

def test_safety_loss_has_a_real_residual_gradient():
    torch.manual_seed(32)
    appraisal = ResidualAppraisal(
        OpenPsiAppraisal(),
        OBSERVATION_DIM,
        enabled=True,
        lambda_pred=0.0,
        lambda_small=0.0,
        lambda_smooth=0.0,
        lambda_safe=1.0,
    )
    context = AppraisalContext(
        observation=np.ones(OBSERVATION_DIM, dtype=np.float32),
        learning=True,
    )
    metrics = appraisal.train_step(
        context,
        create_reference_motivational_state(),
        np.zeros(NUM_OUTCOMES, dtype=np.float32),
        stimulus=Stimulus(0.8, 0.1, 1.0, 0.8),
    )
    assert metrics["appraisal_safety_loss"] > 0.0
    assert residual_gradient(appraisal) > 0.0

def test_smoothness_stores_the_matching_previous_state():
    appraisal = ResidualAppraisal(
        OpenPsiAppraisal(), OBSERVATION_DIM, enabled=True
    )
    first_state = create_reference_motivational_state()
    first_state.G[0] = 0.41
    context = AppraisalContext(
        observation=np.zeros(OBSERVATION_DIM, dtype=np.float32), learning=True
    )
    appraisal.train_step(
        context,
        first_state,
        np.zeros(NUM_OUTCOMES, dtype=np.float32),
        stimulus=Stimulus(0.2, 0.3, 0.1, 0.2),
    )
    assert appraisal._previous_state is not first_state
    np.testing.assert_array_equal(appraisal._previous_state.G, first_state.G)
    first_state.G[0] = 0.99
    assert appraisal._previous_state.G[0] == 0.41
