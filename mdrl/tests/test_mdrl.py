"""Invariants for the MetaMo deep-RL integration. Run with ``python -m pytest mdrl/tests -q``."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config import G_CURIO, G_IND, NUM_GOALS, NUM_MODULATORS, THETA_SAFE
from core.state import MotivationalState
from dynamics.stability import is_in_safe_region
from mdrl.agent import CONTEXT_DIM, MetaMoDRLAgent
from mdrl.bench.conditions import ALL_CONDITIONS, BY_NAME, CORE_CONDITIONS, GROUPS, resolve
from mdrl.bench.protocol import (
    CURIOSITY_PROFILE,
    SAFETY_PROFILE,
    default_regimes,
    sample_training_motive,
    training_skill_library,
)
from mdrl.bench.runner import evaluate_regime, train
from mdrl.bench.report import _values, contrast_table, transfer_table
from mdrl.bench.stats import (
    bootstrap_summary,
    cliffs_delta,
    compare,
    hedges_g,
    paired_compare,
)
from mdrl.candidates.adapter import SubRepAdapter
from mdrl.candidates.descriptors import BLOCK_OFFSETS, DESCRIPTOR_DIM, SkillClass, build_descriptor
from mdrl.candidates.skills import default_skill_library
from mdrl.config import (
    EXTERNAL_EVALUATION_WEIGHTS,
    NUM_OUTCOMES,
    OUT_SAFETY,
    ConditionSpec,
    TrainingConfig,
)
from mdrl.curiosity.module import CuriosityModule
from mdrl.curiosity.models import IntrinsicCuriosityModule
from mdrl.envs.curious_gridworld import (
    ACTION_NAMES,
    NUM_ACTIONS,
    OBSERVATION_DIM,
    CuriousGridWorld,
    LayoutVariant,
    Zone,
)
from mdrl.metrics import EpisodeMetrics, collect_episode_metrics, counterfactual_sensitivity
from mdrl.replay.buffer import MotiveStratifiedReplay
from mdrl.types import Candidate, CandidateKind, Transition

TINY = TrainingConfig(
    train_episodes=4,
    eval_episodes=2,
    max_steps=40,
    min_buffer_size=32,
    batch_size=16,
)

def tiny_agent(name: str = "P1", seed: int = 0) -> MetaMoDRLAgent:
    return MetaMoDRLAgent(BY_NAME[name], TINY, seed=seed)

def fresh_env(**kwargs) -> CuriousGridWorld:
    defaults = dict(seed=0, layout=LayoutVariant.train_variants()[0], max_steps=40)
    defaults.update(kwargs)
    return CuriousGridWorld(**defaults)

def _inspect_all(env: CuriousGridWorld, zone: Zone) -> None:
    """
    Walk the agent onto every cell of a zone.

    Inspection is credited by `step`, on the cell the agent lands on, so each
    target is reached by standing next to it and stepping in.
    """
    from mdrl.envs.curious_gridworld import GRID_SIZE

    for row, col in env.zone_cells[zone]:
        if row + 1 < GRID_SIZE:
            env.agent_pos = (row + 1, col)
            env.step(0)  # UP
        else:
            env.agent_pos = (row - 1, col)
            env.step(1)  # DOWN

class TestEnvironment:
    def test_observation_shape_matches_declared_dim(self):
        env = fresh_env()
        observation = env.reset()
        assert observation["features"].shape == (OBSERVATION_DIM,)
        assert np.isfinite(observation["features"]).all()

    def test_outcome_is_a_vector_not_a_scalar(self):
        env = fresh_env()
        env.reset()
        _, outcome, _, _ = env.step(0)
        assert outcome.shape == (NUM_OUTCOMES,)

    def test_two_routes_exist_and_differ_in_risk(self):
        """The safe/risky tradeoff has to be real or RQ5 is untestable."""
        env = fresh_env()
        env.reset()
        layout = env.layout
        shortcut = (layout.wall_row, layout.shortcut_col)
        safe = (layout.wall_row, layout.safe_col)
        assert shortcut not in env.lava_cells, "the shortcut must be passable"
        assert safe not in env.lava_cells, "the safe crossing must be passable"
        shortcut_neighbours = [
            (shortcut[0], shortcut[1] - 1),
            (shortcut[0], shortcut[1] + 1),
        ]
        assert any(cell in env.lava_cells for cell in shortcut_neighbours), (
            "the shortcut must be flanked by lava, or it is not risky"
        )

    def test_noisy_zone_is_genuinely_unpredictable(self):
        """
        The noisy zone must not be learnable, or the noisy-television test is
        measuring nothing.
        """
        env = fresh_env(noisy_zone_enabled=True)
        env.reset()
        cells = env.zone_cells.get(Zone.NOISY, ())
        assert cells, "the layout must contain a noisy zone"
        env.agent_pos = next(iter(cells))
        observations = []
        for _ in range(40):
            env.agent_pos = next(iter(cells))
            observations.append(env._observation()["features"].copy())
        variation = np.stack(observations).std(axis=0).max()
        assert variation > 0.0, "the noisy zone must vary from the same position"

    def test_deterministic_variant_removes_the_noise(self):
        env = fresh_env(noisy_zone_enabled=False)
        env.reset()
        cells = env.zone_cells.get(Zone.NOISY, ())
        observations = []
        for _ in range(20):
            env.agent_pos = next(iter(cells))
            observations.append(env._observation()["features"].copy())
        assert np.stack(observations).std(axis=0).max() < 1e-6  # float32 rounding only

    def test_same_seed_same_trajectory(self):
        def rollout():
            env = fresh_env(seed=7)
            env.reset()
            trace = []
            for action in [0, 1, 2, 3] * 6:
                observation, outcome, done, _ = env.step(action)
                trace.append((observation["pos"], tuple(outcome)))
                if done:
                    break
            return trace

        assert rollout() == rollout()

    def test_rule_switch_is_recorded(self):
        env = fresh_env(max_steps=60, dynamic_switch_step=10)
        env.reset()
        for _ in range(15):
            env.step(0)
        assert env.rule_switch_steps == [10]

    def test_heldout_layouts_differ_from_training_layouts(self):
        train = {variant.name for variant in LayoutVariant.train_variants()}
        heldout = {variant.name for variant in LayoutVariant.heldout_variants()}
        assert not train & heldout
        train_shapes = {
            (v.wall_row, v.shortcut_col, v.mineral_home) for v in LayoutVariant.train_variants()
        }
        for variant in LayoutVariant.heldout_variants():
            assert (
                variant.wall_row,
                variant.shortcut_col,
                variant.mineral_home,
            ) not in train_shapes

class TestCandidates:
    def test_descriptor_dimension_is_the_sum_of_its_blocks(self):
        total = sum(end - start for start, end in BLOCK_OFFSETS.values())
        assert total == DESCRIPTOR_DIM

    def test_descriptor_is_bounded(self):
        descriptor = build_descriptor(
            kind_index=1,
            skill_class=SkillClass.GOTO_MINERAL_DIRECT,
            direction=-1,
            expected_duration=100.0,
            risk_estimate=5.0,
            safety_cost=5.0,
            mu_hat=np.array([9.0, -9.0, 9.0, -9.0], dtype=np.float32),
            certified=True,
            goal_relation=(4.0, -4.0, 4.0),
        )
        assert descriptor.shape == (DESCRIPTOR_DIM,)
        assert np.abs(descriptor).max() <= 1.0 + 1e-6

    def test_distinct_skills_get_distinct_descriptors(self):
        adapter = SubRepAdapter(default_skill_library(), use_skills=True)
        env = fresh_env()
        observation = env.reset()
        candidates = adapter.generate_candidates(observation, sample_training_motive(np.random.default_rng(0)))
        descriptors = [tuple(np.round(c.descriptor, 6)) for c in candidates]
        assert len(set(descriptors)) == len(descriptors)

    def test_primitives_always_survive_the_certificate_gate(self):
        """The fallback guarantee: a rejected certificate must never empty C+."""
        adapter = SubRepAdapter(default_skill_library(), use_skills=True, certificate_gate=True)
        env = fresh_env()
        observation = env.reset()
        hostile = MotivationalState(G=np.zeros(NUM_GOALS), M=np.zeros(NUM_MODULATORS))
        valid = adapter.valid_candidates(observation, hostile)
        primitives = [c for c in valid if c.kind is CandidateKind.PRIMITIVE]
        assert len(primitives) == NUM_ACTIONS

    def test_certificate_support_actually_gates(self):
        """A skill outside its motive support must be excluded when gating is on."""
        library = default_skill_library()
        adapter = SubRepAdapter(library, use_skills=True, certificate_gate=True)
        env = fresh_env()
        observation = env.reset()

        low_curiosity = sample_training_motive(np.random.default_rng(0))
        low_curiosity.G[G_CURIO] = 0.0
        high_curiosity = low_curiosity.copy()
        high_curiosity.G[G_CURIO] = 0.9

        def skill_ids(state):
            return {
                c.id
                for c in adapter.valid_candidates(observation, state)
                if c.kind is CandidateKind.SKILL
            }

        assert skill_ids(low_curiosity) < skill_ids(high_curiosity) or skill_ids(
            low_curiosity
        ) != skill_ids(high_curiosity)

    def test_disabling_the_gate_admits_more(self):
        env = fresh_env()
        observation = env.reset()
        state = sample_training_motive(np.random.default_rng(0))
        state.G[G_CURIO] = 0.0
        gated = SubRepAdapter(default_skill_library(), use_skills=True, certificate_gate=True)
        open_ = SubRepAdapter(default_skill_library(), use_skills=True, certificate_gate=False)
        assert len(open_.valid_candidates(observation, state)) >= len(
            gated.valid_candidates(observation, state)
        )

    def test_skill_execution_is_temporally_extended(self):
        adapter = SubRepAdapter(default_skill_library(), use_skills=True)
        env = fresh_env(max_steps=200)
        observation = env.reset()
        state = sample_training_motive(np.random.default_rng(0))
        skills = [
            c
            for c in adapter.valid_candidates(observation, state)
            if c.kind is CandidateKind.SKILL
        ]
        assert skills, "at least one skill must be initiable from the start state"
        durations = []
        for skill in skills:
            env.reset()
            result = adapter.execute(skill, env, observation, gamma=0.97)
            durations.append(result.duration)
        assert max(durations) > 1, "skills must consume more than one primitive step"

    def test_execute_always_advances_the_clock(self):
        """Even a skill that cannot move must not stall the loop."""
        adapter = SubRepAdapter(default_skill_library(), use_skills=True)
        env = fresh_env()
        observation = env.reset()
        state = sample_training_motive(np.random.default_rng(0))
        for candidate in adapter.valid_candidates(observation, state):
            env.reset()
            result = adapter.execute(candidate, env, observation, gamma=0.97)
            assert result.duration >= 1

    def test_effect_models_stop_promising_exhausted_information(self):
        """
        A candidate must not advertise an information reward for a cell that has
        already been inspected: a scorer that believes it will keep circling the
        same exhausted zone forever.
        """
        adapter = SubRepAdapter(default_skill_library(), use_skills=True)
        env = fresh_env(max_steps=400)
        observation = env.reset()
        state = sample_training_motive(np.random.default_rng(0))
        target = env.zone_cells[Zone.LEARNABLE][0]
        neighbour = (target[0], target[1] - 1)
        env.agent_pos = neighbour
        observation = env._observation()

        def info_claim(observation):
            return max(
                float(c.mu_hat[3])
                for c in adapter.generate_candidates(observation, state)
                if c.kind is CandidateKind.PRIMITIVE
            )

        before = info_claim(observation)
        assert before > 0.0, "an uninspected zone cell must look informative"

        env.agent_pos = neighbour
        observation, _, _, _ = env.step(3)  # step RIGHT onto the zone cell
        env.agent_pos = neighbour
        after = info_claim(env._observation())
        assert after < before

    def test_inspect_skill_disengages_from_an_exhausted_zone(self):
        skill = [
            s for s in default_skill_library() if s.skill_class is SkillClass.INSPECT_LEARNABLE
        ][0]
        env = fresh_env(max_steps=400)
        observation = env.reset()
        assert skill.can_initiate(observation)
        _inspect_all(env, Zone.LEARNABLE)
        env.agent_pos = (0, 0)
        assert not skill.can_initiate(env._observation())

    def test_the_noisy_trap_stays_available_when_exhausted(self):
        """
        The noisy zone pays no information reward, so an exhaustion rule keyed to
        information alone would delete the curiosity trap entirely.
        """
        skill = [s for s in default_skill_library() if s.skill_class is SkillClass.INSPECT_NOISY][0]
        env = fresh_env(max_steps=400)
        env.reset()
        _inspect_all(env, Zone.NOISY)
        env.agent_pos = (0, 0)
        assert skill.can_initiate(env._observation())

    def test_safe_route_skill_avoids_lava(self):
        """The certified safe skill must actually be safe, or certificates lie."""
        library = [s for s in default_skill_library() if s.skill_class is SkillClass.GOTO_MINERAL_SAFE]
        adapter = SubRepAdapter(library, use_skills=True)
        env = fresh_env(max_steps=200)
        observation = env.reset()
        state = sample_training_motive(np.random.default_rng(0))
        safe = [c for c in adapter.valid_candidates(observation, state) if c.kind is CandidateKind.SKILL]
        assert safe
        adapter.execute(safe[0], env, observation, gamma=0.97)
        assert env.lava_steps == 0

class TestCuriosity:
    def _module(self, mode: str) -> CuriosityModule:
        return CuriosityModule(
            observation_dim=OBSERVATION_DIM,
            descriptor_dim=DESCRIPTOR_DIM,
            mode=mode,
            seed=0,
        )

    def test_learning_progress_is_positive_on_a_learnable_stream(self):
        """A predictable transition should yield net positive progress."""
        module = self._module("lp_h")
        rng = np.random.default_rng(0)
        observation = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
        descriptor = rng.normal(size=DESCRIPTOR_DIM).astype(np.float32)
        nxt = (observation + 0.1).astype(np.float32)
        progress = [
            module.update_and_measure_progress(observation, descriptor, nxt, zone="learnable")
            for _ in range(200)
        ]
        assert float(np.sum(progress)) > 0.0

    def test_learning_progress_decays_on_an_unlearnable_stream(self):
        """
        The noisy-television condition: progress on pure noise must not keep
        paying out the way progress on structure does.
        """
        module = self._module("lp_h")
        rng = np.random.default_rng(1)
        descriptor = rng.normal(size=DESCRIPTOR_DIM).astype(np.float32)
        noise_progress = []
        for _ in range(300):
            observation = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
            nxt = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
            noise_progress.append(
                module.update_and_measure_progress(observation, descriptor, nxt, zone="noisy")
            )
        early = float(np.mean(noise_progress[:50]))
        late = float(np.mean(noise_progress[-50:]))
        assert abs(late) <= abs(early) + 0.05

    def test_raw_error_keeps_paying_on_noise(self):
        """The baseline the proposal argues against must fail the way it claims."""
        module = self._module("raw_error")
        rng = np.random.default_rng(2)
        descriptor = rng.normal(size=DESCRIPTOR_DIM).astype(np.float32)
        rewards = []
        for _ in range(300):
            observation = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
            nxt = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
            rewards.append(
                module.update_and_measure_progress(observation, descriptor, nxt, zone="noisy")
            )
        assert float(np.mean(rewards[-50:])) > 0.01

    def test_every_mode_returns_a_finite_bounded_number(self):
        for mode in ("lp_h", "lp_f", "lp_window", "raw_error", "icm", "rnd", "disagreement", "none"):
            module = self._module(mode)
            rng = np.random.default_rng(3)
            for _ in range(30):
                value = module.update_and_measure_progress(
                    rng.normal(size=OBSERVATION_DIM).astype(np.float32),
                    rng.normal(size=DESCRIPTOR_DIM).astype(np.float32),
                    rng.normal(size=OBSERVATION_DIM).astype(np.float32),
                    zone="learnable",
                    action=int(rng.integers(NUM_ACTIONS)),
                )
                assert np.isfinite(value), mode
                assert abs(value) <= 2.0, (mode, value)

    def test_icm_requires_the_executed_primitive_action(self):
        module = self._module("icm")
        zeros = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        descriptor = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
        with pytest.raises(ValueError, match="primitive action"):
            module.update_and_measure_progress(zeros, descriptor, zeros)

    def test_icm_reward_is_pre_update_forward_feature_error(self):
        """Pathak equation (6), measured before learning the transition."""
        torch.manual_seed(0)
        icm = IntrinsicCuriosityModule(
            observation_dim=OBSERVATION_DIM,
            num_actions=NUM_ACTIONS,
            reward_scale=0.7,
        )
        observation = torch.randn(1, OBSERVATION_DIM)
        next_observation = torch.randn(1, OBSERVATION_DIM)
        with torch.no_grad():
            _, _, error, _ = icm.losses(observation, 2, next_observation)
            expected = 0.7 * float(error.mean().item())
        actual = icm.update_and_score(observation, 2, next_observation)
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)

    def test_icm_inverse_loss_trains_the_feature_encoder(self):
        """Without this gradient path the implementation is not Pathak ICM."""
        torch.manual_seed(1)
        icm = IntrinsicCuriosityModule(OBSERVATION_DIM, NUM_ACTIONS)
        observation = torch.randn(1, OBSERVATION_DIM)
        next_observation = torch.randn(1, OBSERVATION_DIM)
        inverse_loss, _, _, _ = icm.losses(observation, 1, next_observation)
        icm.optimizer.zero_grad()
        inverse_loss.backward()
        encoder_gradient = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in icm.encoder.parameters()
            if parameter.grad is not None
        )
        assert encoder_gradient > 0.0

    def test_icm_exposes_inverse_and_forward_diagnostics(self):
        module = self._module("icm")
        rng = np.random.default_rng(8)
        module.update_and_measure_progress(
            rng.normal(size=OBSERVATION_DIM).astype(np.float32),
            rng.normal(size=DESCRIPTOR_DIM).astype(np.float32),
            rng.normal(size=OBSERVATION_DIM).astype(np.float32),
            action=3,
        )
        diagnostics = module.diagnostics()
        assert set(diagnostics) == {
            "icm_inverse_loss",
            "icm_forward_loss",
            "icm_inverse_accuracy",
        }
        assert all(np.isfinite(value) for value in diagnostics.values())

    def test_icm_inverse_model_learns_grid_actions_above_chance(self):
        """Functional check of the self-supervised task, not just its wiring."""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        icm = IntrinsicCuriosityModule(OBSERVATION_DIM, NUM_ACTIONS)
        recent_accuracy = []
        for episode in range(20):
            env = CuriousGridWorld(seed=episode, max_steps=120)
            observation = env.reset()
            done = False
            while not done:
                action = int(rng.integers(NUM_ACTIONS))
                next_observation, _, done, _ = env.step(action)
                icm.update_and_score(
                    torch.as_tensor(observation["features"]).unsqueeze(0),
                    action,
                    torch.as_tensor(next_observation["features"]).unsqueeze(0),
                )
                recent_accuracy.append(icm.last_inverse_accuracy)
                observation = next_observation

        # Chance accuracy is 0.25. The encoder must beat that.
        assert float(np.mean(recent_accuracy[-300:])) > 0.45

    def test_progress_fades_once_the_stream_is_mastered(self):
        """Learning progress must fade once the stream is mastered."""
        module = self._module("lp_h")
        rng = np.random.default_rng(0)
        observation = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
        descriptor = rng.normal(size=DESCRIPTOR_DIM).astype(np.float32)
        nxt = (observation + 0.1).astype(np.float32)
        progress = [
            module.update_and_measure_progress(observation, descriptor, nxt, zone="learnable")
            for _ in range(600)
        ]
        early = float(np.mean(np.abs(progress[:100])))
        late = float(np.mean(np.abs(progress[-100:])))
        assert late < 0.5 * early, (early, late)

    def test_curiosity_cannot_outweigh_the_task_reward(self):
        """
        beta_0 bounds the per-step intrinsic reward. Collecting a mineral pays
        1.0 once, while curiosity pays every step, so the ceiling has to leave
        the extrinsic signal visible over an episode.
        """
        from mdrl.decision.preference import CuriosityWeightProfile

        assert CuriosityWeightProfile().beta_0 <= 0.25

    def test_none_mode_is_silent(self):
        module = self._module("none")
        rng = np.random.default_rng(4)
        for _ in range(10):
            assert module.update_and_measure_progress(
                rng.normal(size=OBSERVATION_DIM).astype(np.float32),
                rng.normal(size=DESCRIPTOR_DIM).astype(np.float32),
                rng.normal(size=OBSERVATION_DIM).astype(np.float32),
                zone="learnable",
            ) == 0.0

class TestReplay:
    def _transition(self, rng, weights=None) -> Transition:
        n_next = 5
        return Transition(
            context=rng.normal(size=CONTEXT_DIM).astype(np.float32),
            descriptor=rng.normal(size=DESCRIPTOR_DIM).astype(np.float32),
            goal_vector=rng.uniform(size=NUM_GOALS),
            modulator_vector=rng.uniform(size=NUM_MODULATORS),
            weights=weights if weights is not None else rng.uniform(size=NUM_OUTCOMES),
            beta=float(rng.uniform()),
            outcome=rng.normal(size=NUM_OUTCOMES).astype(np.float32),
            lp_reward=float(rng.normal()),
            safety_cost=float(rng.uniform()),
            next_context=rng.normal(size=CONTEXT_DIM).astype(np.float32),
            next_descriptors=rng.normal(size=(n_next, DESCRIPTOR_DIM)).astype(np.float32),
            next_weights=rng.uniform(size=NUM_OUTCOMES),
            next_beta=float(rng.uniform()),
            next_safety_costs=rng.uniform(size=n_next).astype(np.float32),
            done=False,
            duration=int(rng.integers(1, 10)),
            certified=True,
            projection_magnitude=0.0,
        )

    def test_ragged_candidate_sets_are_padded_and_masked(self):
        """
        Candidate sets vary in size step to step, so the batch is padded to the
        widest row it drew and the padding must be masked out of the max.
        """
        rng = np.random.default_rng(0)
        buffer = MotiveStratifiedReplay(capacity=100, rng=rng)
        sizes = (2, 7, 4, 9, 3, 5)
        for size in sizes:
            transition = self._transition(rng)
            transition.next_descriptors = rng.normal(size=(size, DESCRIPTOR_DIM)).astype(np.float32)
            transition.next_safety_costs = rng.uniform(size=size).astype(np.float32)
            buffer.push(transition)
        batch = buffer.sample(64)
        assert batch.next_descriptors.shape[1] == max(sizes)
        assert batch.next_mask.shape == batch.next_descriptors.shape[:2]
        assert batch.next_mask.any(dim=1).all(), "every row must have a live candidate"
        assert set(batch.next_mask.sum(dim=1).tolist()) <= set(sizes)
        padded = ~batch.next_mask
        assert torch.allclose(
            batch.next_descriptors[padded], torch.zeros_like(batch.next_descriptors[padded])
        )

    def test_capacity_is_respected(self):
        rng = np.random.default_rng(0)
        buffer = MotiveStratifiedReplay(capacity=20, rng=rng)
        for _ in range(100):
            buffer.push(self._transition(rng))
        assert len(buffer) == 20

    def test_relabeling_changes_preferences_but_not_outcomes(self):
        """
        Preference relabeling is only sound because outcome components are
        preference-independent. This checks the stored outcomes are untouched.
        """
        rng = np.random.default_rng(0)
        buffer = MotiveStratifiedReplay(capacity=200, rng=rng)
        for _ in range(64):
            buffer.push(self._transition(rng))
        before = [record.outcome.copy() for record in buffer.records()]
        buffer.sample(32, relabel_fraction=1.0)
        after = [record.outcome for record in buffer.records()]
        for a, b in zip(before, after):
            assert np.allclose(a, b)

    def test_stratified_sampling_spreads_across_motives(self):
        rng = np.random.default_rng(0)
        buffer = MotiveStratifiedReplay(capacity=500, rng=rng)
        for index in range(200):
            weights = np.zeros(NUM_OUTCOMES)
            weights[index % NUM_OUTCOMES] = 1.0
            buffer.push(self._transition(rng, weights=weights))
        batch = buffer.sample(64, stratified_fraction=1.0)
        dominant = batch.weights.argmax(dim=1).unique()
        assert dominant.numel() >= 2, "stratified sampling must not collapse to one stratum"

class TestStability:
    def test_stabilizer_keeps_the_state_in_the_safe_region(self):
        agent = tiny_agent("P1")
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        for _ in range(25):
            observation, _, done = agent.step(env, observation, learn=False)
            assert is_in_safe_region(agent.state), agent.state.G
            if done:
                break

    def test_a_violent_proposal_is_projected_back(self):
        agent = tiny_agent("P1")
        agent.reset_episode()
        wild = np.full(NUM_GOALS, 5.0)
        next_state, diagnostics = agent.stabilizer.update(agent.state, agent.state, wild)
        assert is_in_safe_region(next_state)
        assert diagnostics.projection_magnitude >= 0.0
        assert np.abs(diagnostics.accepted_delta_g).max() <= np.abs(wild).max()

    def test_projection_magnitude_is_zero_on_a_benign_proposal(self):
        """
        Guards a subtle failure: if the projector always nudges something, the
        projection diagnostic is a constant and reveals nothing.
        """
        agent = tiny_agent("P1")
        agent.reset_episode()
        _, diagnostics = agent.stabilizer.update(
            agent.state, agent.state, np.zeros(NUM_GOALS)
        )
        assert diagnostics.projection_magnitude == pytest.approx(0.0, abs=1e-9)

    def test_disabling_the_stabilizer_actually_disables_it(self):
        agent = tiny_agent("A8-no-stabilizer")
        agent.reset_episode()
        wild = np.full(NUM_GOALS, 0.9)
        before = agent.state.copy()
        after, _ = agent.stabilizer.update(before, before, wild)
        damped, _ = tiny_agent("P1").stabilizer.update(before, before, wild)
        assert float(np.linalg.norm(after.G - before.G)) > float(
            np.linalg.norm(damped.G - before.G)
        )

    def test_projection_is_charged_back_as_a_safety_cost(self):
        """Section 8.4: the stabilizer must not be a free shield."""
        from mdrl.config import PROJECTION_PENALTY

        assert PROJECTION_PENALTY > 0.0

class TestConditions:
    def test_names_are_unique(self):
        names = [spec.name for spec in ALL_CONDITIONS]
        assert len(names) == len(set(names))

    def test_untrained_goal_residual_condition_is_disabled(self):
        assert "A7-residual-goal" not in BY_NAME

    def test_every_ablation_differs_from_p1_in_exactly_one_flag(self):
        """
        The single-flag property is what makes an ablation interpretable. If two
        flags move at once, the contrast tables are not attributable.
        """
        p1 = BY_NAME["P1"]
        exempt = {"name", "label", "notes"}
        # A fixed output head cannot index skills, so this ablation drops them too.
        allowed_pairs = {("candidates", "decision")}
        for spec in ALL_CONDITIONS:
            if not spec.name.startswith("A"):
                continue
            differing = {
                field
                for field in p1.__dataclass_fields__
                if field not in exempt and getattr(spec, field) != getattr(p1, field)
            }
            assert differing, f"{spec.name} is identical to P1"
            assert len(differing) == 1 or tuple(sorted(differing)) in allowed_pairs, (
                f"{spec.name} differs from P1 in {sorted(differing)}"
            )

    def test_groups_resolve_to_known_conditions(self):
        for group in GROUPS:
            assert resolve([group])

    def test_resolve_rejects_unknown_names(self):
        with pytest.raises(KeyError):
            resolve(["not-a-condition"])

    def test_resolve_deduplicates(self):
        specs = resolve(["P1", "P1", "core"])
        assert len({spec.name for spec in specs}) == len(specs)

    def test_baselines_do_not_secretly_use_metamo(self):
        b0 = BY_NAME["B0"]
        assert not b0.uses_metamo
        assert not b0.uses_curiosity
        assert not b0.motive_in_context
        assert not b0.stabilizer

    def test_curiosity_group_covers_every_estimator(self):
        modes = {BY_NAME[name].curiosity for name in GROUPS["curiosity"]}
        assert modes >= {"lp_h", "lp_f", "lp_window", "raw_error", "icm", "rnd", "disagreement", "none"}

class TestAgent:
    @pytest.mark.parametrize("name", [spec.name for spec in CORE_CONDITIONS])
    def test_every_core_condition_runs_an_episode(self, name):
        agent = MetaMoDRLAgent(BY_NAME[name], TINY, seed=0)
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        done = False
        while not done:
            observation, record, done = agent.step(env, observation, learn=True)
            assert np.isfinite(record.outcome).all()
            assert record.duration >= 1
        assert agent.episode_records

    @pytest.mark.parametrize(
        "name", ["A1-no-motive-input", "A3-scalar-value", "A4-fixed-output", "A6-icm", "A6-rnd", "A7-no-goal-update"]
    )
    def test_representative_ablations_run(self, name):
        agent = MetaMoDRLAgent(BY_NAME[name], TINY, seed=0)
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        done = False
        while not done:
            observation, _, done = agent.step(env, observation, learn=True)
        assert agent.episode_records

    def test_context_vector_matches_the_declared_dimension(self):
        agent = tiny_agent("P1")
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        stimulus = agent.build_stimulus(observation)
        appraised = agent.appraise(agent.state, observation, stimulus)
        context = agent.build_context(observation, appraised, stimulus)
        assert context.context_vector(True).shape == (CONTEXT_DIM,)
        assert context.context_vector(False).shape == (CONTEXT_DIM,)

    def test_ablating_motive_input_zeroes_the_motive_block_only(self):
        """
        A1 must remove information, not capacity: the vector keeps its length so
        the network is identical in size.
        """
        agent = tiny_agent("P1")
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        stimulus = agent.build_stimulus(observation)
        appraised = agent.appraise(agent.state, observation, stimulus)
        context = agent.build_context(observation, appraised, stimulus)
        with_motive = context.context_vector(True)
        without = context.context_vector(False)
        assert with_motive.shape == without.shape
        assert not np.allclose(with_motive, without)

    def test_fixed_output_conditions_never_see_skills(self):
        agent = tiny_agent("A4-fixed-output")
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        candidates = agent.candidates_for(observation, agent.state)
        assert all(c.kind is CandidateKind.PRIMITIVE for c in candidates)

    def test_learning_actually_changes_the_network(self):
        agent = tiny_agent("P1")
        before = [p.detach().clone() for p in agent.network.parameters()]
        env = fresh_env(max_steps=120)
        for _ in range(6):
            observation = env.reset()
            agent.reset_episode()
            done = False
            while not done:
                observation, _, done = agent.step(env, observation, learn=True)
            env = fresh_env(max_steps=120, seed=int(agent.rng.integers(10_000)))
        assert agent.gradient_steps > 0
        after = list(agent.network.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(before, after))

    def test_frozen_preference_ablation_holds_weights_constant(self):
        agent = tiny_agent("A2-frozen-weights")
        base = sample_training_motive(np.random.default_rng(0))
        shifted = SAFETY_PROFILE.apply(base)
        assert np.allclose(agent.preference.weights(base), agent.preference.weights(shifted))

    def test_dynamic_preference_responds_to_motive(self):
        agent = tiny_agent("P1")
        base = sample_training_motive(np.random.default_rng(0))
        safety = SAFETY_PROFILE.apply(base)
        curiosity = CURIOSITY_PROFILE.apply(base)
        assert not np.allclose(
            agent.preference.weights(safety), agent.preference.weights(curiosity)
        )

    def test_nominal_preference_is_not_dominated_by_the_penalty_component(self):
        """
        Safety is a pure penalty: it is zero at best. If the nominal motive puts
        most of the preference mass on it, the optimal policy under the default
        motive is to stop moving, and every condition looks equally inert.
        """
        agent = tiny_agent("P1")
        weights = agent.preference.weights(agent.initial_state)
        assert weights.argmax() == 0, "the nominal motive should lean toward the task"
        assert weights[OUT_SAFETY] < 0.4
        assert weights.min() > 0.05, "no component should be effectively switched off"

    def test_each_profile_promotes_its_own_objective(self):
        from mdrl.bench.protocol import RESOURCE_PROFILE, TASK_PROFILE
        from mdrl.config import OUT_INFO, OUT_RESOURCE, OUT_TASK

        agent = tiny_agent("P1")
        base = agent.initial_state
        expectations = [
            (SAFETY_PROFILE, OUT_SAFETY),
            (CURIOSITY_PROFILE, OUT_INFO),
            (RESOURCE_PROFILE, OUT_RESOURCE),
            (TASK_PROFILE, OUT_TASK),
        ]
        for profile, component in expectations:
            weights = agent.preference.weights(profile.apply(base))
            assert int(weights.argmax()) == component, (profile.name, np.round(weights, 3))
            # Dominant, not one-hot, so the tradeoff remains.
            assert weights.max() < 0.85, (profile.name, np.round(weights, 3))

    def test_safety_motive_raises_the_safety_weight(self):
        """The preference map must have the sign the interpretation assumes."""
        agent = tiny_agent("P1")
        base = sample_training_motive(np.random.default_rng(0))
        safety = agent.preference.weights(SAFETY_PROFILE.apply(base))
        curious = agent.preference.weights(CURIOSITY_PROFILE.apply(base))
        assert safety[OUT_SAFETY] > curious[OUT_SAFETY]

    def test_same_seed_same_run(self):
        def rollout():
            agent = tiny_agent("P1", seed=11)
            env = fresh_env(seed=3)
            observation = env.reset()
            agent.reset_episode()
            ids = []
            done = False
            while not done:
                observation, record, done = agent.step(env, observation, learn=True)
                ids.append(record.candidate_id)
            return ids

        assert rollout() == rollout()

    def test_intervention_moves_the_motivational_state(self):
        agent = tiny_agent("P1")
        agent.reset_episode()
        before = agent.state.copy()
        agent.apply_intervention(SAFETY_PROFILE.as_intervention(0))
        assert not np.allclose(before.G, agent.state.G)

    def test_intervention_is_inert_without_a_motivational_layer(self):
        agent = tiny_agent("B0")
        agent.reset_episode()
        before = agent.state.copy()
        agent.apply_intervention(SAFETY_PROFILE.as_intervention(0))
        assert np.allclose(before.G, agent.state.G)

class TestMetrics:
    def test_external_return_uses_one_fixed_cross_condition_evaluator(self):
        outcomes = np.array([2.0, -1.0, 0.5, 3.0])
        metrics = EpisodeMetrics(outcome_totals=outcomes)
        expected = float(np.asarray(EXTERNAL_EVALUATION_WEIGHTS) @ outcomes)
        assert metrics.external_return() == pytest.approx(expected)
        assert metrics.as_dict()["external_return"] == pytest.approx(expected)

    def test_report_derives_external_return_from_archived_outcomes(self):
        run = {
            "regimes": {
                "in_distribution": {
                    "outcome_task": 2.0,
                    "outcome_safety": -1.0,
                    "outcome_resource": 0.5,
                    "outcome_info": 3.0,
                }
            }
        }
        expected = float(
            np.asarray(EXTERNAL_EVALUATION_WEIGHTS)
            @ np.array([2.0, -1.0, 0.5, 3.0])
        )
        assert _values([run], "in_distribution", "external_return") == pytest.approx(
            [expected]
        )

    def test_transfer_table_includes_seed_uncertainty(self):
        runs = [
            {
                "regimes": {
                    "in_distribution": {
                        "external_return": value,
                    }
                }
            }
            for value in (1.0, 2.0, 3.0)
        ]
        table = transfer_table({"condition": runs})
        assert "2.000 [" in table

    def test_contrast_derives_fixed_return_for_archived_runs(self):
        def run(seed, task):
            return {
                "seed": seed,
                "regimes": {
                    "in_distribution": {
                        "outcome_task": task,
                        "outcome_safety": 0.0,
                        "outcome_resource": 0.0,
                        "outcome_info": 0.0,
                    }
                },
            }

        table = contrast_table(
            {
                "treatment": [run(seed, 2.0) for seed in range(3)],
                "control": [run(seed, 1.0) for seed in range(3)],
            },
            "treatment",
            "control",
            "in_distribution",
            metrics=(("external_return", "Fixed external return", True),),
        )
        assert "Fixed external return" in table
        assert "0.400" in table

    def test_episode_metrics_are_finite_and_consistent(self):
        agent = tiny_agent("P1")
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        done = False
        while not done:
            observation, _, done = agent.step(env, observation, learn=False)
        metrics = collect_episode_metrics(agent, env, gamma=0.97)
        assert metrics.env_steps > 0
        assert metrics.decisions > 0
        assert metrics.decisions <= metrics.env_steps
        assert 0.0 <= metrics.lava_rate() <= 1.0
        assert 0.0 <= metrics.unsafe_rate() <= 1.0
        assert 0.0 <= metrics.skill_rate() <= 1.0
        for value in metrics.as_dict().values():
            assert isinstance(value, float)

    def test_counterfactual_sensitivity_is_zero_against_itself(self):
        """PSR against an identical motive must be 0, or the probe is noise."""
        agent = tiny_agent("P1")
        agent.set_epsilon(0.0)
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        observations = [observation]
        for _ in range(8):
            observation, _, done = agent.step(env, observation, learn=False)
            observations.append(observation)
            if done:
                break
        state = agent.state.copy()
        result = counterfactual_sensitivity(agent, observations, state, state.copy())
        assert result["policy_switch_rate"] == pytest.approx(0.0)
        assert result["motivational_consistency"] == pytest.approx(0.0, abs=1e-6)

    def test_counterfactual_probe_is_symmetric_in_switch_rate(self):
        agent = tiny_agent("P1")
        agent.set_epsilon(0.0)
        env = fresh_env()
        observation = env.reset()
        agent.reset_episode()
        observations = [observation]
        for _ in range(8):
            observation, _, done = agent.step(env, observation, learn=False)
            observations.append(observation)
            if done:
                break
        base = agent.state.copy()
        a, b = SAFETY_PROFILE.apply(base), CURIOSITY_PROFILE.apply(base)
        assert counterfactual_sensitivity(agent, observations, a, b)[
            "policy_switch_rate"
        ] == pytest.approx(
            counterfactual_sensitivity(agent, observations, b, a)["policy_switch_rate"]
        )

class TestProtocol:
    def test_regimes_have_unique_names(self):
        names = [regime.name for regime in default_regimes(120)]
        assert len(names) == len(set(names))

    def test_heldout_regime_uses_heldout_maps(self):
        regimes = {regime.name: regime for regime in default_regimes(120)}
        train = {variant.name for variant in LayoutVariant.train_variants()}
        assert not {v.name for v in regimes["heldout_maps"].layouts} & train

    def test_training_library_withholds_the_transfer_skills(self):
        trained = {skill.skill_id for skill in training_skill_library()}
        full = {skill.skill_id for skill in default_skill_library()}
        assert trained < full, "some skills must be withheld for the transfer test"

    def test_new_candidate_regime_introduces_unseen_skills(self):
        regimes = {regime.name: regime for regime in default_regimes(120)}
        assert regimes["new_candidates"].skills == "full"

    def test_extreme_profiles_stay_in_the_safe_region(self):
        base = sample_training_motive(np.random.default_rng(0))
        for profile in (SAFETY_PROFILE, CURIOSITY_PROFILE):
            state = profile.apply(base)
            assert is_in_safe_region(state)
            assert (state.G >= 0.0).all() and (state.G <= 1.0).all()

    def test_profiles_are_far_from_the_training_distribution(self):
        """Regime 4 is only out of support if the profiles actually leave it."""
        rng = np.random.default_rng(0)
        samples = np.stack([sample_training_motive(rng).G for _ in range(200)])
        base = sample_training_motive(rng)
        for profile in (SAFETY_PROFILE, CURIOSITY_PROFILE):
            shifted = profile.apply(base).G
            nearest = np.abs(samples - shifted).sum(axis=1).min()
            spread = np.abs(samples - samples.mean(axis=0)).sum(axis=1).max()
            assert nearest > 0.5 * spread, profile.name

    def test_train_and_evaluate_round_trip(self):
        agent = tiny_agent("P1")
        curve = train(agent, TINY, seed=0)
        assert len(curve) == TINY.train_episodes
        assert all(np.isfinite(point["scalar_return"]) for point in curve)
        regime = default_regimes(TINY.max_steps)[0]
        summary = evaluate_regime(agent, regime, TINY, seed=0)
        assert "scalar_return" in summary
        assert all(isinstance(value, float) for value in summary.values())

    @pytest.mark.parametrize("name", ["P1", "A6-icm", "P2-residual-icm"])
    def test_evaluation_does_not_mutate_learned_state(self, name):
        agent = tiny_agent(name)
        train(agent, TINY, seed=0)
        before = agent.gradient_steps
        state_before = agent.learning_state_hash()
        evaluate_regime(agent, default_regimes(TINY.max_steps)[0], TINY, seed=0)
        assert agent.gradient_steps == before
        assert agent.learning_state_hash() == state_before

    def test_staged_appraisal_training_freezes_every_checkpoint_artifact(self):
        source = tiny_agent("P2-residual-icm", seed=17)
        train(source, TINY, seed=17, scope="joint")
        checkpoint = source.decision_checkpoint()

        staged = tiny_agent("P2-residual-icm", seed=17)
        staged.load_decision_checkpoint(checkpoint)
        frozen_before = staged.decision_state_hash()
        residual_before = [
            parameter.detach().clone()
            for parameter in staged.appraisal.network.parameters()
        ]

        train(staged, TINY, seed=17, episodes=1, scope="appraisal")

        assert staged.decision_state_hash() == frozen_before
        assert any(
            not torch.equal(before, after)
            for before, after in zip(
                residual_before, staged.appraisal.network.parameters()
            )
        )
        assert not staged.decision_learning_enabled
        assert not staged.policy_curiosity_learning_enabled
        assert not staged.appraisal_curiosity_learning_enabled

    def test_residual_initialization_is_matched_across_signal_conditions(self):
        agents = [
            tiny_agent(name, seed=23)
            for name in ("P2-residual-no-curiosity", "P2", "P2-residual-icm")
        ]
        reference = agents[0].appraisal.network.state_dict()
        for agent in agents[1:]:
            for key, expected in reference.items():
                assert torch.equal(expected, agent.appraisal.network.state_dict()[key])

    def test_frozen_appraisal_signal_uses_an_episode_local_online_copy(self):
        agent = tiny_agent("P2", seed=29)
        agent.set_training_scope("appraisal")
        frozen_hash = agent.decision_state_hash()
        env = fresh_env(max_steps=20)
        observation = env.reset()
        agent.reset_episode()
        while True:
            observation, _, done = agent.step(env, observation, learn=True)
            if done:
                break

        active = agent.active_appraisal_curiosity()
        assert active is not agent.appraisal_curiosity
        assert active.total_steps > 0
        assert agent.appraisal_curiosity.total_steps == 0
        assert agent.decision_state_hash() == frozen_hash

    def test_appraisal_icm_diagnostics_are_reported_separately(self):
        agent = tiny_agent("P2-residual-icm", seed=31)
        agent.set_learning_mode(False)
        env = fresh_env(max_steps=20)
        observation = env.reset()
        agent.reset_episode()
        while True:
            observation, _, done = agent.step(env, observation, learn=False)
            if done:
                break
        record = collect_episode_metrics(agent, env).as_dict()
        assert "appraisal_icm_inverse_accuracy" in record
        assert "appraisal_icm_forward_loss" in record

    def test_evaluation_restores_the_base_motive(self):
        agent = tiny_agent("P1")
        base = agent.initial_state.copy()
        regime = [r for r in default_regimes(TINY.max_steps) if r.name == "far_motive_safety"][0]
        evaluate_regime(agent, regime, TINY, seed=0)
        assert np.allclose(agent.initial_state.G, base.G)

class TestStats:
    def test_bootstrap_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=3.0, scale=1.0, size=40)
        summary = bootstrap_summary(values, rng=rng)
        assert summary.ci_low <= summary.mean <= summary.ci_high
        assert summary.n == 40

    def test_bootstrap_handles_degenerate_input(self):
        assert bootstrap_summary([]).n == 0
        assert np.isnan(bootstrap_summary([]).mean)
        assert bootstrap_summary([2.0]).mean == 2.0

    def test_nans_are_dropped_not_propagated(self):
        summary = bootstrap_summary([1.0, float("nan"), 3.0])
        assert summary.n == 2
        assert summary.mean == pytest.approx(2.0)

    def test_effect_sizes_have_the_right_sign_and_scale(self):
        rng = np.random.default_rng(0)
        high = rng.normal(5.0, 1.0, size=30)
        low = rng.normal(0.0, 1.0, size=30)
        assert hedges_g(high, low) > 2.0
        assert cliffs_delta(high, low) == pytest.approx(1.0, abs=0.05)
        assert hedges_g(low, high) < 0.0
        assert cliffs_delta(low, high) == pytest.approx(-1.0, abs=0.05)

    def test_identical_arms_show_no_effect(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=30)
        assert hedges_g(values, values) == pytest.approx(0.0, abs=1e-9)
        assert cliffs_delta(values, values) == pytest.approx(0.0, abs=1e-9)

    def test_compare_reports_both_arms(self):
        rng = np.random.default_rng(0)
        stats = compare(rng.normal(2.0, 1.0, 20), rng.normal(0.0, 1.0, 20), rng=rng)
        assert stats["difference"] == pytest.approx(
            stats["treatment_mean"] - stats["control_mean"]
        )
        assert stats["treatment_ci_low"] <= stats["treatment_mean"] <= stats["treatment_ci_high"]

    def test_paired_compare_uses_matched_differences(self):
        control = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        treatment = control + np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        stats = paired_compare(treatment, control)
        assert stats["n"] == 5
        assert stats["difference"] == pytest.approx(1.4)
        assert stats["difference_ci_low"] > 0.0
        assert stats["hedges_dz"] > 0.0
        assert stats["rank_biserial"] == pytest.approx(1.0)

    def test_paired_compare_reports_exact_ties(self):
        values = [1.0, 2.0, 3.0]
        stats = paired_compare(values, values)
        assert stats["difference"] == 0.0
        assert stats["difference_ci_low"] == 0.0
        assert stats["difference_ci_high"] == 0.0
        assert stats["rank_biserial"] == 0.0

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
