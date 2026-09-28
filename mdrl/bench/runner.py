"""Train one condition under one seed, then run the shared evaluation protocol."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.state import MotivationalState
from mdrl.agent import MetaMoDRLAgent
from mdrl.bench.protocol import (
    EXTREME_PROFILES,
    CURIOSITY_PROFILE,
    EvaluationRegime,
    MotiveProfile,
    SAFETY_PROFILE,
    default_regimes,
    full_skill_library,
    restricted_skill_library,
    sample_training_motive,
    sample_unseen_motive,
    training_interventions,
    training_skill_library,
)
from mdrl.bench.conditions import BY_NAME
from mdrl.config import (
    EXTERNAL_EVALUATION_WEIGHTS,
    OUT_SAFETY,
    ConditionSpec,
    TrainingConfig,
)
from mdrl.envs.curious_gridworld import CuriousGridWorld, LayoutVariant
from mdrl.metrics import EpisodeMetrics, aggregate, collect_episode_metrics, counterfactual_sensitivity

EVAL_EPSILON = 0.02
RULE_EPOCH_EVERY = 20

@dataclass
class RunResult:
    """Everything one (condition, seed) run produces."""

    condition: str
    label: str
    seed: int
    spec: Dict[str, Any]
    curve: List[Dict[str, float]] = field(default_factory=list)
    regimes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    counterfactual: Dict[str, float] = field(default_factory=dict)
    adaptation: Dict[str, float] = field(default_factory=dict)
    evaluation_integrity: Dict[str, Any] = field(default_factory=dict)
    training_provenance: Dict[str, Any] = field(default_factory=dict)
    wallclock_seconds: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

def _run_episode(
    agent: MetaMoDRLAgent,
    env: CuriousGridWorld,
    interventions: Sequence,
    learn: bool,
) -> None:
    """Run one episode, applying any scheduled motive interventions in order."""
    observation = env.reset()
    agent.reset_episode()
    pending = sorted(interventions, key=lambda item: item.step)
    applied = 0
    done = False

    while not done:
        observation, _, done = agent.step(env, observation, learn=learn)
        while applied < len(pending) and env.step_count >= pending[applied].step:
            agent.apply_intervention(pending[applied])
            applied += 1

def _make_env(
    seed: int,
    layout: LayoutVariant,
    max_steps: int,
    rule_epoch: int,
    interventions: Sequence,
    noisy_zone: bool = True,
) -> CuriousGridWorld:
    return CuriousGridWorld(
        seed=seed,
        layout=layout,
        max_steps=max_steps,
        rule_epoch=rule_epoch,
        dynamic_switch_step=max_steps * 2 // 3,
        interventions=interventions,
        noisy_zone_enabled=noisy_zone,
    )

def train(
    agent: MetaMoDRLAgent,
    training: TrainingConfig,
    seed: int,
    verbose: bool = False,
    episodes: Optional[int] = None,
    scope: str = "joint",
) -> List[Dict[str, float]]:
    """Train and return the per-episode learning curve."""
    agent.set_learning_mode(True)
    agent.set_training_scope(scope)
    layouts = LayoutVariant.train_variants()
    agent.adapter.skill_library = training_skill_library(agent.curiosity.progress_probe)
    curve: List[Dict[str, float]] = []

    episode_count = training.train_episodes if episodes is None else episodes
    for episode in range(episode_count):
        episode_rng = np.random.default_rng(seed * 100_003 + episode)
        layout = layouts[episode % len(layouts)]
        motive = sample_training_motive(episode_rng)
        interventions = training_interventions(episode_rng, training.max_steps)

        agent.set_initial_motive(motive)
        env = _make_env(
            seed=seed * 1_009 + episode,
            layout=layout,
            max_steps=training.max_steps,
            rule_epoch=episode // RULE_EPOCH_EVERY,
            interventions=interventions,
        )
        _run_episode(agent, env, interventions, learn=True)
        if scope == "joint":
            agent.decay_epsilon()

        metrics = collect_episode_metrics(agent, env, gamma=training.gamma)
        curve.append(
            {
                "episode": float(episode),
                "scalar_return": metrics.scalar_return,
                "external_return": metrics.external_return(),
                "minerals": float(metrics.minerals),
                "lava_rate": metrics.lava_rate(),
                "epsilon": float(agent.epsilon),
                "noisy_dwell_rate": metrics.noisy_dwell_rate(),
            }
        )
        if verbose and (episode + 1) % 20 == 0:
            recent = curve[-20:]
            print(
                f"    ep {episode + 1:>4}  return {np.mean([c['scalar_return'] for c in recent]):+7.3f}"
                f"  minerals {np.mean([c['minerals'] for c in recent]):5.2f}"
                f"  lava {np.mean([c['lava_rate'] for c in recent]):5.3f}"
                f"  eps {agent.epsilon:5.3f}",
                flush=True,
            )
    return curve

def _regime_motive(
    regime: EvaluationRegime, base: MotivationalState, rng: np.random.Generator
) -> MotivationalState:
    if regime.motive == "training":
        return sample_training_motive(rng)
    if regime.motive == "unseen":
        return sample_unseen_motive(rng)
    for profile in EXTREME_PROFILES:
        if profile.name == regime.motive:
            return profile.apply(base)
    raise KeyError(f"unknown motive setting {regime.motive!r}")

def _regime_library(agent: MetaMoDRLAgent, regime: EvaluationRegime):
    probe = agent.curiosity.progress_probe
    if regime.skills == "full":
        return full_skill_library(probe)
    if regime.skills == "restricted":
        return restricted_skill_library(probe)
    return training_skill_library(probe)

def evaluate_regime(
    agent: MetaMoDRLAgent,
    regime: EvaluationRegime,
    training: TrainingConfig,
    seed: int,
) -> Dict[str, float]:
    """Run one evaluation regime and aggregate its episodes."""
    agent.adapter.skill_library = _regime_library(agent, regime)
    agent.set_epsilon(EVAL_EPSILON)
    base = agent.initial_state.copy()
    episodes: List[EpisodeMetrics] = []

    for episode in range(training.eval_episodes):
        rng = np.random.default_rng(seed * 7_919 + episode)
        layout = regime.layouts[episode % len(regime.layouts)]
        motive = _regime_motive(regime, base, rng)
        interventions = (
            [regime.intervention_profile.as_intervention(regime.intervention_step)]
            if regime.intervention_step is not None and regime.intervention_profile is not None
            else []
        )

        agent.set_initial_motive(motive)
        env = _make_env(
            seed=900_000 + seed * 1_013 + episode,
            layout=layout,
            max_steps=training.max_steps,
            rule_epoch=episode % 3,
            interventions=interventions,
            noisy_zone=regime.noisy_zone,
        )
        _run_episode(agent, env, interventions, learn=False)
        episodes.append(collect_episode_metrics(agent, env, gamma=training.gamma))

    agent.set_initial_motive(base)
    return aggregate(episodes)

def counterfactual_probe(
    agent: MetaMoDRLAgent, training: TrainingConfig, seed: int, samples: int = 60
) -> Dict[str, float]:
    """
    Collect observations from a neutral rollout, then ask whether the same
    observation yields different choices under two motivational states.
    """
    agent.adapter.skill_library = training_skill_library(agent.curiosity.progress_probe)
    agent.set_epsilon(EVAL_EPSILON)
    base = agent.initial_state.copy()

    observations: List[Dict[str, Any]] = []
    env = _make_env(
        seed=770_000 + seed,
        layout=LayoutVariant.train_variants()[0],
        max_steps=training.max_steps,
        rule_epoch=0,
        interventions=[],
    )
    observation = env.reset()
    agent.reset_episode()
    done = False
    while not done and len(observations) < samples:
        observations.append(observation)
        observation, _, done = agent.step(env, observation, learn=False)

    state_a = SAFETY_PROFILE.apply(base)
    state_b = CURIOSITY_PROFILE.apply(base)
    forward = counterfactual_sensitivity(agent, observations, state_a, state_b)
    backward = counterfactual_sensitivity(agent, observations, state_b, state_a)

    return {
        "policy_switch_rate": forward["policy_switch_rate"],
        "motivational_consistency_safety_to_curiosity": forward["motivational_consistency"],
        "motivational_consistency_curiosity_to_safety": backward["motivational_consistency"],
        "counterfactual_samples": forward["counterfactual_samples"],
    }

def motive_switch_adaptation(
    agent: MetaMoDRLAgent,
    training: TrainingConfig,
    seed: int,
    profile: MotiveProfile = SAFETY_PROFILE,
    episodes: int = 12,
) -> Dict[str, float]:
    """Zero-shot cost of a mid-episode motive switch versus starting under the new preference."""
    agent.adapter.skill_library = training_skill_library(agent.curiosity.progress_probe)
    agent.set_epsilon(EVAL_EPSILON)
    base = agent.initial_state.copy()
    switch_step = training.max_steps // 3

    switched_scores: List[float] = []
    reference_scores: List[float] = []
    switched_safety: List[float] = []
    reference_safety: List[float] = []

    for episode in range(episodes):
        env_seed = 660_000 + seed * 977 + episode
        layout = LayoutVariant.train_variants()[episode % 3]

        # Arm 1: start neutral, switch at switch_step.
        agent.set_initial_motive(base)
        intervention = [profile.as_intervention(switch_step)]
        env = _make_env(env_seed, layout, training.max_steps, 0, intervention)
        _run_episode(agent, env, intervention, learn=False)
        post = _post_switch_scores(agent, switch_step)
        switched_scores.append(post["return_per_step"])
        switched_safety.append(post["safety_per_step"])

        # Arm 2: start already under the new preference.
        agent.set_initial_motive(profile.apply(base))
        env = _make_env(env_seed, layout, training.max_steps, 0, [])
        _run_episode(agent, env, [], learn=False)
        post = _post_switch_scores(agent, switch_step)
        reference_scores.append(post["return_per_step"])
        reference_safety.append(post["safety_per_step"])

    agent.set_initial_motive(base)
    switched = float(np.mean(switched_scores))
    reference = float(np.mean(reference_scores))
    return {
        "switch_profile": profile.name,
        "zero_shot_return_per_step": switched,
        "reference_return_per_step": reference,
        "adaptation_gap": reference - switched,
        "zero_shot_safety_per_step": float(np.mean(switched_safety)),
        "reference_safety_per_step": float(np.mean(reference_safety)),
    }

def _post_switch_scores(agent: MetaMoDRLAgent, switch_step: int) -> Dict[str, float]:
    """Per-step return and safety over the portion of the episode after the switch."""
    elapsed = 0
    returns: List[float] = []
    safety: List[float] = []
    steps = 0
    for record in agent.episode_records:
        if elapsed >= switch_step:
            returns.append(float(record.weights @ record.outcome))
            safety.append(float(record.outcome[OUT_SAFETY]))
            steps += record.duration
        elapsed += record.duration
    if steps == 0:
        return {"return_per_step": 0.0, "safety_per_step": 0.0}
    return {
        "return_per_step": float(np.sum(returns) / steps),
        "safety_per_step": float(np.sum(safety) / steps),
    }

def run_condition_seed(
    spec: ConditionSpec,
    training: TrainingConfig,
    seed: int,
    regimes: Optional[Sequence[EvaluationRegime]] = None,
    verbose: bool = False,
    checkpoint_path: Optional[str] = None,
) -> RunResult:
    """Train one condition under one seed and walk the evaluation protocol."""
    torch.set_num_threads(1)
    started = time.time()

    agent = MetaMoDRLAgent(spec, training, seed=seed)
    if verbose:
        print(f"  [{spec.name} seed {seed}] {spec.describe()}", flush=True)

    provenance: Dict[str, Any] = {
        "stage": spec.training_stage,
        "base_condition": spec.base_condition or None,
        "checkpoint_path": checkpoint_path,
        "trainable_modules": ["decision", "policy_curiosity"],
        "frozen_modules": [],
        "phases": [],
    }
    if spec.training_stage == "appraisal":
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"staged condition {spec.name} requires checkpoint {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        agent.load_decision_checkpoint(checkpoint)
        base_hash = checkpoint.get("state_hash", "")
        decision_hash_before = agent.decision_state_hash()
        expected_decision_hash = checkpoint.get("decision_hash")
        if expected_decision_hash and decision_hash_before != expected_decision_hash:
            raise RuntimeError("loaded decision checkpoint hash does not match")
        curve = train(
            agent,
            training,
            seed,
            verbose=verbose,
            episodes=training.appraisal_train_episodes,
            scope="appraisal",
        )
        decision_hash_after = agent.decision_state_hash()
        if decision_hash_after != decision_hash_before:
            raise RuntimeError(
                f"staged appraisal training mutated decision state for {spec.name}"
            )
        provenance.update(
            {
                "base_checkpoint_hash": base_hash,
                "decision_hash_before": decision_hash_before,
                "decision_hash_after": decision_hash_after,
                "trainable_modules": ["residual_appraisal"],
                "frozen_modules": [
                    "decision",
                    "target_decision",
                    "policy_curiosity",
                    "appraisal_curiosity",
                ],
                "phases": [
                    {
                        "name": "residual_only",
                        "episodes": training.appraisal_train_episodes,
                        "trainable_modules": ["residual_appraisal"],
                        "frozen_modules": [
                            "decision",
                            "target_decision",
                            "policy_curiosity",
                            "appraisal_curiosity",
                        ],
                    }
                ],
            }
        )
        if training.appraisal_joint_finetune_episodes > 0:
            joint_curve = train(
                agent,
                training,
                seed,
                verbose=verbose,
                episodes=training.appraisal_joint_finetune_episodes,
                scope="joint",
            )
            for point in joint_curve:
                point["episode"] += float(training.appraisal_train_episodes)
            curve.extend(joint_curve)
            provenance["trainable_modules"] = [
                "decision",
                "policy_curiosity",
                "appraisal_curiosity",
                "residual_appraisal",
            ]
            provenance["frozen_modules"] = ["target_decision"]
            provenance["phases"].append(
                {
                    "name": "joint_finetune",
                    "episodes": training.appraisal_joint_finetune_episodes,
                    "trainable_modules": [
                        "decision",
                        "policy_curiosity",
                        "appraisal_curiosity",
                        "residual_appraisal",
                    ],
                    "frozen_modules": ["target_decision"],
                }
            )
    else:
        curve = train(agent, training, seed, verbose=verbose, scope="joint")
        provenance["phases"] = [
            {
                "name": "joint",
                "episodes": training.train_episodes,
                "trainable_modules": ["decision", "policy_curiosity"],
                "frozen_modules": ["target_decision"],
            }
        ]
    agent.set_learning_mode(False)
    agent.set_epsilon(EVAL_EPSILON)
    evaluation_hash_before = agent.learning_state_hash()

    regimes = regimes or default_regimes(training.max_steps)
    results: Dict[str, Dict[str, float]] = {}
    for regime in regimes:
        results[regime.name] = evaluate_regime(agent, regime, training, seed)

    counterfactual = counterfactual_probe(agent, training, seed)
    adaptation = motive_switch_adaptation(agent, training, seed)
    evaluation_hash_after = agent.learning_state_hash()
    if evaluation_hash_after != evaluation_hash_before:
        raise RuntimeError(
            f"evaluation mutated learned state for {spec.name} seed {seed}"
        )

    return RunResult(
        condition=spec.name,
        label=spec.label,
        seed=seed,
        spec={
            "appraisal": spec.appraisal,
            "decision": spec.decision,
            "value_head": spec.value_head,
            "candidates": spec.candidates,
            "certificate_gate": spec.certificate_gate,
            "motive_in_context": spec.motive_in_context,
            "dynamic_weights": spec.dynamic_weights,
            "curiosity": spec.curiosity,
            "goal_update": spec.goal_update,
            "stabilizer": spec.stabilizer,
            "appraisal_signal": spec.appraisal_signal,
            "training_stage": spec.training_stage,
            "base_condition": spec.base_condition,
            "external_evaluation_weights": list(EXTERNAL_EVALUATION_WEIGHTS),
            "notes": spec.notes,
        },
        curve=curve,
        regimes=results,
        counterfactual=counterfactual,
        adaptation=adaptation,
        evaluation_integrity={
            "frozen": True,
            "state_hash_before": evaluation_hash_before,
            "state_hash_after": evaluation_hash_after,
        },
        training_provenance=provenance,
        wallclock_seconds=time.time() - started,
    )

def _worker(payload: Tuple[ConditionSpec, TrainingConfig, int, bool, Optional[str]]) -> Dict[str, Any]:
    spec, training, seed, verbose, checkpoint_path = payload
    return run_condition_seed(
        spec,
        training,
        seed,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
    ).to_json()

def _checkpoint_worker(
    payload: Tuple[ConditionSpec, TrainingConfig, int, str]
) -> Tuple[str, str, int, str, float]:
    """Train and atomically persist one seed-matched base checkpoint."""
    base_spec, training, seed, path = payload
    torch.set_num_threads(1)
    started = time.time()
    base_agent = MetaMoDRLAgent(base_spec, training, seed=seed)
    train(base_agent, training, seed, verbose=False, scope="joint")
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(base_agent.decision_checkpoint(), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return (
        base_spec.base_condition or "P1",
        base_spec.appraisal_signal,
        seed,
        path,
        time.time() - started,
    )

def _valid_checkpoint(path: str, seed: int, appraisal_signal: str) -> bool:
    """Accept only readable checkpoints matching this staged artifact."""
    if not os.path.exists(path):
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if int(checkpoint.get("seed", -1)) != int(seed):
        return False
    signal = checkpoint.get("appraisal_curiosity")
    if appraisal_signal == "none":
        return signal is None
    return isinstance(signal, dict) and signal.get("mode") == appraisal_signal

def run_matrix(
    specs: Sequence[ConditionSpec],
    training: TrainingConfig,
    seeds: Sequence[int],
    output_dir: str,
    workers: int = 1,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run every (condition, seed) pair and write one JSON file per run."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    required_bases = sorted(
        {
            (spec.base_condition, spec.appraisal_signal)
            for spec in specs
            if spec.training_stage == "appraisal"
        }
    )
    checkpoints: Dict[Tuple[str, str, int], str] = {}
    checkpoint_payloads = []
    for base_name, appraisal_signal in required_bases:
        base_spec = BY_NAME[base_name]
        if appraisal_signal != "none":
            base_spec = base_spec.variant(
                f"{base_name}-signal-{appraisal_signal}",
                f"{base_spec.label} with frozen {appraisal_signal} appraisal signal",
                appraisal_signal=appraisal_signal,
            )
        for seed in seeds:
            path = os.path.join(
                checkpoint_dir,
                f"{base_name}__appraisal-{appraisal_signal}__seed{seed}.pt",
            )
            checkpoints[(base_name, appraisal_signal, seed)] = path
            if _valid_checkpoint(path, seed, appraisal_signal):
                print(
                    f"[checkpoint reuse] {base_name}/{appraisal_signal} seed {seed}",
                    flush=True,
                )
            else:
                checkpoint_payloads.append((base_spec, training, seed, path))

    if checkpoint_payloads:
        print(
            f"Preparing {len(checkpoint_payloads)} staged checkpoints "
            f"with {min(workers, len(checkpoint_payloads))} worker(s) ...",
            flush=True,
        )
        if workers <= 1:
            completed_checkpoints = map(_checkpoint_worker, checkpoint_payloads)
            for index, result in enumerate(completed_checkpoints, start=1):
                _, signal, seed, _, seconds = result
                print(
                    f"[checkpoint {index}/{len(checkpoint_payloads)}] "
                    f"{signal} seed {seed} done in {seconds:.1f}s",
                    flush=True,
                )
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_checkpoint_worker, payload)
                    for payload in checkpoint_payloads
                ]
                for index, future in enumerate(as_completed(futures), start=1):
                    _, signal, seed, _, seconds = future.result()
                    print(
                        f"[checkpoint {index}/{len(futures)}] {signal} seed {seed} "
                        f"done in {seconds:.1f}s",
                        flush=True,
                    )

    all_payloads = [
        (
            spec,
            training,
            seed,
            verbose and workers == 1,
            checkpoints.get((spec.base_condition, spec.appraisal_signal, seed)),
        )
        for spec in specs
        for seed in seeds
    ]
    results: List[Dict[str, Any]] = []
    payloads = []
    for payload in all_payloads:
        spec, _, seed, _, _ = payload
        run_path = os.path.join(output_dir, f"{spec.name}__seed{seed}.json")
        try:
            with open(run_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("condition") == spec.name and existing.get("seed") == seed:
                results.append(existing)
                print(f"[run reuse] {spec.name} seed {seed}", flush=True)
                continue
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        payloads.append(payload)

    if not payloads:
        return results

    if workers <= 1:
        for index, payload in enumerate(payloads, start=1):
            spec, _, seed, _, _ = payload
            print(f"[{index}/{len(payloads)}] {spec.name} seed {seed} ...", flush=True)
            result = _worker(payload)
            _write(result, output_dir)
            results.append(result)
            print(f"    done in {result['wallclock_seconds']:.1f}s", flush=True)
        return results

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Write each finished run immediately so a long matrix can be inspected early.
        for index, result in enumerate(pool.map(_worker, payloads, chunksize=1), start=1):
            _write(result, output_dir)
            results.append(result)
            print(
                f"[{index}/{len(payloads)}] {result['condition']} seed {result['seed']} "
                f"done in {result['wallclock_seconds']:.1f}s",
                flush=True,
            )
    return results

def _write(result: Dict[str, Any], output_dir: str) -> None:
    path = os.path.join(output_dir, f"{result['condition']}__seed{result['seed']}.json")
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=float)
    os.replace(temporary, path)

def load_results(output_dir: str) -> List[Dict[str, Any]]:
    """Read back every run JSON in a results directory."""
    results = []
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".json") or name.startswith("report"):
            continue
        with open(os.path.join(output_dir, name), encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results
