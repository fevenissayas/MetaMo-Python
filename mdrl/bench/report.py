"""Tables and figures from raw benchmark runs, with seed counts and bootstrap intervals."""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mdrl.bench.stats import bootstrap_summary, paired_compare
from mdrl.config import EXTERNAL_EVALUATION_WEIGHTS, OUTCOME_COMPONENTS

HEADLINE_METRICS: Tuple[Tuple[str, str, bool], ...] = (
    # (key, display name, higher_is_better)
    ("external_return", "Fixed external return", True),
    ("scalar_return", "Agent-weighted return", True),
    ("success_rate", "Minerals/100 steps", True),
    ("lava_rate", "Lava rate", False),
    ("unsafe_rate", "Unsafe rate", False),
    ("energy_efficiency", "Energy eff.", True),
    ("skill_rate", "Skill use", True),
    ("safe_route_rate", "Safe-route rate", True),
    ("noisy_dwell_rate", "Noisy dwell", False),
    ("learnable_coverage", "Learnable cov.", True),
    ("descriptor_regret", "Descriptor regret", False),
)

STABILITY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("pre_projection_violation_rate", "Pre-projection violations"),
    ("post_projection_violation_rate", "Post-projection violations"),
    ("projection_frequency", "Projection frequency"),
    ("projection_magnitude", "Mean projection"),
    ("boundary_band_rate", "Boundary-band occupancy"),
    ("contraction_ratio", "Mean contraction c"),
    ("contractivity_pass_rate", "c <= 1 rate"),
    ("lax_distributive_error", "Lax distributive error"),
    ("self_model_drift", "Self-model drift"),
    ("raw_vs_accepted_delta", "Delta G clipped by H"),
)

def _by_condition(results: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["condition"]].append(result)
    return grouped

def _block_value(block: Dict[str, Any], metric: str) -> Optional[float]:
    if metric in block:
        return float(block[metric])
    if metric == "external_return" and all(
        f"outcome_{name}" in block for name in OUTCOME_COMPONENTS
    ):
        return float(
            sum(
                weight * float(block[f"outcome_{name}"])
                for weight, name in zip(
                    EXTERNAL_EVALUATION_WEIGHTS, OUTCOME_COMPONENTS
                )
            )
        )
    return None

def _values(runs: Sequence[Dict[str, Any]], regime: str, metric: str) -> List[float]:
    out = []
    for run in runs:
        block = run.get("regimes", {}).get(regime)
        if not block:
            continue
        value = _block_value(block, metric)
        if value is not None:
            out.append(value)
    return out

def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"

def _summary_cell(values: Sequence[float], digits: int = 3) -> str:
    summary = bootstrap_summary(values)
    if summary.n == 0:
        return "--"
    return (
        f"{_fmt(summary.mean, digits)} "
        f"[{_fmt(summary.ci_low, digits)}, {_fmt(summary.ci_high, digits)}]"
    )

def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def headline_table(grouped: Dict[str, List[Dict[str, Any]]], regime: str) -> str:
    header = ["Condition", "seeds"] + [name for _, name, _ in HEADLINE_METRICS]
    rows = []
    for condition, runs in grouped.items():
        cells = [condition, str(len(runs))]
        for metric, _, _ in HEADLINE_METRICS:
            summary = bootstrap_summary(_values(runs, regime, metric))
            if summary.n == 0:
                cells.append("--")
            else:
                cells.append(_summary_cell(_values(runs, regime, metric)))
        rows.append(cells)
    return _table(header, rows)

def contrast_table(
    grouped: Dict[str, List[Dict[str, Any]]],
    treatment: str,
    control: str,
    regime: str,
    metrics: Sequence[Tuple[str, str, bool]] = HEADLINE_METRICS,
) -> Optional[str]:
    if treatment not in grouped or control not in grouped:
        return None
    treatment_by_seed = {run["seed"]: run for run in grouped[treatment]}
    control_by_seed = {run["seed"]: run for run in grouped[control]}
    matched_seeds = sorted(set(treatment_by_seed) & set(control_by_seed))
    rows = []
    for metric, name, higher_is_better in metrics:
        treatment_values = []
        control_values = []
        for seed in matched_seeds:
            treatment_block = treatment_by_seed[seed].get("regimes", {}).get(regime, {})
            control_block = control_by_seed[seed].get("regimes", {}).get(regime, {})
            treatment_value = _block_value(treatment_block, metric)
            control_value = _block_value(control_block, metric)
            if treatment_value is not None and control_value is not None:
                treatment_values.append(treatment_value)
                control_values.append(control_value)
        stats = paired_compare(
            treatment_values,
            control_values,
        )
        if np.isnan(stats["difference"]):
            continue
        if np.isclose(stats["difference"], 0.0, atol=5e-7):
            result = "tie"
        else:
            treatment_better = (
                stats["difference"] > 0
                if higher_is_better
                else stats["difference"] < 0
            )
            result = treatment if treatment_better else control
        rows.append(
            [
                name,
                _fmt(stats["treatment_mean"]),
                _fmt(stats["control_mean"]),
                _fmt(stats["difference"]),
                f"[{_fmt(stats['difference_ci_low'])}, {_fmt(stats['difference_ci_high'])}]",
                _fmt(stats["hedges_dz"], 2),
                _fmt(stats["rank_biserial"], 2),
                result,
            ]
        )
    if not rows:
        return None
    return _table(
        [
            "Metric",
            treatment,
            control,
            "paired diff",
            "95% paired CI",
            "Hedges dz",
            "rank-biserial",
            "point direction",
        ],
        rows,
    )

def transfer_table(grouped: Dict[str, List[Dict[str, Any]]], metric: str = "external_return") -> str:
    regimes = sorted(
        {
            regime
            for runs in grouped.values()
            for run in runs
            for regime in run.get("regimes", {})
        }
    )
    header = ["Condition"] + regimes
    rows = []
    for condition, runs in grouped.items():
        cells = [condition]
        for regime in regimes:
            cells.append(_summary_cell(_values(runs, regime, metric)))
        rows.append(cells)
    return _table(header, rows)

def training_stability_table(grouped: Dict[str, List[Dict[str, Any]]], window: int = 20) -> str:
    """Peak versus final training return on collection episodes, not fixed-validation scores."""
    header = [
        "Condition",
        "Highest rolling return",
        "Trajectory episode",
        "Final 50 episodes",
        "Final minus highest",
    ]
    rows = []
    for condition, runs in grouped.items():
        curves = [[point["scalar_return"] for point in run["curve"]] for run in runs]
        if not curves or not curves[0]:
            continue
        length = min(len(curve) for curve in curves)
        mean_curve = np.array([curve[:length] for curve in curves]).mean(axis=0)
        if length <= window:
            continue
        smoothed = np.convolve(mean_curve, np.ones(window) / window, mode="valid")
        peak = float(smoothed.max())
        final = float(mean_curve[-50:].mean())
        rows.append(
            [
                condition,
                _fmt(peak),
                str(int(smoothed.argmax()) + window),
                _fmt(final),
                _fmt(final - peak),
            ]
        )
    rows.sort(key=lambda row: float(row[1]), reverse=True)
    return _table(header, rows)

def motive_table(grouped: Dict[str, List[Dict[str, Any]]]) -> str:
    header = [
        "Condition",
        "PSR",
        "MC safety->curiosity",
        "MC curiosity->safety",
        "Agent-weighted zero-shot/step",
        "Agent-weighted reference/step",
        "Within-agent adaptation gap",
    ]
    rows = []
    for condition, runs in grouped.items():
        def probe(block: str, key: str) -> str:
            values = [run[block][key] for run in runs if key in run.get(block, {})]
            return _summary_cell(values)

        rows.append(
            [
                condition,
                probe("counterfactual", "policy_switch_rate"),
                probe("counterfactual", "motivational_consistency_safety_to_curiosity"),
                probe("counterfactual", "motivational_consistency_curiosity_to_safety"),
                probe("adaptation", "zero_shot_return_per_step"),
                probe("adaptation", "reference_return_per_step"),
                probe("adaptation", "adaptation_gap"),
            ]
        )
    return _table(header, rows)

def safety_table(grouped: Dict[str, List[Dict[str, Any]]], regime: str) -> str:
    header = ["Condition"] + [name for _, name in STABILITY_METRICS]
    rows = []
    for condition, runs in grouped.items():
        cells = [condition]
        for metric, _ in STABILITY_METRICS:
            cells.append(_summary_cell(_values(runs, regime, metric), digits=4))
        rows.append(cells)
    return _table(header, rows)

def curiosity_table(grouped: Dict[str, List[Dict[str, Any]]], regime: str) -> str:
    header = [
        "Condition",
        "Noisy dwell",
        "Late noisy dwell",
        "Learnable cov.",
        "Dynamic cov.",
        "Re-engagement",
        "WM err (learnable)",
        "WM err (noisy)",
        "ICM inverse acc.",
        "ICM forward loss",
        "App. LP learnable",
        "App. LP noisy",
        "App. ICM inverse acc.",
        "App. ICM forward loss",
    ]
    keys = [
        "noisy_dwell_rate",
        "noisy_dwell_late",
        "learnable_coverage",
        "dynamic_coverage",
        "dynamic_reengagement",
        "wm_error_learnable",
        "wm_error_noisy",
        "icm_inverse_accuracy",
        "icm_forward_loss",
        "appraisal_wm_progress_learnable",
        "appraisal_wm_progress_noisy",
        "appraisal_icm_inverse_accuracy",
        "appraisal_icm_forward_loss",
    ]
    rows = []
    for condition, runs in grouped.items():
        cells = [condition]
        for key in keys:
            cells.append(_summary_cell(_values(runs, regime, key)))
        rows.append(cells)
    return _table(header, rows)

def write_plots(results: Sequence[Dict[str, Any]], output_dir: str) -> List[str]:
    """Learning curves and a headline bar chart. Silently skipped without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    grouped = _by_condition(results)
    paths = []

    curve_metric = (
        "external_return"
        if results
        and all(
            all("external_return" in point for point in run.get("curve", []))
            for run in results
            if run.get("curve")
        )
        else "scalar_return"
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for condition, runs in sorted(grouped.items()):
        curves = [[point[curve_metric] for point in run["curve"]] for run in runs]
        lava = [[point["lava_rate"] for point in run["curve"]] for run in runs]
        if not curves or not curves[0]:
            continue
        length = min(len(curve) for curve in curves)
        stacked = np.array([curve[:length] for curve in curves])
        lava_stacked = np.array([curve[:length] for curve in lava])
        window = max(1, length // 20)
        smooth = np.array(
            [np.convolve(row, np.ones(window) / window, mode="valid") for row in stacked]
        )
        smooth_lava = np.array(
            [np.convolve(row, np.ones(window) / window, mode="valid") for row in lava_stacked]
        )
        x = np.arange(smooth.shape[1])
        axes[0].plot(x, smooth.mean(axis=0), label=condition)
        axes[0].fill_between(
            x,
            smooth.mean(axis=0) - smooth.std(axis=0),
            smooth.mean(axis=0) + smooth.std(axis=0),
            alpha=0.12,
        )
        axes[1].plot(x, smooth_lava.mean(axis=0), label=condition)

    axes[0].set_title("Collection trajectory (not fixed validation; mean +/- sd)")
    axes[0].set_xlabel("phase-relative episode")
    axes[0].set_ylabel(
        "fixed external return"
        if curve_metric == "external_return"
        else "agent-weighted return"
    )
    axes[1].set_title("Lava-step rate")
    axes[1].set_xlabel("phase-relative episode")
    axes[1].set_ylabel("fraction of steps in lava")
    axes[0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    path = os.path.join(output_dir, "learning_curves.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(path)

    metrics = ["external_return", "success_rate", "unsafe_rate", "skill_rate"]
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.0))
    conditions = sorted(grouped)
    for axis, metric in zip(np.atleast_1d(axes), metrics):
        means, errors = [], []
        for condition in conditions:
            summary = bootstrap_summary(_values(grouped[condition], "in_distribution", metric))
            means.append(summary.mean)
            errors.append(
                [
                    max(0.0, summary.mean - summary.ci_low),
                    max(0.0, summary.ci_high - summary.mean),
                ]
            )
        error_array = np.array(errors).T if errors else None
        axis.bar(range(len(conditions)), means, yerr=error_array, capsize=3)
        axis.set_xticks(range(len(conditions)))
        axis.set_xticklabels(conditions, rotation=75, fontsize=7)
        axis.set_title(metric)
    figure.tight_layout()
    path = os.path.join(output_dir, "headline.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(path)

    regimes = sorted(
        {regime for runs in grouped.values() for run in runs for regime in run.get("regimes", {})}
    )
    if regimes:
        matrix = np.array(
            [
                [
                    bootstrap_summary(_values(grouped[condition], regime, "external_return")).mean
                    for regime in regimes
                ]
                for condition in conditions
            ]
        )
        figure, axis = plt.subplots(figsize=(1.0 * len(regimes) + 3, 0.45 * len(conditions) + 2))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(regimes)))
        axis.set_xticklabels(regimes, rotation=75, fontsize=7)
        axis.set_yticks(range(len(conditions)))
        axis.set_yticklabels(conditions, fontsize=7)
        axis.set_title("Fixed external return by evaluation regime")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        path = os.path.join(output_dir, "transfer.png")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(path)

    return paths

RQ_CONTRASTS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "RQ1 system-level -- learned system versus handcrafted reference",
        "P1",
        "B2",
        "in_distribution",
    ),
    (
        "RQ1 system-level -- full MetaMo learner versus curious DQN",
        "P1",
        "B1",
        "in_distribution",
    ),
    (
        "RQ2 controlled -- does motive context improve the vector-value policy?",
        "P1",
        "A1-no-motive-input",
        "in_distribution",
    ),
    (
        "RQ2 -- does dynamic reweighting matter beyond visibility?",
        "P1",
        "A2-frozen-weights",
        "in_distribution",
    ),
    (
        "RQ3 -- does learning-progress curiosity beat raw error?",
        "P1",
        "A6-raw-error",
        "in_distribution",
    ),
    (
        "RQ3 -- does curiosity help at all?",
        "P1",
        "A6-no-curiosity",
        "in_distribution",
    ),
    (
        "RQ3 -- does Pathak ICM beat raw prediction error?",
        "A6-icm",
        "A6-raw-error",
        "in_distribution",
    ),
    (
        "RQ3 -- does LP-H beat Pathak ICM?",
        "P1",
        "A6-icm",
        "in_distribution",
    ),
    (
        "RQ4 system-level -- skill-capable candidate policy versus primitive fixed head on held-out maps",
        "P1",
        "A4-fixed-output",
        "heldout_maps",
    ),
    (
        "RQ4 system-level -- skill-capable candidate policy versus primitive fixed head with unseen skills",
        "P1",
        "A4-fixed-output",
        "new_candidates",
    ),
    (
        "RQ5 -- what does the stabilizer cost and buy?",
        "P1",
        "A8-no-stabilizer",
        "in_distribution",
    ),
    (
        "RQ5 -- vector values versus early scalarization",
        "P1",
        "A3-scalar-value",
        "far_motive_safety",
    ),
    (
        "RQ6 -- does LP-H improve residual appraisal over no curiosity?",
        "P2",
        "P2-residual-no-curiosity",
        "in_distribution",
    ),
    (
        "RQ6 -- LP-H versus ICM for residual appraisal",
        "P2",
        "P2-residual-icm",
        "in_distribution",
    ),
)

MIN_SEEDS = 3

def build_report(results: Sequence[Dict[str, Any]], output_dir: str, min_seeds: int = MIN_SEEDS) -> str:
    grouped = _by_condition(results)
    # Drop conditions with too few seeds; a two-seed interval is not a measurement.
    underpowered = {name: len(runs) for name, runs in grouped.items() if len(runs) < min_seeds}
    grouped = {name: runs for name, runs in grouped.items() if len(runs) >= min_seeds}
    results = [run for run in results if run["condition"] in grouped]
    seeds = sorted({run["seed"] for run in results})
    lines: List[str] = []

    lines.append("# MetaMo-DRL benchmark report")
    lines.append("")
    lines.append(
        f"{len(grouped)} conditions x {len(seeds)} seeds = {len(results)} runs. "
        f"Seeds: {seeds}. Intervals are 95% percentile bootstrap over seeds."
    )
    if underpowered:
        listed = ", ".join(f"{name} ({count} seeds)" for name, count in sorted(underpowered.items()))
        lines.append("")
        lines.append(
            f"Excluded for having fewer than {min_seeds} seeds: {listed}. "
            "Re-run those conditions before drawing any conclusion about them."
        )
    lines.append("")

    lines.append("## Condition inventory")
    lines.append("")
    inventory_rows = []
    for condition, runs in sorted(grouped.items()):
        spec = runs[0]["spec"]
        phases = runs[0].get("training_provenance", {}).get("phases", [])
        phase_label = ", ".join(
            f"{phase.get('name', 'unknown')}:{phase.get('episodes', '?')}"
            for phase in phases
        ) or "--"
        inventory_rows.append(
            [
                condition,
                runs[0]["label"],
                spec["decision"],
                spec["value_head"],
                spec["candidates"],
                spec["curiosity"],
                spec.get("appraisal_signal", "none"),
                spec.get("training_stage", "joint"),
                phase_label,
                "on" if spec["certificate_gate"] else "off",
                "on" if spec["motive_in_context"] else "off",
                "on" if spec["dynamic_weights"] else "off",
                "on" if spec["stabilizer"] else "off",
            ]
        )
    lines.append(
        _table(
            ["id", "label", "decision", "values", "candidates", "curiosity", "appraisal signal", "stage", "training phases", "cert", "motive", "dyn w", "stab"],
            inventory_rows,
        )
    )
    lines.append("")

    lines.append("## Headline results (in-distribution)")
    lines.append("")
    lines.append(
        "`Fixed external return` applies the same outcome weights "
        f"{list(EXTERNAL_EVALUATION_WEIGHTS)} to every condition and is the primary "
        "cross-condition utility metric. `Agent-weighted return` uses each agent's "
        "own dynamic motivational weights and is retained only as a secondary, "
        "within-agent quantity."
    )
    lines.append("")
    lines.append(headline_table(dict(sorted(grouped.items())), "in_distribution"))
    lines.append("")

    lines.append("## Training-trajectory diagnostics (not validation)")
    lines.append("")
    lines.append(
        "These rows summarize data-collection episodes, not evaluations of saved "
        "checkpoints. Exploration, maps, motives, and rule epochs change along each "
        "trajectory; staged P2 episodes are also phase-relative. Consequently, a "
        "highest-to-final change must not be interpreted as optimization decay or "
        "used for checkpoint selection. Final-policy conclusions come only from the "
        "evaluation tables."
    )
    lines.append("")
    lines.append(training_stability_table(dict(sorted(grouped.items()))))
    lines.append("")

    lines.append("## Research-question contrasts")
    lines.append("")
    lines.append(
        "The RQ1 and RQ4 titles explicitly identify system-level comparisons that cannot "
        "be attributed to one component. Contrasts labeled controlled change the named "
        "factor. Runs are paired by seed. The table reports a paired "
        "bootstrap interval for the mean difference, bias-corrected Hedges dz, and the "
        "paired rank-biserial effect size. Seed counts remain small, so effect estimates "
        "should not be read as definitive hypothesis tests."
    )
    lines.append("")
    for title, treatment, control, regime in RQ_CONTRASTS:
        table = contrast_table(grouped, treatment, control, regime)
        if table is None:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"Regime: `{regime}`")
        lines.append("")
        lines.append(table)
        lines.append("")

    lines.append("## Transfer across evaluation regimes (fixed external return)")
    lines.append("")
    lines.append(
        "Every cell uses the same fixed outcome weights "
        f"{list(EXTERNAL_EVALUATION_WEIGHTS)} in task/safety/resource/information order; "
        "values are mean [95% bootstrap interval] over seeds."
    )
    lines.append("")
    lines.append(transfer_table(dict(sorted(grouped.items()))))
    lines.append("")

    lines.append("## Motive sensitivity and switch adaptation")
    lines.append("")
    lines.append(
        "PSR is the fraction of probe states where the greedy choice differs between a "
        "safety-dominant and a curiosity-dominant motive. It is only meaningful next to "
        "motivational consistency, which asks whether the switch improved predicted utility "
        "under the new motive: a high PSR with a negative consistency is noise, not motivation. "
        "These probes are inherently within-agent because they ask whether an agent follows "
        "its own changed motive; they are not common-utility rankings across conditions. Cells "
        "are mean [95% bootstrap interval] over seeds. Adaptation gap is agent-weighted "
        "reference minus zero-shot return per step, so positive values indicate a switch cost."
    )
    lines.append("")
    lines.append(motive_table(dict(sorted(grouped.items()))))
    lines.append("")

    lines.append("## Curiosity quality")
    lines.append("")
    lines.append(
        "The noisy zone is the trap: its transitions are unpredictable by construction, so "
        "dwell time there measures susceptibility to the noisy-television failure. The dynamic "
        "zone switches rule mid-episode, so re-engagement after the switch measures whether "
        "curiosity can return to something it had already mastered. Cells are mean "
        "[95% bootstrap interval] over seeds."
    )
    lines.append("")
    lines.append(curiosity_table(dict(sorted(grouped.items())), "in_distribution"))
    lines.append("")

    lines.append("## Safety and stability diagnostics")
    lines.append("")
    lines.append(
        "Pre-projection violations count states the decision process proposed that leave the "
        "safe region; post-projection violations count what survived the stabilizer. Mean "
        "projection magnitude is how hard the stabilizer had to work, and is charged back to "
        "the learner as a safety cost so that a clean post-projection record cannot be bought "
        "by leaning on the projection. Cells are mean [95% bootstrap interval] over seeds."
    )
    lines.append("")
    lines.append(safety_table(dict(sorted(grouped.items())), "in_distribution"))
    lines.append("")

    plots = write_plots(results, output_dir)
    if plots:
        lines.append("## Figures")
        lines.append("")
        for path in plots:
            name = os.path.basename(path)
            lines.append(f"![{name}]({name})")
            lines.append("")

    lines.append("## Runtime")
    lines.append("")
    total = sum(run.get("wallclock_seconds", 0.0) for run in results)
    lines.append(f"Total wallclock across runs: {total / 60.0:.1f} minutes.")
    lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(report)

    summary = {
        condition: {
            regime: {
                metric: bootstrap_summary(_values(runs, regime, metric)).as_dict()
                for metric, _, _ in HEADLINE_METRICS
            }
            for regime in sorted(runs[0].get("regimes", {}))
        }
        for condition, runs in grouped.items()
    }
    with open(os.path.join(output_dir, "report_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)

    return report
