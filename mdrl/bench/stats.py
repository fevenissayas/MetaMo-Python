"""Seed summaries: mean, median, bootstrap intervals, Hedges' g, and Cliff's delta."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

@dataclass
class Summary:
    """Central tendency and spread of one metric over seeds."""

    n: int
    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": float(self.n),
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }

def bootstrap_summary(
    values: Sequence[float],
    resamples: int = 5000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Summary:
    """Percentile bootstrap over the mean."""
    clean = np.asarray(
        [value for value in values if value is not None and not np.isnan(value)], dtype=float
    )
    if clean.size == 0:
        nan = float("nan")
        return Summary(0, nan, nan, nan, nan, nan)
    if clean.size == 1:
        single = float(clean[0])
        return Summary(1, single, single, 0.0, single, single)

    rng = rng or np.random.default_rng(0)
    draws = rng.integers(0, clean.size, size=(resamples, clean.size))
    means = clean[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return Summary(
        n=int(clean.size),
        mean=float(clean.mean()),
        median=float(np.median(clean)),
        std=float(clean.std(ddof=1)),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1.0 - alpha)),
    )

def hedges_g(treatment: Sequence[float], control: Sequence[float]) -> float:
    """Standardized mean difference with the small-sample correction."""
    a = np.asarray([v for v in treatment if not np.isnan(v)], dtype=float)
    b = np.asarray([v for v in control if not np.isnan(v)], dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled_var = (
        (a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)
    ) / (a.size + b.size - 2)
    if pooled_var <= 0:
        return 0.0
    d = (a.mean() - b.mean()) / np.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * (a.size + b.size) - 9.0)
    return float(d * correction)

def cliffs_delta(treatment: Sequence[float], control: Sequence[float]) -> float:
    """
    Non-parametric effect size in [-1, 1].

    Reported alongside Hedges' g because it makes no distributional assumption,
    which matters at the seed counts these runs can afford.
    """
    a = np.asarray([v for v in treatment if not np.isnan(v)], dtype=float)
    b = np.asarray([v for v in control if not np.isnan(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = int((a[:, None] > b[None, :]).sum())
    lesser = int((a[:, None] < b[None, :]).sum())
    return float((greater - lesser) / (a.size * b.size))

def compare(
    treatment: Sequence[float], control: Sequence[float], rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    """Summaries of both arms plus the difference and its effect sizes."""
    treatment_summary = bootstrap_summary(treatment, rng=rng)
    control_summary = bootstrap_summary(control, rng=rng)
    return {
        "treatment_mean": treatment_summary.mean,
        "control_mean": control_summary.mean,
        "difference": treatment_summary.mean - control_summary.mean,
        "hedges_g": hedges_g(treatment, control),
        "cliffs_delta": cliffs_delta(treatment, control),
        "treatment_ci_low": treatment_summary.ci_low,
        "treatment_ci_high": treatment_summary.ci_high,
        "control_ci_low": control_summary.ci_low,
        "control_ci_high": control_summary.ci_high,
    }

def paired_compare(
    treatment: Sequence[float],
    control: Sequence[float],
    resamples: int = 5000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Matched-seed contrast with a paired bootstrap and paired effect sizes."""
    pairs = np.asarray(
        [
            (a, b)
            for a, b in zip(treatment, control)
            if a is not None
            and b is not None
            and not np.isnan(a)
            and not np.isnan(b)
        ],
        dtype=float,
    )
    if pairs.size == 0:
        nan = float("nan")
        return {
            "n": 0.0,
            "treatment_mean": nan,
            "control_mean": nan,
            "difference": nan,
            "difference_ci_low": nan,
            "difference_ci_high": nan,
            "hedges_dz": nan,
            "rank_biserial": nan,
        }

    differences = pairs[:, 0] - pairs[:, 1]
    n = differences.size
    mean_difference = float(differences.mean())
    if n == 1:
        ci_low = ci_high = mean_difference
        hedges_dz = float("nan")
    else:
        rng = rng or np.random.default_rng(0)
        draws = rng.integers(0, n, size=(resamples, n))
        bootstrap_means = differences[draws].mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        ci_low, ci_high = np.quantile(
            bootstrap_means, [alpha, 1.0 - alpha]
        )
        standard_deviation = differences.std(ddof=1)
        if standard_deviation <= 0.0:
            hedges_dz = 0.0
        else:
            correction = 1.0 - 3.0 / (4.0 * n - 5.0)
            hedges_dz = correction * mean_difference / standard_deviation

    nonzero = differences[~np.isclose(differences, 0.0)]
    if nonzero.size == 0:
        rank_biserial = 0.0
    else:
        order = np.argsort(np.abs(nonzero), kind="stable")
        ranks = np.empty(nonzero.size, dtype=float)
        sorted_absolute = np.abs(nonzero)[order]
        start = 0
        while start < nonzero.size:
            end = start + 1
            while (
                end < nonzero.size
                and np.isclose(sorted_absolute[end], sorted_absolute[start])
            ):
                end += 1
            average_rank = 0.5 * ((start + 1) + end)
            ranks[order[start:end]] = average_rank
            start = end
        rank_biserial = float(
            (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum())
            / ranks.sum()
        )

    return {
        "n": float(n),
        "treatment_mean": float(pairs[:, 0].mean()),
        "control_mean": float(pairs[:, 1].mean()),
        "difference": mean_difference,
        "difference_ci_low": float(ci_low),
        "difference_ci_high": float(ci_high),
        "hedges_dz": float(hedges_dz),
        "rank_biserial": rank_biserial,
    }
