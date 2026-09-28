"""
Benchmark harness: conditions, evaluation protocol, statistics, and reporting.
"""

from mdrl.bench.conditions import (
    ABLATIONS,
    ALL_CONDITIONS,
    BY_NAME,
    CORE_CONDITIONS,
    GROUPS,
    resolve,
)
from mdrl.bench.protocol import EvaluationRegime, MotiveProfile, default_regimes
from mdrl.bench.report import build_report
from mdrl.bench.runner import RunResult, load_results, run_condition_seed, run_matrix
from mdrl.bench.stats import bootstrap_summary, cliffs_delta, compare, hedges_g

__all__ = [
    "ABLATIONS",
    "ALL_CONDITIONS",
    "BY_NAME",
    "CORE_CONDITIONS",
    "EvaluationRegime",
    "GROUPS",
    "MotiveProfile",
    "RunResult",
    "bootstrap_summary",
    "build_report",
    "cliffs_delta",
    "compare",
    "default_regimes",
    "hedges_g",
    "load_results",
    "resolve",
    "run_condition_seed",
    "run_matrix",
]
