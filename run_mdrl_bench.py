"""Run the MetaMo-DRL benchmark.

    python run_mdrl_bench.py --conditions core --seeds 5 --workers 8

Accepts condition names (P1) or groups (core, ablation, curiosity, smoke, all).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdrl.bench.conditions import GROUPS, resolve
from mdrl.bench.report import build_report
from mdrl.bench.runner import load_results, run_matrix
from mdrl.config import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["smoke"],
        help=f"condition or group names; groups are {sorted(GROUPS)}",
    )
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds per condition")
    parser.add_argument("--seed-base", type=int, default=0, help="first seed value")
    parser.add_argument("--episodes", type=int, default=None, help="training episodes per run")
    parser.add_argument(
        "--appraisal-episodes",
        type=int,
        default=None,
        help="residual-appraisal episodes after loading the frozen base checkpoint",
    )
    parser.add_argument(
        "--appraisal-joint-finetune-episodes",
        type=int,
        default=None,
        help="optional labeled joint fine-tuning after residual-only training",
    )
    parser.add_argument("--eval-episodes", type=int, default=None, help="episodes per evaluation regime")
    parser.add_argument("--max-steps", type=int, default=None, help="environment steps per episode")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    parser.add_argument("--out", default="results/mdrl", help="output directory")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from existing runs")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = os.path.abspath(args.out)

    if args.report_only:
        results = load_results(output_dir)
        if not results:
            print(f"no run files found in {output_dir}", file=sys.stderr)
            return 1
        build_report(results, output_dir)
        print(f"report written to {os.path.join(output_dir, 'report.md')}")
        return 0

    specs = resolve(args.conditions)
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))

    overrides = {}
    if args.episodes is not None:
        overrides["train_episodes"] = args.episodes
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.appraisal_episodes is not None:
        overrides["appraisal_train_episodes"] = args.appraisal_episodes
    if args.appraisal_joint_finetune_episodes is not None:
        overrides["appraisal_joint_finetune_episodes"] = (
            args.appraisal_joint_finetune_episodes
        )
    training = TrainingConfig(**overrides)

    print(f"conditions : {[spec.name for spec in specs]}")
    print(f"seeds      : {seeds}")
    print(f"training   : {training.train_episodes} episodes x {training.max_steps} steps")
    print(f"evaluation : {training.eval_episodes} episodes per regime")
    if any(spec.training_stage == "appraisal" for spec in specs):
        print(f"appraisal  : {training.appraisal_train_episodes} staged episodes")
        print(
            "joint tune : "
            f"{training.appraisal_joint_finetune_episodes} optional episodes"
        )
    print(f"output     : {output_dir}")
    print()

    results = run_matrix(
        specs,
        training,
        seeds,
        output_dir=output_dir,
        workers=args.workers,
        verbose=not args.quiet,
    )

    build_report(results, output_dir)
    print()
    print(f"report written to {os.path.join(output_dir, 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
