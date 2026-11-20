#!/usr/bin/env python3
"""Reproducible dataset generator CLI for the GENESIS RL audit environment.

Generates deterministic synthetic episodes via ``audit_env.generator.generate_episode``
and persists each episode into its own subdirectory via ``save_episode``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is importable when running as `python3 scripts/generate_data.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audit_env.generator import generate_episode, save_episode


def _resolve_num_errors(index: int, fixed_num_errors: int | None) -> int:
    """Return error count for episode *index* (deterministic mix if None)."""
    if fixed_num_errors is not None:
        return fixed_num_errors
    if index % 5 == 4:
        return 0
    return 2 + (index % 4)


def generate_split(
    out_dir: Path,
    num_episodes: int,
    base_seed: int,
    prefix: str,
    num_errors: int | None,
    target_docs: int,
    target_ledger_rows: int,
    include_compound: bool,
    include_distractors: bool,
    period_months: int,
) -> list[dict[str, Any]]:
    """Generate *num_episodes* deterministic episodes into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for i in range(num_episodes):
        ep_seed = base_seed + i
        ep_id = f"{prefix}-{ep_seed:06d}"
        err_count = _resolve_num_errors(i, num_errors)

        config: dict[str, Any] = {
            "num_errors": err_count,
            "target_docs": target_docs,
            "target_ledger_rows": target_ledger_rows,
            "include_compound": include_compound,
            "include_distractors": include_distractors,
            "period_months": period_months,
            "episode_id": ep_id,
        }

        episode = generate_episode(seed=ep_seed, config=config)
        ep_dir = out_dir / ep_id
        save_episode(episode, ep_dir)

        summaries.append({
            "episode_id": episode.manifest.episode_id,
            "seed": ep_seed,
            "dir": str(ep_dir),
            "num_records": episode.manifest.num_records,
            "num_documents": episode.manifest.num_documents,
            "num_truth_issues": len(episode.truth_issues),
        })

    return summaries

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic financial-audit episodes."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory where episode subdirectories are created (default: data).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to generate (per split if --splits is used; default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; episode i uses seed + i (default: 42).",
    )
    parser.add_argument(
        "--num-errors",
        type=int,
        default=None,
        help="Fixed number of errors per episode (0-12). Default: deterministic mix of 0 and 2-5.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ep",
        help="Prefix for episode IDs and folder names when --splits is not set (default: ep).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split names (e.g. --splits train dev) to create separate subfolders.",
    )
    parser.add_argument(
        "--target-docs",
        type=int,
        default=60,
        help="Target number of documents per episode (20-200; default: 60).",
    )
    parser.add_argument(
        "--target-ledger-rows",
        type=int,
        default=400,
        help="Target number of ledger rows per episode (default: 400).",
    )
    parser.add_argument(
        "--period-months",
        type=int,
        default=6,
        help="Reporting period length in months (1-12; default: 6).",
    )
    parser.add_argument(
        "--no-compound",
        action="store_true",
        help="Disable interacting/compound error scenarios.",
    )
    parser.add_argument(
        "--no-distractors",
        action="store_true",
        help="Disable clean distractors (split payments, refunds, recurring payments).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.num_episodes < 1:
        parser.error("--num-episodes must be >= 1")
    if args.num_errors is not None and not (0 <= args.num_errors <= 12):
        parser.error("--num-errors must be between 0 and 12")

    include_compound = not args.no_compound
    include_distractors = not args.no_distractors
    all_summaries: list[dict[str, Any]] = []

    if args.splits:
        for split_idx, split_name in enumerate(args.splits):
            split_seed = args.seed + split_idx * 10000
            split_dir = args.out_dir / split_name
            summaries = generate_split(
                out_dir=split_dir,
                num_episodes=args.num_episodes,
                base_seed=split_seed,
                prefix=split_name,
                num_errors=args.num_errors,
                target_docs=args.target_docs,
                target_ledger_rows=args.target_ledger_rows,
                include_compound=include_compound,
                include_distractors=include_distractors,
                period_months=args.period_months,
            )
            all_summaries.extend(summaries)
    else:
        summaries = generate_split(
            out_dir=args.out_dir,
            num_episodes=args.num_episodes,
            base_seed=args.seed,
            prefix=args.prefix,
            num_errors=args.num_errors,
            target_docs=args.target_docs,
            target_ledger_rows=args.target_ledger_rows,
            include_compound=include_compound,
            include_distractors=include_distractors,
            period_months=args.period_months,
        )
        all_summaries.extend(summaries)

    print(f"Generated {len(all_summaries)} episode(s) in '{args.out_dir}':")
    for item in all_summaries:
        print(
            f"  - {item['episode_id']} (seed={item['seed']}, "
            f"records={item['num_records']}, docs={item['num_documents']}, "
            f"issues={item['num_truth_issues']}) -> {item['dir']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


