#!/usr/bin/env python3
"""Reproducible evaluation runner CLI for the GENESIS RL audit environment.

Loads all saved episodes from a directory via ``load_episode``, runs all four
baseline policies from ``audit_env.policies`` through ``run_episode``, prints
a summary metrics table to stdout, and writes full trajectories to a JSONL file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Ensure repository root is importable when running as `python3 scripts/run_eval.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audit_env.env import AuditEnv
from audit_env.generator import EpisodeData, load_episode
from audit_env.policies import (
    OraclePolicy,
    RandomPolicy,
    StrongBaselinePolicy,
    Trajectory,
    WeakHeuristicPolicy,
    run_episode,
)


def discover_episode_dirs(episodes_dir: Path) -> list[Path]:
    """Return sorted list of directories containing a ``manifest.json`` file."""
    if not episodes_dir.exists() or not episodes_dir.is_dir():
        return []
    manifests = sorted(episodes_dir.rglob("manifest.json"))
    return [m.parent for m in manifests]


def _trajectory_to_dict(
    policy_name: str,
    traj: Trajectory,
    eval_result: dict[str, Any],
    elapsed_sec: float,
    episode_dir: Path,
) -> dict[str, Any]:
    """Serialize one (episode, policy) run into a JSON-compatible dictionary."""
    return {
        "episode_id": traj.episode_id,
        "policy": policy_name,
        "episode_dir": str(episode_dir),
        "num_steps": len(traj.steps),
        "total_reward": float(traj.total_reward),
        "elapsed_sec": round(elapsed_sec, 6),
        "metrics": {
            "valid": bool(eval_result["valid"]),
            "precision": float(eval_result["precision"]),
            "recall": float(eval_result["recall"]),
            "f1": float(eval_result["f1"]),
            "reward": float(eval_result["reward"]),
            "tp": int(eval_result["tp"]),
            "fp": int(eval_result["fp"]),
            "fn": int(eval_result["fn"]),
            "steps": len(traj.steps),
        },
        "steps": [
            {
                "step": s.step,
                "action": s.action,
                "observation": s.observation,
                "reward": float(s.reward),
                "terminated": bool(s.terminated),
                "truncated": bool(s.truncated),
            }
            for s in traj.steps
        ],
    }


def _format_summary_table(rows: list[dict[str, Any]]) -> str:
    """Format an aligned summary table of per-policy evaluation metrics."""
    headers = [
        "Policy",
        "Episodes",
        "Mean F1",
        "Precision",
        "Recall",
        "Mean Reward",
        "Mean Steps",
        "Mean Time(s)",
    ]
    formatted_rows = [
        [
            r["policy"],
            str(r["episodes"]),
            f"{r['mean_f1']:.4f}",
            f"{r['mean_precision']:.4f}",
            f"{r['mean_recall']:.4f}",
            f"{r['mean_reward']:+.4f}",
            f"{r['mean_steps']:.2f}",
            f"{r['mean_time']:.4f}",
        ]
        for r in rows
    ]

    widths = [len(h) for h in headers]
    for row in formatted_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_line(cells: list[str]) -> str:
        return " | ".join(
            cell.ljust(widths[idx]) if idx == 0 else cell.rjust(widths[idx])
            for idx, cell in enumerate(cells)
        )

    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([_fmt_line(headers), sep, *(_fmt_line(r) for r in formatted_rows)])

def evaluate_episodes(
    episode_entries: list[tuple[Path, EpisodeData]],
    out_file: Path,
    seed: int = 0,
    eval_mode: bool = True,
) -> list[dict[str, Any]]:
    """Run all 4 policies over *episode_entries* and write trajectories to *out_file*."""
    policy_specs: list[tuple[str, Callable[[int, EpisodeData], Any]]] = [
        ("RandomPolicy", lambda ep_idx, _ep: RandomPolicy(seed=seed + ep_idx)),
        ("WeakHeuristicPolicy", lambda _ep_idx, _ep: WeakHeuristicPolicy()),
        ("StrongBaselinePolicy", lambda _ep_idx, _ep: StrongBaselinePolicy()),
        ("OraclePolicy", lambda _ep_idx, ep: OraclePolicy(ep.truth_issues)),
    ]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    agg: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in policy_specs}

    with open(out_file, "w", encoding="utf-8") as fh:
        for ep_idx, (ep_dir, ep_data) in enumerate(episode_entries):
            for policy_name, policy_factory in policy_specs:
                env = AuditEnv(eval_mode=eval_mode)
                policy = policy_factory(ep_idx, ep_data)

                t0 = time.perf_counter()
                traj = run_episode(env, policy, episode_data=ep_data)
                elapsed = time.perf_counter() - t0

                eval_result = env.last_eval_result
                if eval_result is None:
                    raise RuntimeError(
                        f"Missing last_eval_result for {policy_name} on {ep_dir}"
                    )

                record = _trajectory_to_dict(
                    policy_name=policy_name,
                    traj=traj,
                    eval_result=eval_result,
                    elapsed_sec=elapsed,
                    episode_dir=ep_dir,
                )
                fh.write(json.dumps(record) + "\n")

                agg[policy_name].append({
                    "f1": float(eval_result["f1"]),
                    "precision": float(eval_result["precision"]),
                    "recall": float(eval_result["recall"]),
                    "reward": float(eval_result["reward"]),
                    "steps": len(traj.steps),
                    "time": elapsed,
                })

    summary_rows: list[dict[str, Any]] = []
    for policy_name, _ in policy_specs:
        items = agg[policy_name]
        n = len(items)
        summary_rows.append({
            "policy": policy_name,
            "episodes": n,
            "mean_f1": sum(x["f1"] for x in items) / n,
            "mean_precision": sum(x["precision"] for x in items) / n,
            "mean_recall": sum(x["recall"] for x in items) / n,
            "mean_reward": sum(x["reward"] for x in items) / n,
            "mean_steps": sum(x["steps"] for x in items) / n,
            "mean_time": sum(x["time"] for x in items) / n,
        })

    return summary_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all 4 baseline policies on saved episodes and export trajectories."
    )
    parser.add_argument(
        "--episodes-dir",
        "--data-dir",
        dest="episodes_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing saved episode subdirectories (default: data).",
    )
    parser.add_argument(
        "--out",
        "--out-file",
        "--trajectories",
        dest="out_file",
        type=Path,
        default=Path("trajectories.jsonl"),
        help="Output JSONL path for recorded trajectories (default: trajectories.jsonl).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for RandomPolicy (default: 0).",
    )
    parser.add_argument(
        "--open-mode",
        action="store_true",
        help="Run AuditEnv with eval_mode=False (includes terminal metrics in final observation).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ep_dirs = discover_episode_dirs(args.episodes_dir)
    if not ep_dirs:
        print(
            f"Error: no episodes (manifest.json) found under '{args.episodes_dir}'.",
            file=sys.stderr,
        )
        return 1

    episode_entries = [(d, load_episode(d)) for d in ep_dirs]
    summary_rows = evaluate_episodes(
        episode_entries=episode_entries,
        out_file=args.out_file,
        seed=args.seed,
        eval_mode=not args.open_mode,
    )

    print(_format_summary_table(summary_rows))
    print(
        f"\nSaved {len(episode_entries) * 4} trajectories "
        f"({len(episode_entries)} episodes x 4 policies) to {args.out_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


