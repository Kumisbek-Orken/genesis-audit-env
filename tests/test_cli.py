"""End-to-end CLI tests for scripts/generate_data.py and scripts/run_eval.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GEN_SCRIPT = _REPO_ROOT / "scripts" / "generate_data.py"
_EVAL_SCRIPT = _REPO_ROOT / "scripts" / "run_eval.py"


def _run_cli(script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_generate_data_cli(tmp_path: Path) -> None:
    """scripts/generate_data.py creates episode subdirectories with open and closed files."""
    res = _run_cli(
        _GEN_SCRIPT,
        [
            "--out-dir", str(tmp_path),
            "--num-episodes", "3",
            "--seed", "10",
            "--num-errors", "3",
        ],
    )
    assert res.returncode == 0, f"stderr: {res.stderr}"

    ep_dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(ep_dirs) == 3

    for ep_dir in ep_dirs:
        assert (ep_dir / "manifest.json").is_file()
        assert (ep_dir / "policy.json").is_file()
        assert (ep_dir / "ledger.csv").is_file()
        assert (ep_dir / "invoices.jsonl").is_file()
        assert (ep_dir / "payments.jsonl").is_file()
        assert (ep_dir / "fx.csv").is_file()
        assert (ep_dir / "closed" / "truth.json").is_file()
        assert (ep_dir / "closed" / "provenance.json").is_file()


def test_generate_data_cli_deterministic(tmp_path: Path) -> None:
    """Identical CLI seed and arguments produce byte-identical episode files."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    for out_dir in (dir_a, dir_b):
        res = _run_cli(
            _GEN_SCRIPT,
            [
                "--out-dir", str(out_dir),
                "--num-episodes", "2",
                "--seed", "77",
                "--num-errors", "2",
            ],
        )
        assert res.returncode == 0, f"stderr: {res.stderr}"

    dirs_a = sorted(p.name for p in dir_a.iterdir() if p.is_dir())
    dirs_b = sorted(p.name for p in dir_b.iterdir() if p.is_dir())
    assert dirs_a == dirs_b

    for name in dirs_a:
        assert (dir_a / name / "manifest.json").read_text() == (dir_b / name / "manifest.json").read_text()
        assert (dir_a / name / "closed" / "truth.json").read_text() == (dir_b / name / "closed" / "truth.json").read_text()

def test_run_eval_cli_outputs_table_and_valid_jsonl(tmp_path: Path) -> None:
    """scripts/run_eval.py prints summary table and writes valid JSONL trajectories without truth leaks."""
    ep_dir = tmp_path / "episodes"
    traj_path = tmp_path / "trajectories.jsonl"

    gen_res = _run_cli(
        _GEN_SCRIPT,
        [
            "--out-dir", str(ep_dir),
            "--num-episodes", "3",
            "--seed", "21",
            "--num-errors", "3",
        ],
    )
    assert gen_res.returncode == 0, f"generate_data failed: {gen_res.stderr}"

    eval_res = _run_cli(
        _EVAL_SCRIPT,
        [
            "--episodes-dir", str(ep_dir),
            "--out", str(traj_path),
            "--seed", "5",
        ],
    )
    assert eval_res.returncode == 0, f"run_eval failed: {eval_res.stderr}"

    stdout = eval_res.stdout
    for expected in (
        "Policy",
        "Episodes",
        "Mean F1",
        "Precision",
        "Recall",
        "Mean Reward",
        "Mean Steps",
        "RandomPolicy",
        "WeakHeuristicPolicy",
        "StrongBaselinePolicy",
        "OraclePolicy",
    ):
        assert expected in stdout, f"Missing '{expected}' in stdout:\n{stdout}"

    lines = [ln for ln in traj_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 12

    for line in lines:
        rec = json.loads(line)
        for key in ("episode_id", "policy", "num_steps", "total_reward", "metrics", "steps"):
            assert key in rec
        assert rec["num_steps"] == len(rec["steps"])
        assert 1 <= rec["num_steps"] <= 100

        for step_rec in rec["steps"]:
            for sk in ("step", "action", "observation", "reward", "terminated", "truncated"):
                assert sk in step_rec
            obs_dump = json.dumps(step_rec["observation"])
            assert "truth_issues" not in obs_dump
            assert "acceptable_evidence_sets" not in obs_dump
            assert "provenance" not in obs_dump


def test_run_eval_cli_monotonicity(tmp_path: Path) -> None:
    """In saved trajectories, OraclePolicy achieves F1 == 1.0 and StrongBaselinePolicy beats RandomPolicy."""
    ep_dir = tmp_path / "episodes"
    traj_path = tmp_path / "trajectories.jsonl"

    gen_res = _run_cli(
        _GEN_SCRIPT,
        [
            "--out-dir", str(ep_dir),
            "--num-episodes", "4",
            "--seed", "100",
            "--num-errors", "3",
        ],
    )
    assert gen_res.returncode == 0, f"generate_data failed: {gen_res.stderr}"

    eval_res = _run_cli(
        _EVAL_SCRIPT,
        [
            "--episodes-dir", str(ep_dir),
            "--out", str(traj_path),
            "--seed", "0",
        ],
    )
    assert eval_res.returncode == 0, f"run_eval failed: {eval_res.stderr}"

    by_policy: dict[str, list[dict]] = {}
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_policy.setdefault(rec["policy"], []).append(rec)

    assert set(by_policy.keys()) == {
        "RandomPolicy",
        "WeakHeuristicPolicy",
        "StrongBaselinePolicy",
        "OraclePolicy",
    }

    for oracle_rec in by_policy["OraclePolicy"]:
        assert oracle_rec["metrics"]["f1"] == pytest.approx(1.0)

    mean_f1_strong = sum(r["metrics"]["f1"] for r in by_policy["StrongBaselinePolicy"]) / len(by_policy["StrongBaselinePolicy"])
    mean_f1_random = sum(r["metrics"]["f1"] for r in by_policy["RandomPolicy"]) / len(by_policy["RandomPolicy"])
    assert mean_f1_strong > mean_f1_random, (
        f"Expected StrongBaselinePolicy ({mean_f1_strong:.4f}) > RandomPolicy ({mean_f1_random:.4f})"
    )


