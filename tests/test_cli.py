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
    """Public trajectories include only non-Oracle policies and contain no reference fields."""
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
    assert len(lines) == 9
    public_policies = set()

    for line in lines:
        rec = json.loads(line)
        public_policies.add(rec["policy"])
        assert rec["policy"] != "OraclePolicy"
        for key in ("episode_id", "policy", "num_steps", "total_reward", "metrics", "steps"):
            assert key in rec
        assert rec["num_steps"] == len(rec["steps"])
        assert 1 <= rec["num_steps"] <= 100
        serialized = json.dumps(rec)
        for forbidden in (
            '"truth_issues"',
            '"expected_amount_minor"',
            '"acceptable_evidence_sets"',
            '"provenance"',
        ):
            assert forbidden not in serialized

        for step_rec in rec["steps"]:
            for sk in ("step", "action", "observation", "reward", "terminated", "truncated"):
                assert sk in step_rec
            obs_dump = json.dumps(step_rec["observation"])
            assert "truth_issues" not in obs_dump
            assert "acceptable_evidence_sets" not in obs_dump
            assert "provenance" not in obs_dump

    assert public_policies == {
        "RandomPolicy",
        "WeakHeuristicPolicy",
        "StrongBaselinePolicy",
    }

    same_path_res = _run_cli(
        _EVAL_SCRIPT,
        [
            "--episodes-dir", str(ep_dir),
            "--out", str(traj_path),
            "--oracle-out", str(traj_path),
        ],
    )
    assert same_path_res.returncode != 0
    assert "--oracle-out must be a different path from --out" in same_path_res.stderr


def test_run_eval_cli_monotonicity(tmp_path: Path) -> None:
    """Oracle metrics are available separately while public scores remain monotonic."""
    ep_dir = tmp_path / "episodes"
    traj_path = tmp_path / "trajectories.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"

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
            "--oracle-out", str(oracle_path),
            "--seed", "0",
        ],
    )
    assert eval_res.returncode == 0, f"run_eval failed: {eval_res.stderr}"

    public_records = [
        json.loads(line)
        for line in traj_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {rec["policy"] for rec in public_records} == {
        "RandomPolicy",
        "WeakHeuristicPolicy",
        "StrongBaselinePolicy",
    }
    oracle_records = [
        json.loads(line)
        for line in oracle_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(oracle_records) == 4
    assert all(rec["policy"] == "OraclePolicy" for rec in oracle_records)
    assert all(rec["metrics"]["f1"] == pytest.approx(1.0) for rec in oracle_records)

    mean_f1_by_policy = {}
    for line in eval_res.stdout.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) >= 3 and cells[0] in {
            "RandomPolicy",
            "StrongBaselinePolicy",
            "OraclePolicy",
        }:
            mean_f1_by_policy[cells[0]] = float(cells[2])
    assert mean_f1_by_policy["OraclePolicy"] == pytest.approx(1.0)
    assert mean_f1_by_policy["OraclePolicy"] > mean_f1_by_policy["StrongBaselinePolicy"]
    assert mean_f1_by_policy["StrongBaselinePolicy"] > mean_f1_by_policy["RandomPolicy"]


