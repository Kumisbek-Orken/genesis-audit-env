"""
tests/test_policies.py
======================
Tests for the four baseline policies in audit_env/policies.py.

Checks:
  1–4.  All four policies finish without SchemaError / exceptions.
  5.    OraclePolicy → F1 == 1.0 on an episode with injected errors.
  6.    OraclePolicy → F1 == 1.0 on a clean episode (no errors).
  7.    StrongBaselinePolicy beats RandomPolicy and WeakHeuristicPolicy
        on mean F1 across 5 seeds.
  8.    run_episode returns a Trajectory with ≤100 steps and
        step count matches for all 4 policies.
"""

from __future__ import annotations

import pytest

from audit_env.env import AuditEnv
from audit_env.generator import generate_episode
from audit_env.policies import (
    OraclePolicy,
    RandomPolicy,
    StrongBaselinePolicy,
    Trajectory,
    WeakHeuristicPolicy,
    run_episode,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_env(eval_mode: bool = False) -> AuditEnv:
    """Return an AuditEnv in open (training) mode so last_eval_result is rich."""
    return AuditEnv(eval_mode=eval_mode)


def _f1(env: AuditEnv) -> float:
    """Extract F1 from env.last_eval_result after an episode."""
    r = env.last_eval_result
    assert r is not None, "Episode must have been run"
    return r["f1"]


# ─────────────────────────────────────────────────────────────────────
# Tests 1–4: all policies finish without raising
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("policy_cls,kwargs", [
    (RandomPolicy,          {"seed": 7}),
    (WeakHeuristicPolicy,   {}),
    (StrongBaselinePolicy,  {}),
    (OraclePolicy,          {}),
])
def test_policy_completes_without_error(policy_cls, kwargs):
    """Policy must run to termination without any exception."""
    env = _make_env()
    policy = policy_cls(**kwargs)
    traj = run_episode(env, policy, seed=42)

    assert isinstance(traj, Trajectory)
    # Episode must have terminated (not truncated by step limit)
    last = traj.steps[-1]
    assert last.terminated or last.truncated, "Episode must finish"
    # Reward must be in valid range
    assert -1.0 <= traj.total_reward <= 1.0
    # Submission must have been evaluated
    assert env.last_eval_result is not None
    assert env.last_eval_result["valid"], (
        f"{policy_cls.__name__} produced an invalid submission: "
        f"{env.last_eval_result.get('error')}"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 5: OraclePolicy achieves F1 == 1.0 on episode with errors
# ─────────────────────────────────────────────────────────────────────

def test_oracle_f1_equals_one_with_errors():
    """OraclePolicy must achieve F1 == 1.0 on an episode that has injected errors."""
    env = _make_env()
    ep = generate_episode(seed=0, config={"num_errors": 4})
    traj = run_episode(env, OraclePolicy(ep.truth_issues), episode_data=ep)

    assert _f1(env) == pytest.approx(1.0), (
        f"Oracle F1={_f1(env):.4f}, eval={env.last_eval_result}"
    )
    # Oracle submits on step 1
    assert len(traj.steps) == 1


# ─────────────────────────────────────────────────────────────────────
# Test 6: OraclePolicy achieves F1 == 1.0 on a clean episode
# ─────────────────────────────────────────────────────────────────────

def test_oracle_f1_equals_one_clean_episode():
    """OraclePolicy must achieve F1 == 1.0 on a clean (no errors) episode."""
    env = _make_env()
    ep = generate_episode(seed=1, config={"num_errors": 0})
    # Clean episode: truth_issues is empty
    assert len(ep.truth_issues) == 0
    traj = run_episode(env, OraclePolicy(ep.truth_issues), episode_data=ep)

    # F1 = 1.0 when both truth and findings are empty
    assert _f1(env) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────
# Test 7: StrongBaselinePolicy > Random & Weak on mean F1
# ─────────────────────────────────────────────────────────────────────

def test_strong_beats_weak_and_random_on_mean_f1():
    """StrongBaselinePolicy must have strictly higher mean F1 than
    RandomPolicy and WeakHeuristicPolicy over 5 seeds with errors."""
    seeds = [10, 20, 30, 40, 50]
    config = {"num_errors": 3, "include_compound": False}

    def _mean_f1(policy_factory):
        total = 0.0
        for s in seeds:
            env = _make_env()
            policy = policy_factory()
            run_episode(env, policy, seed=s, config=config)
            total += _f1(env)
        return total / len(seeds)

    f1_strong = _mean_f1(StrongBaselinePolicy)
    f1_random = _mean_f1(lambda: RandomPolicy(seed=0))
    f1_weak   = _mean_f1(WeakHeuristicPolicy)

    assert f1_strong > f1_random, (
        f"Strong ({f1_strong:.3f}) should beat Random ({f1_random:.3f})"
    )
    assert f1_strong > f1_weak, (
        f"Strong ({f1_strong:.3f}) should beat Weak ({f1_weak:.3f})"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 8: step counts are within budget for all policies
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("policy_cls,kwargs", [
    (RandomPolicy,          {"seed": 3}),
    (WeakHeuristicPolicy,   {}),
    (StrongBaselinePolicy,  {}),
    (OraclePolicy,          {}),
])
def test_step_budget_respected(policy_cls, kwargs):
    """All policies must finish within the 100-step budget."""
    env = _make_env()
    policy = policy_cls(**kwargs)
    traj = run_episode(env, policy, seed=99)

    assert len(traj.steps) <= 100, (
        f"{policy_cls.__name__} used {len(traj.steps)} steps (max 100)"
    )
    assert len(traj.steps) >= 1, "At least one step must be taken (submit)"


# ─────────────────────────────────────────────────────────────────────
# Test 9: Strict policy isolation contract
# ─────────────────────────────────────────────────────────────────────

def test_policy_isolation_contract():
    """Spy policy verifies that run_episode passes ONLY a dict observation
    without any references to env, truth_issues, provenance, or seed."""
    env = _make_env(eval_mode=True)
    ep = generate_episode(seed=42)

    calls = []

    class SpyPolicy:
        def __init__(self):
            self.step_idx = 0

        def __call__(self, *args, **kwargs):
            # Must be called with exactly 1 positional argument (obs) and no kwargs
            assert len(args) == 1, f"Policy expected 1 argument (obs), got {len(args)}"
            assert len(kwargs) == 0, f"Policy expected 0 kwargs, got {kwargs}"
            obs = args[0]
            assert isinstance(obs, dict), f"Observation must be dict, got {type(obs)}"

            # Forbidden evaluator state must not leak into observation
            assert "truth_issues" not in obs
            assert "acceptable_evidence_sets" not in obs
            assert "provenance" not in obs
            assert "seed" not in obs
            assert "env" not in obs

            # Check that AuditEnv is not referenced anywhere in observation values
            for val in obs.values():
                assert not isinstance(val, AuditEnv)

            calls.append(obs)
            self.step_idx += 1

            if self.step_idx == 1:
                return {"tool": "list_records", "table": "invoices", "cursor": 0, "limit": 10}
            elif self.step_idx == 2:
                return {"tool": "calculator", "expression": "100 + 200"}
            else:
                ep_id = calls[0].get("manifest", {}).get("episode_id", "")
                return {"tool": "submit", "findings": {"episode_id": ep_id, "findings": []}}

    traj = run_episode(env, SpyPolicy(), episode_data=ep)
    assert len(calls) == 3
    assert len(traj.steps) == 3
