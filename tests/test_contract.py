"""
Contract tests for AuditEnv — SPEC §04, §06, §10.

Covers:
  - Lifecycle: reset/step, double-reset isolation, step-after-terminal error.
  - All 5 tools: list_records, get_record, get_document, calculator, submit.
  - Pagination: pages ≤50, cursor stability.
  - Invalid actions: bad tool name, missing params, non-existent IDs.
  - Step budget: truncation at 100 steps without submit.
  - Leak scan: recursive scan of ALL observation and info dicts for
    forbidden keywords (truth, provenance, seed, issue_id).
  - Generator audit: injected records are shuffled (not clustered at end).
"""

from __future__ import annotations

import json
import re

import pytest

from audit_env.env import AuditEnv, _MAX_STEPS
from audit_env.generator import generate_episode


# =====================================================================
# Helpers
# =====================================================================

_LEAK_PATTERNS = re.compile(
    r"\b(truth|provenance|seed|issue_id|ground.?truth)\b", re.IGNORECASE
)


def _scan_for_leaks(obj, path="root"):
    """Recursively scan a dict/list/str for forbidden keywords."""
    violations = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and _LEAK_PATTERNS.search(k):
                violations.append(f"{path}.key({k!r})")
            _inner = _scan_for_leaks(v, f"{path}[{k!r}]")
            violations.extend(_inner)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            violations.extend(_scan_for_leaks(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        if _LEAK_PATTERNS.search(obj):
            violations.append(f"{path}={obj!r}")
    return violations


def _make_env_with_errors(seed=42, num_errors=3):
    env = AuditEnv(eval_mode=True)
    obs, info = env.reset(seed=seed, config={"num_errors": num_errors})
    return env, obs, info


# =====================================================================
# 1. Lifecycle
# =====================================================================

class TestLifecycle:
    def test_reset_returns_observation_with_required_keys(self):
        env, obs, info = _make_env_with_errors()
        assert "task" in obs
        assert "manifest" in obs
        assert "policy_summary" in obs
        assert "tools" in obs
        assert obs["remaining_steps"] == _MAX_STEPS

    def test_step_before_reset_raises(self):
        env = AuditEnv()
        with pytest.raises(RuntimeError, match="reset"):
            env.step({"tool": "calculator", "expression": "1+1"})

    def test_step_after_terminal_raises(self):
        env, _, _ = _make_env_with_errors(num_errors=0)
        ep_id = env._episode.manifest.episode_id
        env.step({"tool": "submit", "findings": {
            "episode_id": ep_id, "findings": []}})
        with pytest.raises(RuntimeError, match="finished"):
            env.step({"tool": "calculator", "expression": "1+1"})

    def test_double_reset_isolates_state(self):
        env = AuditEnv()
        env.reset(seed=1)
        ep_id_1 = env._episode.manifest.episode_id
        env.reset(seed=2)
        ep_id_2 = env._episode.manifest.episode_id
        assert ep_id_1 != ep_id_2
        assert env._step_count == 0
        assert not env._terminated


# =====================================================================
# 2. Tools
# =====================================================================

class TestTools:
    def test_list_records_all_tables(self):
        env, _, _ = _make_env_with_errors()
        for table in ("ledger", "invoices", "payments", "fx_rates"):
            obs, r, t, tr, info = env.step({
                "tool": "list_records", "table": table})
            assert obs["type"] == "list_records"
            assert len(obs["records"]) <= 50
            assert obs["table"] == table
            assert r == 0.0
            assert not t and not tr

    def test_pagination_max_50(self):
        env, _, _ = _make_env_with_errors()
        obs, *_ = env.step({
            "tool": "list_records", "table": "ledger", "limit": 999})
        assert len(obs["records"]) <= 50

    def test_cursor_pagination(self):
        env, _, _ = _make_env_with_errors()
        obs1, *_ = env.step({
            "tool": "list_records", "table": "ledger", "limit": 10})
        if obs1["next_cursor"] is not None:
            obs2, *_ = env.step({
                "tool": "list_records", "table": "ledger",
                "cursor": obs1["next_cursor"], "limit": 10})
            # Pages should not overlap
            ids1 = {r["row_id"] for r in obs1["records"]}
            ids2 = {r["row_id"] for r in obs2["records"]}
            assert ids1.isdisjoint(ids2)

    def test_get_record_success(self):
        env, _, _ = _make_env_with_errors()
        row_id = env._episode.ledger[0].row_id
        obs, r, t, tr, _ = env.step({
            "tool": "get_record", "table": "ledger", "id": row_id})
        assert obs["type"] == "get_record"
        assert obs["record"]["row_id"] == row_id
        assert r == 0.0

    def test_get_document_success(self):
        env, _, _ = _make_env_with_errors()
        doc_id = env._episode.invoices[0].doc_id
        obs, *_ = env.step({"tool": "get_document", "doc_id": doc_id})
        assert obs["type"] == "get_document"
        assert obs["document"]["doc_id"] == doc_id

    def test_calculator_success(self):
        env, _, _ = _make_env_with_errors()
        obs, *_ = env.step({"tool": "calculator", "expression": "100 * 3 + 50"})
        from decimal import Decimal; assert Decimal(obs["result"]) == Decimal("350")

    def test_submit_returns_terminal(self):
        env, _, _ = _make_env_with_errors(num_errors=0)
        ep_id = env._episode.manifest.episode_id
        obs, reward, terminated, truncated, info = env.step({
            "tool": "submit",
            "findings": {"episode_id": ep_id, "findings": []},
        })
        assert terminated is True
        assert truncated is False
        assert obs["type"] == "submit"
        # Clean books + empty findings → F1=1
        assert reward > 0


# =====================================================================
# 3. Invalid actions
# =====================================================================

class TestInvalidActions:
    def test_unknown_tool(self):
        env, _, _ = _make_env_with_errors()
        obs, r, t, tr, _ = env.step({"tool": "hack_the_planet"})
        assert obs["type"] == "error"
        assert r == 0.0 and not t and not tr

    def test_bad_table(self):
        env, _, _ = _make_env_with_errors()
        obs, *_ = env.step({"tool": "list_records", "table": "secrets"})
        assert obs["type"] == "error"

    def test_nonexistent_id(self):
        env, _, _ = _make_env_with_errors()
        obs, *_ = env.step({
            "tool": "get_record", "table": "ledger", "id": "NOPE-9999"})
        assert obs["type"] == "error"

    def test_missing_tool_key(self):
        env, _, _ = _make_env_with_errors()
        obs, r, *_ = env.step({"action": "list_records"})
        assert obs["type"] == "error"
        assert r == 0.0


# =====================================================================
# 4. Step budget and truncation
# =====================================================================

class TestTruncation:
    def test_truncation_at_100_steps(self):
        env, _, _ = _make_env_with_errors(seed=7, num_errors=2)
        for i in range(_MAX_STEPS):
            obs, reward, terminated, truncated, info = env.step(
                {"tool": "calculator", "expression": "1+1"})
            if truncated or terminated:
                break
        assert truncated is True
        assert terminated is False
        assert obs.get("truncated") is True
        # Truncation with errors present → reward should be ≤ -1 (clipped)
        assert reward <= 0


# =====================================================================
# 5. Leak scan (eval_mode)
# =====================================================================

class TestLeakScan:
    def test_no_leaks_in_reset_observation(self):
        env, obs, info = _make_env_with_errors()
        violations = _scan_for_leaks(obs) + _scan_for_leaks(info)
        assert violations == [], f"Leak found: {violations}"

    def test_no_leaks_in_step_observations(self):
        env, _, _ = _make_env_with_errors()
        all_violations = []

        # list_records
        for table in ("ledger", "invoices", "payments", "fx_rates"):
            obs, *_ = env.step({"tool": "list_records", "table": table})
            all_violations.extend(_scan_for_leaks(obs, f"list_{table}"))

        # get_record
        obs, *_ = env.step({
            "tool": "get_record", "table": "ledger",
            "id": env._episode.ledger[0].row_id})
        all_violations.extend(_scan_for_leaks(obs, "get_record"))

        # get_document
        obs, *_ = env.step({
            "tool": "get_document",
            "doc_id": env._episode.invoices[0].doc_id})
        all_violations.extend(_scan_for_leaks(obs, "get_document"))

        # error
        obs, *_ = env.step({"tool": "get_record", "table": "ledger", "id": "BAD"})
        all_violations.extend(_scan_for_leaks(obs, "error"))

        # submit
        ep_id = env._episode.manifest.episode_id
        obs, _, _, _, info = env.step({
            "tool": "submit",
            "findings": {"episode_id": ep_id, "findings": []},
        })
        all_violations.extend(_scan_for_leaks(obs, "submit_obs"))
        all_violations.extend(_scan_for_leaks(info, "submit_info"))

        assert all_violations == [], f"Leaks found: {all_violations}"


# =====================================================================
# 6. Generator audit — injected records shuffled
# =====================================================================

class TestGeneratorShuffle:
    def test_error_records_are_not_all_at_end(self):
        """Injected invoices/ledger rows must be shuffled into the lists,
        not appended as a contiguous block at the tail."""
        ep = generate_episode(seed=42, config={"num_errors": 4})
        # Collect IDs that appear in truth_issues
        error_ids = set()
        for issue in ep.truth_issues:
            error_ids.update(issue.affected_ids)
            for ev_set in issue.acceptable_evidence_sets:
                error_ids.update(ev_set)

        # Check invoices: at least one error-related invoice should NOT
        # be in the last 25% of the list
        inv_ids = [inv.invoice_id for inv in ep.invoices]
        cutoff = max(1, len(inv_ids) * 3 // 4)
        early_error_invs = [
            iid for iid in inv_ids[:cutoff] if iid in error_ids
        ]
        assert len(early_error_invs) > 0, (
            "All error-related invoices are in the last 25% — not shuffled"
        )
