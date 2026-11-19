"""
Matcher — evaluates agent submissions against ground-truth issues.

Implements section 05 of SPEC.md:
  - Strict 1-to-1 bipartite matching between findings and truth issues.
  - A finding is TP only when ALL five conditions hold:
        1. type matches,
        2. canonical frozenset(affected_ids) matches,
        3. currency matches,
        4. |claimed_amount_minor − expected_amount_minor| ≤ policy.amount_tolerance_minor,
        5. set(evidence_ids) is a superset of at least one acceptable_evidence_set.
  - Deterministic greedy matching: iterate truth_issues in order; for each,
    pick the first (by submission order) unmatched finding that satisfies all
    five conditions.  This rule is fixed and published per SPEC §05.
  - Unmatched and duplicate findings → FP; unmatched truth issues → FN.
  - Metrics:  P = TP/(TP+FP),  R = TP/(TP+FN),  F1 = 2TP/(2TP+FP+FN).
    Special cases: denominator=0 → metric=0, except when BOTH truth and
    findings are empty → F1 = 1.0.
  - Terminal reward: clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1).
  - Invalid/malformed submission → reward = −1.0 immediately.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from audit_env.schemas import (
    Finding,
    Policy,
    SchemaError,
    Submission,
    TruthIssue,
    validate_submission,
)


# ── helpers ──────────────────────────────────────────────────────────

def _finding_matches_truth(
    finding: Finding,
    truth: TruthIssue,
    tolerance: int,
) -> bool:
    """Return True when *finding* satisfies all five TP conditions vs *truth*."""
    # 1. type
    if finding.type != truth.type:
        return False
    # 2. canonical set of affected IDs
    if frozenset(finding.affected_ids) != frozenset(truth.affected_ids):
        return False
    # 3. currency
    if finding.currency != truth.currency:
        return False
    # 4. amount within tolerance
    if abs(finding.claimed_amount_minor - truth.expected_amount_minor) > tolerance:
        return False
    # 5. evidence — agent must supply a superset of at least one acceptable set
    evidence = set(finding.evidence_ids)
    if not truth.acceptable_evidence_sets:
        # If the truth has no required evidence, any evidence (or none) is fine.
        return True
    for acceptable in truth.acceptable_evidence_sets:
        if evidence.issuperset(acceptable):
            return True
    return False


def _compute_metrics(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    """Return (precision, recall, F1) with correct edge-case handling."""
    # Special case: clean books with correct silence → F1 = 1.0
    if tp == 0 and fp == 0 and fn == 0:
        return (1.0, 1.0, 1.0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    return (precision, recall, f1)


def _terminal_reward(f1: float, inspection_steps: int) -> float:
    """clip(2·F1 − 1 − 0.002·steps, −1, 1)"""
    raw = 2.0 * f1 - 1.0 - 0.002 * inspection_steps
    return max(-1.0, min(1.0, raw))


# ── public API ───────────────────────────────────────────────────────

def evaluate_submission(
    findings_payload: Any,
    truth_issues: list[TruthIssue],
    policy: Policy,
    inspection_steps: int,
    *,
    expected_episode_id: str | None = None,
) -> dict:
    """Evaluate an agent submission against ground-truth issues.

    Parameters
    ----------
    findings_payload:
        Raw dict (JSON-parsed) of the agent's ``findings.json``.
    truth_issues:
        Ordered list of ground-truth issues for the episode.
    policy:
        Audit policy (carries ``amount_tolerance_minor``).
    inspection_steps:
        Number of exploration steps used by the agent (cost term).
    expected_episode_id:
        If given, ``findings_payload["episode_id"]`` must match.

    Returns
    -------
    dict with keys:
        valid        – bool, whether the submission was structurally valid.
        reward       – float in [−1, 1].
        precision    – float.
        recall       – float.
        f1           – float.
        tp / fp / fn – int counts.
        matches      – list of (truth_issue_id, finding_index) pairs.
        error        – str or None.

    Matching rule
    -------------
    Deterministic greedy by truth-issue order: for each truth issue (in
    list order) the first unmatched finding (in submission order) that
    satisfies all five TP conditions is consumed.  Remaining unmatched
    findings become FP, remaining unmatched truth issues become FN.
    """

    # ── 1. Validate structure ────────────────────────────────────────
    try:
        submission: Submission = validate_submission(findings_payload)
    except SchemaError as exc:
        return {
            "valid": False,
            "reward": -1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(truth_issues),
            "matches": [],
            "error": f"Invalid submission: {exc}",
        }

    # ── 2. Episode-ID check (if requested) ───────────────────────────
    if (
        expected_episode_id is not None
        and submission.episode_id != expected_episode_id
    ):
        return {
            "valid": False,
            "reward": -1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(truth_issues),
            "matches": [],
            "error": (
                f"episode_id mismatch: expected {expected_episode_id!r}, "
                f"got {submission.episode_id!r}"
            ),
        }

    # ── 3. Greedy 1-to-1 matching ────────────────────────────────────
    findings = list(submission.findings)
    tolerance = policy.amount_tolerance_minor
    matched_finding_indices: set[int] = set()
    matches: list[tuple[str, int]] = []  # (truth.issue_id, finding_index)
    tp = 0

    for truth in truth_issues:
        for idx, finding in enumerate(findings):
            if idx in matched_finding_indices:
                continue
            if _finding_matches_truth(finding, truth, tolerance):
                matched_finding_indices.add(idx)
                matches.append((truth.issue_id, idx))
                tp += 1
                break  # move to next truth issue

    fp = len(findings) - tp
    fn = len(truth_issues) - tp

    # ── 4. Metrics ───────────────────────────────────────────────────
    precision, recall, f1 = _compute_metrics(tp, fp, fn)
    reward = _terminal_reward(f1, inspection_steps)

    return {
        "valid": True,
        "reward": reward,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
        "error": None,
    }
