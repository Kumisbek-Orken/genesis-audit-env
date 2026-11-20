"""
Invariance tests — SPEC §06 (Reproducibility and Generalization, 15 pts).

Take a generated episode with errors, apply a random bijective ID rename
and row shuffle to ALL open tables, update truth_issues accordingly, and
verify that an oracle submit through AuditEnv yields identical F1/reward.
"""

from __future__ import annotations

import copy
import random

import pytest

from audit_env.env import AuditEnv
from audit_env.generator import EpisodeData, generate_episode
from audit_env.schemas import (
    FXRate,
    Invoice,
    LedgerRow,
    LineItem,
    Manifest,
    Payment,
    Policy,
    ProvenanceEntry,
    TruthIssue,
)


# =====================================================================
# Helpers
# =====================================================================

def _build_id_map(episode: EpisodeData, rng: random.Random) -> dict[str, str]:
    """Collect every ID in the episode and assign a random bijective rename."""
    all_ids: set[str] = set()
    for row in episode.ledger:
        all_ids.add(row.row_id)
        all_ids.add(row.entity_id)
        if row.invoice_id:
            all_ids.add(row.invoice_id)
        if row.payment_id:
            all_ids.add(row.payment_id)
    for inv in episode.invoices:
        all_ids.add(inv.doc_id)
        all_ids.add(inv.invoice_id)
        all_ids.add(inv.entity_id)
    for pay in episode.payments:
        all_ids.add(pay.doc_id)
        all_ids.add(pay.payment_id)

    sorted_ids = sorted(all_ids)
    shuffled = sorted_ids[:]
    rng.shuffle(shuffled)
    while len(shuffled) >= 3 and shuffled == sorted_ids:
        rng.shuffle(shuffled)

    id_map = {old: f"X-{i:05d}" for i, old in enumerate(shuffled)}
    if len(sorted_ids) >= 3:
        renamed_order = sorted(sorted_ids, key=id_map.__getitem__)
        assert renamed_order != sorted_ids, "ID renaming must change relative order"
    return id_map


def _remap(value: str | None, m: dict[str, str]) -> str | None:
    if value is None:
        return None
    return m.get(value, value)


def _permute_episode(episode: EpisodeData, rng_seed: int) -> EpisodeData:
    """Shuffle row order and bijectivly rename all IDs."""
    rng = random.Random(rng_seed)
    m = _build_id_map(episode, rng)

    # Remap ledger
    new_ledger = []
    for row in episode.ledger:
        new_ledger.append(LedgerRow(
            row_id=m[row.row_id],
            timestamp=row.timestamp,
            account=row.account,
            debit_minor=row.debit_minor,
            credit_minor=row.credit_minor,
            currency=row.currency,
            entity_id=m[row.entity_id],
            invoice_id=_remap(row.invoice_id, m),
            payment_id=_remap(row.payment_id, m),
        ))
    rng.shuffle(new_ledger)

    # Remap invoices
    new_invoices = []
    for inv in episode.invoices:
        new_invoices.append(Invoice(
            doc_id=m[inv.doc_id],
            invoice_id=m[inv.invoice_id],
            entity_id=m[inv.entity_id],
            issued_at=inv.issued_at,
            due_at=inv.due_at,
            currency=inv.currency,
            total_minor=inv.total_minor,
            line_items=inv.line_items,
        ))
    rng.shuffle(new_invoices)

    # Remap payments
    new_payments = []
    for pay in episode.payments:
        new_refs = tuple(m.get(r, r) for r in pay.invoice_refs)
        new_payments.append(Payment(
            doc_id=m[pay.doc_id],
            payment_id=m[pay.payment_id],
            timestamp=pay.timestamp,
            currency=pay.currency,
            amount_minor=pay.amount_minor,
            invoice_refs=new_refs,
        ))
    rng.shuffle(new_payments)

    # Remap truth_issues
    new_truth = []
    for issue in episode.truth_issues:
        new_affected = tuple(m.get(a, a) for a in issue.affected_ids)
        new_evidence_sets = tuple(
            tuple(m.get(e, e) for e in ev_set)
            for ev_set in issue.acceptable_evidence_sets
        )
        new_truth.append(TruthIssue(
            issue_id=issue.issue_id,  # issue_id is internal, keep it
            type=issue.type,
            affected_ids=new_affected,
            expected_amount_minor=issue.expected_amount_minor,
            currency=issue.currency,
            acceptable_evidence_sets=new_evidence_sets,
        ))

    # Remap provenance
    new_prov = []
    for p in episode.provenance:
        new_prov.append(ProvenanceEntry(
            step=p.step,
            action=p.action,
            target_ids=tuple(m.get(t, t) for t in p.target_ids),
            description=p.description,
        ))

    # FX rates unchanged (no ID fields)
    new_fx = list(episode.fx_rates)
    rng.shuffle(new_fx)

    return EpisodeData(
        manifest=episode.manifest,
        policy=episode.policy,
        ledger=new_ledger,
        invoices=new_invoices,
        payments=new_payments,
        fx_rates=new_fx,
        truth_issues=new_truth,
        provenance=new_prov,
    )


def _oracle_submit(episode: EpisodeData) -> dict:
    """Build a perfect findings payload from truth_issues."""
    findings = []
    for issue in episode.truth_issues:
        # Pick the first (shortest) acceptable evidence set
        evidence = list(issue.acceptable_evidence_sets[0]) if issue.acceptable_evidence_sets else []
        findings.append({
            "type": issue.type,
            "affected_ids": list(issue.affected_ids),
            "claimed_amount_minor": issue.expected_amount_minor,
            "currency": issue.currency,
            "evidence_ids": evidence,
            "explanation": f"Oracle finding for {issue.issue_id}.",
        })
    return {
        "episode_id": episode.manifest.episode_id,
        "findings": findings,
    }


# =====================================================================
# Tests
# =====================================================================

class TestInvariance:
    """Permuting IDs and row order must preserve oracle F1 and reward."""

    def test_oracle_on_original_gives_perfect_f1(self):
        ep = generate_episode(seed=50, config={"num_errors": 4})
        env = AuditEnv(eval_mode=False)
        env.reset(episode_data=ep)
        payload = _oracle_submit(ep)
        obs, reward, terminated, _, info = env.step(
            {"tool": "submit", "findings": payload})
        assert terminated
        assert env.last_eval_result["f1"] == 1.0

    def test_permuted_oracle_matches_original(self):
        ep_orig = generate_episode(seed=50, config={"num_errors": 4})
        ep_perm = _permute_episode(ep_orig, rng_seed=9999)

        # Oracle on original
        env1 = AuditEnv(eval_mode=False)
        env1.reset(episode_data=ep_orig)
        env1.step({"tool": "submit", "findings": _oracle_submit(ep_orig)})
        r1 = env1.last_eval_result

        # Oracle on permuted
        env2 = AuditEnv(eval_mode=False)
        env2.reset(episode_data=ep_perm)
        env2.step({"tool": "submit", "findings": _oracle_submit(ep_perm)})
        r2 = env2.last_eval_result

        assert r1["f1"] == r2["f1"], f"F1 mismatch: {r1['f1']} vs {r2['f1']}"
        assert abs(r1["reward"] - r2["reward"]) < 1e-9, (
            f"Reward mismatch: {r1['reward']} vs {r2['reward']}"
        )

    def test_permuted_oracle_multiple_seeds(self):
        """Check invariance across several episode seeds."""
        for seed in [10, 20, 30]:
            ep = generate_episode(seed=seed, config={"num_errors": 5})
            ep_p = _permute_episode(ep, rng_seed=seed * 7)

            env1 = AuditEnv()
            env1.reset(episode_data=ep)
            env1.step({"tool": "submit", "findings": _oracle_submit(ep)})

            env2 = AuditEnv()
            env2.reset(episode_data=ep_p)
            env2.step({"tool": "submit", "findings": _oracle_submit(ep_p)})

            assert env1.last_eval_result["f1"] == env2.last_eval_result["f1"], (
                f"Seed {seed}: F1 mismatch"
            )
            assert abs(env1.last_eval_result["reward"] - env2.last_eval_result["reward"]) < 1e-9

    def test_clean_episode_permuted(self):
        """Clean episode (no errors): silence gives F1=1 regardless of permutation."""
        ep = generate_episode(seed=99, config={"num_errors": 0})
        ep_p = _permute_episode(ep, rng_seed=12345)

        env = AuditEnv()
        env.reset(episode_data=ep_p)
        obs, reward, _, _, _ = env.step({
            "tool": "submit",
            "findings": {"episode_id": ep_p.manifest.episode_id, "findings": []},
        })
        assert env.last_eval_result["f1"] == 1.0
        assert reward > 0
