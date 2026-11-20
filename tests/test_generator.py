"""
Unit and property tests for audit_env.generator.

Covers:
  1. Accounting balance (sum(debit_minor) == sum(credit_minor)) on clean episodes (num_errors=0).
  2. Empty truth_issues and provenance on clean episodes.
  3. Accounting balance across multiple seeds.
  4. Full determinism (identical seeds and configs yield identical episodes).
  5. Different seeds produce distinct episodes.
  6. Distractors presence in clean episodes (split payments, refunds, recurring payments).
  7. Distractors disable flag (include_distractors=False).
  8. Error injection scaling with num_errors.
  9. Generation of all 4 canonical error types.
  10. Generation of compound error scenarios.
  11. Provenance log completeness for every injected mutation.
  12. Referential integrity: all affected_ids exist in open invoices/payments.
  13. Referential integrity: all evidence_ids exist in open docs or ledger rows.
  14. Manifest consistency: counts and ISO-8601 timestamps.
  15. FX rates presence and valid format.
  16. Lossless save_episode and load_episode roundtrip.
  17. Directory structure: open files in root, closed files in closed/ folder.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import pytest

from audit_env.generator import (
    EpisodeData,
    generate_episode,
    load_episode,
    save_episode,
)
from audit_env.schemas import VALID_ISSUE_TYPES


# =====================================================================
# 1-3: Accounting Balance & Clean Books
# =====================================================================

def test_clean_episode_accounting_balance():
    """Verify sum(debit_minor) == sum(credit_minor) for a clean episode (num_errors=0)."""
    episode = generate_episode(seed=42, config={"num_errors": 0})
    total_debit = sum(row.debit_minor for row in episode.ledger)
    total_credit = sum(row.credit_minor for row in episode.ledger)

    assert total_debit > 0
    assert total_debit == total_credit


def test_clean_episode_empty_truth_and_provenance():
    """Verify clean episodes (num_errors=0) produce no truth issues or provenance entries."""
    episode = generate_episode(seed=101, config={"num_errors": 0})
    assert len(episode.truth_issues) == 0
    assert len(episode.provenance) == 0


def test_accounting_balance_across_various_seeds():
    """Verify double-entry balance holds on clean episodes across different seeds and currencies."""
    for seed in [1, 7, 77, 314, 999]:
        episode = generate_episode(seed=seed, config={"num_errors": 0})
        debit = sum(r.debit_minor for r in episode.ledger)
        credit = sum(r.credit_minor for r in episode.ledger)
        assert debit == credit, f"Seed {seed} failed balance: debit={debit}, credit={credit}"


# =====================================================================
# 4-5: Determinism
# =====================================================================

def test_determinism_identical_runs():
    """Two calls with the exact same seed and config produce identical objects and data."""
    ep1 = generate_episode(seed=2026, config={"num_errors": 4, "target_docs": 40})
    ep2 = generate_episode(seed=2026, config={"num_errors": 4, "target_docs": 40})

    assert ep1.manifest == ep2.manifest
    assert ep1.policy == ep2.policy
    assert ep1.ledger == ep2.ledger
    assert ep1.invoices == ep2.invoices
    assert ep1.payments == ep2.payments
    assert ep1.fx_rates == ep2.fx_rates
    assert ep1.truth_issues == ep2.truth_issues
    assert ep1.provenance == ep2.provenance


def test_different_seeds_produce_different_episodes():
    """Different seeds must yield distinct episodes (different IDs and row counts)."""
    ep1 = generate_episode(seed=111)
    ep2 = generate_episode(seed=222)

    assert ep1.manifest.episode_id != ep2.manifest.episode_id
    assert [r.row_id for r in ep1.ledger] != [r.row_id for r in ep2.ledger]


# =====================================================================
# 6-7: Distractors (Clean Cases)
# =====================================================================

def test_distractors_presence_in_clean_episodes():
    """Verify legitimate distractors exist (split payments, refunds, recurring) without false issues."""
    ep = generate_episode(seed=55, config={"num_errors": 0, "include_distractors": True})

    # Refund / credit note invoices have negative total_minor
    refund_invoices = [inv for inv in ep.invoices if inv.total_minor < 0]
    assert len(refund_invoices) > 0, "Expected refund invoices when include_distractors=True"

    # Split payments: an invoice referenced by multiple payments
    inv_pay_count: dict[str, int] = {}
    for pay in ep.payments:
        for ref in pay.invoice_refs:
            inv_pay_count[ref] = inv_pay_count.get(ref, 0) + 1
    split_invs = [inv_id for inv_id, count in inv_pay_count.items() if count >= 2]
    assert len(split_invs) > 0, "Expected split payments when include_distractors=True"

    # Recurring payments: multiple invoices with identical entity_id and total_minor
    invoice_signatures = [(inv.entity_id, inv.total_minor) for inv in ep.invoices]
    has_duplicates = len(invoice_signatures) > len(set(invoice_signatures))
    assert has_duplicates, "Expected recurring invoices with equal amounts"

    # Most importantly: none of these distractors must appear in truth_issues
    assert len(ep.truth_issues) == 0


def test_distractors_can_be_disabled():
    """When include_distractors is False, no refund invoices should be created."""
    ep = generate_episode(seed=55, config={"num_errors": 0, "include_distractors": False})
    refund_invoices = [inv for inv in ep.invoices if inv.total_minor < 0]
    assert len(refund_invoices) == 0


# =====================================================================
# 8-11: Error Injection & Provenance
# =====================================================================

def test_error_injection_num_errors():
    """Verify truth_issues are populated when num_errors > 0."""
    ep = generate_episode(seed=42, config={"num_errors": 4})
    assert len(ep.truth_issues) > 0
    assert len(ep.truth_issues) <= 12
    for issue in ep.truth_issues:
        assert issue.type in VALID_ISSUE_TYPES


def test_error_types_all_four_generated():
    """Verify that across episodes, all 4 canonical error types can be generated."""
    observed_types: set[str] = set()

    for seed in range(10, 25):
        ep = generate_episode(seed=seed, config={"num_errors": 8, "include_compound": False})
        for issue in ep.truth_issues:
            observed_types.add(issue.type)
        if observed_types == set(VALID_ISSUE_TYPES):
            break

    assert observed_types == set(VALID_ISSUE_TYPES), (
        f"Missing error types: {set(VALID_ISSUE_TYPES) - observed_types}"
    )


def test_compound_scenarios_generated():
    """Verify compound error scenarios are recorded in provenance when enabled."""
    found_compound = False
    for seed in range(100, 115):
        ep = generate_episode(seed=seed, config={"num_errors": 6, "include_compound": True})
        actions = [p.action for p in ep.provenance]
        if any(a.startswith("inject_compound") for a in actions):
            found_compound = True
            break

    assert found_compound, "Expected compound error scenarios to be generated and logged"


def test_provenance_records_every_mutation():
    """Verify provenance entries have valid steps, actions, target_ids, and descriptions."""
    ep = generate_episode(seed=777, config={"num_errors": 5})
    assert len(ep.provenance) > 0

    for entry in ep.provenance:
        assert entry.step >= 1
        assert len(entry.action) > 0
        assert len(entry.target_ids) > 0
        assert len(entry.description) > 0


# =====================================================================
# 12-13: Referential Integrity
# =====================================================================

def test_referential_integrity_affected_ids():
    """Every ID in affected_ids must exist in open invoices or payments."""
    for seed in [12, 34, 56]:
        ep = generate_episode(seed=seed, config={"num_errors": 6})
        all_inv_ids = {inv.invoice_id for inv in ep.invoices}
        all_pay_ids = {pay.payment_id for pay in ep.payments}
        known_ids = all_inv_ids | all_pay_ids

        for issue in ep.truth_issues:
            for affected_id in issue.affected_ids:
                assert affected_id in known_ids, (
                    f"affected_id '{affected_id}' from issue '{issue.issue_id}' "
                    f"not found in invoices or payments"
                )


def test_referential_integrity_evidence_ids():
    """Every ID in acceptable_evidence_sets must exist in open documents or ledger rows."""
    for seed in [12, 34, 56]:
        ep = generate_episode(seed=seed, config={"num_errors": 6})
        all_doc_ids = {inv.doc_id for inv in ep.invoices} | {pay.doc_id for pay in ep.payments}
        all_row_ids = {row.row_id for row in ep.ledger}
        known_evidence = all_doc_ids | all_row_ids

        for issue in ep.truth_issues:
            for ev_set in issue.acceptable_evidence_sets:
                for ev_id in ev_set:
                    assert ev_id in known_evidence, (
                        f"evidence_id '{ev_id}' in issue '{issue.issue_id}' "
                        f"not found in doc_ids or ledger row_ids"
                    )


# =====================================================================
# 14-15: Manifest & FX Consistency
# =====================================================================

def test_manifest_consistency():
    """Verify manifest records match the actual counts in episode data."""
    ep = generate_episode(seed=888, config={"num_errors": 3})
    assert ep.manifest.num_records == len(ep.ledger)
    assert ep.manifest.num_documents == len(ep.invoices) + len(ep.payments)
    assert ep.manifest.period_start < ep.manifest.period_end
    assert ep.manifest.schema_version == "1.0"


def test_fx_rates_coverage():
    """Verify daily FX rates are generated as decimal strings and match date range."""
    ep = generate_episode(seed=432)
    assert len(ep.fx_rates) > 0
    for fx in ep.fx_rates:
        assert fx.source_currency == ep.manifest.base_currency
        assert float(fx.rate) > 0
        assert len(fx.date) == 10  # YYYY-MM-DD


# =====================================================================
# 16-17: Save and Load Roundtrip
# =====================================================================

def test_save_and_load_episode_lossless(tmp_path: Path):
    """Verify save_episode and load_episode accurately restore all data structures."""
    ep_orig = generate_episode(seed=1234, config={"num_errors": 5})
    save_episode(ep_orig, tmp_path)
    ep_loaded = load_episode(tmp_path)

    assert ep_loaded.manifest == ep_orig.manifest
    assert ep_loaded.policy == ep_orig.policy
    assert ep_loaded.ledger == ep_orig.ledger
    assert ep_loaded.invoices == ep_orig.invoices
    assert ep_loaded.payments == ep_orig.payments
    assert ep_loaded.fx_rates == ep_orig.fx_rates
    assert ep_loaded.truth_issues == ep_orig.truth_issues
    assert ep_loaded.provenance == ep_orig.provenance


def test_save_episode_file_structure(tmp_path: Path):
    """Verify open files are saved in root directory and closed files in closed/ subdirectory."""
    ep = generate_episode(seed=5678, config={"num_errors": 2})
    save_episode(ep, tmp_path)

    # Open files in root
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "policy.json").exists()
    assert (tmp_path / "ledger.csv").exists()
    assert (tmp_path / "invoices.jsonl").exists()
    assert (tmp_path / "payments.jsonl").exists()
    assert (tmp_path / "fx.csv").exists()

    # Closed evaluator files
    closed_dir = tmp_path / "closed"
    assert closed_dir.is_dir()
    assert (closed_dir / "truth.json").exists()
    assert (closed_dir / "provenance.json").exists()


# =====================================================================
# 18: Independent Verification via Open Env Tools
# =====================================================================

def test_independent_verification_via_env_tools():
    """Verify that affected_ids and evidence sets for all error types, compound scenarios,
    and clean books can be independently fetched and verified via open AuditEnv tools."""
    from audit_env.env import AuditEnv

    types_seen = set()
    verified_cross_currency_partial = False

    def read_table(env, table: str) -> list[dict]:
        records = []
        cursor = 0
        while True:
            response, _, _, _, _ = env.step({
                "tool": "list_records", "table": table, "cursor": cursor, "limit": 50,
            })
            assert response["type"] == "list_records"
            records.extend(response["records"])
            cursor = response["next_cursor"]
            if cursor is None:
                return records

    cases = (
        (14, {"num_errors": 8, "include_compound": True}),
        (42, {"num_errors": 8, "include_compound": False}),
        (999, {"num_errors": 0}),
    )
    for seed, config in cases:
        env = AuditEnv(eval_mode=True)
        episode = generate_episode(seed=seed, config=config)
        obs, _ = env.reset(episode_data=episode)
        assert obs["policy_summary"]["fx_date_rule"] == episode.policy.fx_date_rule
        base_currency = obs["manifest"]["base_currency"]

        # Fetch all amounts and FX rates only through the public environment tools.
        invoices = read_table(env, "invoices")
        payments = read_table(env, "payments")
        fx_rows = read_table(env, "fx_rates")
        ledger = read_table(env, "ledger")
        assert ledger
        if config["num_errors"] == 0:
            assert not episode.truth_issues
            continue

        invoices_by_id = {row["invoice_id"]: row for row in invoices}
        payments_by_id = {row["payment_id"]: row for row in payments}
        rates = {
            (row["date"], row["source_currency"], row["target_currency"]): Decimal(row["rate"])
            for row in fx_rows
        }
        policy = obs["policy_summary"]

        def rounded_minor(value: Decimal) -> int:
            return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))

        def payment_amount_in_invoice_currency(payment: dict, invoice: dict) -> int:
            if payment["currency"] == invoice["currency"]:
                return payment["amount_minor"]
            rate_date = (
                invoice["issued_at"][:10]
                if policy["fx_date_rule"] == "invoice_date"
                else payment["timestamp"][:10]
            )
            if payment["currency"] == base_currency:
                rate = rates[(rate_date, base_currency, invoice["currency"])]
                return rounded_minor(Decimal(payment["amount_minor"]) / rate)
            if invoice["currency"] == base_currency:
                rate = rates[(rate_date, base_currency, payment["currency"])]
                return rounded_minor(Decimal(payment["amount_minor"]) * rate)

            payment_rate = rates[(rate_date, base_currency, payment["currency"])]
            invoice_rate = rates[(rate_date, base_currency, invoice["currency"])]
            return rounded_minor(
                Decimal(payment["amount_minor"]) * payment_rate / invoice_rate
            )

        for issue in episode.truth_issues:
            types_seen.add(issue.type)
            for aid in issue.affected_ids:
                if aid.startswith("INV-"):
                    response, *_ = env.step({
                        "tool": "get_record", "table": "invoices", "id": aid,
                    })
                    assert response["type"] == "get_record"
                    assert response["record"]["invoice_id"] == aid
                elif aid.startswith("PAY-"):
                    response, *_ = env.step({
                        "tool": "get_record", "table": "payments", "id": aid,
                    })
                    assert response["type"] == "get_record"
                    assert response["record"]["payment_id"] == aid

            assert issue.acceptable_evidence_sets
            evidence_set_verified = False
            for ev_set in issue.acceptable_evidence_sets:
                ev_records = []
                for evidence_id in ev_set:
                    if evidence_id.startswith("DOC-"):
                        response, *_ = env.step({
                            "tool": "get_document", "doc_id": evidence_id,
                        })
                        if response.get("type") == "get_document" and response.get("document"):
                            ev_records.append(response["document"])
                    elif evidence_id.startswith("ROW-"):
                        response, *_ = env.step({
                            "tool": "get_record", "table": "ledger", "id": evidence_id,
                        })
                        if response.get("type") == "get_record" and response.get("record"):
                            ev_records.append(response["record"])

                if len(ev_records) == len(ev_set):
                    evidence_set_verified = True
                    if issue.type == "duplicate_invoice":
                        doc1, doc2 = ev_records[0], ev_records[1]
                        assert doc1["entity_id"] == doc2["entity_id"]
                        assert doc1["total_minor"] == doc2["total_minor"] == issue.expected_amount_minor
                    elif issue.type == "cutoff_error":
                        inv_doc = ev_records[0]
                        assert inv_doc["issued_at"][:10] > episode.manifest.period_end[:10]
                        assert inv_doc["total_minor"] == issue.expected_amount_minor
                    break
            assert evidence_set_verified, (
                f"No complete acceptable evidence set found for issue {issue.issue_id}"
            )

            if issue.type == "wrong_fx_date_or_rate":
                invoice_id, payment_id = issue.affected_ids
                invoice = invoices_by_id[invoice_id]
                payment = payments_by_id[payment_id]
                assert invoice["currency"] != payment["currency"] == base_currency
                rate_date = (
                    invoice["issued_at"][:10]
                    if policy["fx_date_rule"] == "invoice_date"
                    else payment["timestamp"][:10]
                )
                correct_rate = rates[(rate_date, base_currency, invoice["currency"])]
                expected_payment_amount = rounded_minor(
                    Decimal(invoice["total_minor"]) * correct_rate
                )
                assert expected_payment_amount != payment["amount_minor"]
                calculated_discrepancy = abs(
                    expected_payment_amount - payment["amount_minor"]
                )
                assert abs(
                    calculated_discrepancy - issue.expected_amount_minor
                ) <= episode.policy.amount_tolerance_minor

            elif issue.type == "unallocated_or_partial_payment":
                invoice_id = next(
                    aid for aid in issue.affected_ids if aid in invoices_by_id
                )
                invoice = invoices_by_id[invoice_id]
                affected_payments = [
                    payments_by_id[aid]
                    for aid in issue.affected_ids
                    if aid in payments_by_id
                ]
                assert affected_payments
                paid_amount = sum(
                    payment_amount_in_invoice_currency(payment, invoice)
                    for payment in affected_payments
                )
                calculated_shortfall = invoice["total_minor"] - paid_amount
                assert abs(
                    calculated_shortfall - issue.expected_amount_minor
                ) <= episode.policy.amount_tolerance_minor
                if any(payment["currency"] != invoice["currency"] for payment in affected_payments):
                    verified_cross_currency_partial = True

    assert len(types_seen) >= 4, f"Expected all 4 types to be tested, got {types_seen}"
    assert verified_cross_currency_partial, "Expected a cross-currency partial-payment case"
