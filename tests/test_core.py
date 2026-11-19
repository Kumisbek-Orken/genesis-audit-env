"""
Unit tests for audit_env.schemas and audit_env.calculator.

Covers:
  - Valid and invalid dataclass construction for every schema entity.
  - Submission limits: >30 findings, >10 evidence_ids, >1000 chars explanation.
  - Canonical error types validation.
  - Calculator: basic arithmetic, Decimal precision, edge cases.
  - Calculator: code-injection attempts (__import__, os.system, etc.).
  - Calculator: disallowed constructs (variables, function calls, power, etc.).
"""

from __future__ import annotations

import pytest

from audit_env.schemas import (
    VALID_ISSUE_TYPES,
    MAX_FINDINGS,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_LEN,
    SchemaError,
    Manifest,
    Policy,
    LedgerRow,
    LineItem,
    Invoice,
    Payment,
    FXRate,
    TruthIssue,
    ProvenanceEntry,
    Finding,
    Submission,
    validate_submission,
)
from audit_env.calculator import CalculatorError, calculate


# ===================================================================
# Helpers – reusable valid data
# ===================================================================

def _valid_manifest_dict() -> dict:
    return {
        "schema_version": "1.0",
        "episode_id": "ep-001",
        "base_currency": "USD",
        "period_start": "2026-01-01T00:00:00+05:00",
        "period_end": "2026-12-31T23:59:59+05:00",
        "timezone": "Asia/Almaty",
        "num_records": 500,
        "num_documents": 50,
        "rounding_rule": "ROUND_HALF_EVEN",
    }


def _valid_finding_dict() -> dict:
    return {
        "type": "duplicate_invoice",
        "affected_ids": ["inv-001", "inv-002"],
        "claimed_amount_minor": 100000,
        "currency": "USD",
        "evidence_ids": ["doc-001", "row-005"],
        "explanation": "Invoice inv-002 is a duplicate of inv-001.",
    }


def _valid_submission_dict(n_findings: int = 1) -> dict:
    return {
        "episode_id": "ep-001",
        "findings": [_valid_finding_dict() for _ in range(n_findings)],
    }


# ===================================================================
# SCHEMAS – Manifest
# ===================================================================

class TestManifest:
    def test_valid(self):
        m = Manifest.from_dict(_valid_manifest_dict())
        assert m.episode_id == "ep-001"
        assert m.to_dict() == _valid_manifest_dict()

    def test_roundtrip(self):
        d = _valid_manifest_dict()
        assert Manifest.from_dict(d).to_dict() == d

    def test_missing_field(self):
        d = _valid_manifest_dict()
        del d["episode_id"]
        with pytest.raises(SchemaError):
            Manifest.from_dict(d)

    def test_bad_timestamp(self):
        d = _valid_manifest_dict()
        d["period_start"] = "not-a-date"
        with pytest.raises(SchemaError, match="ISO-8601"):
            Manifest.from_dict(d)

    def test_num_records_not_int(self):
        d = _valid_manifest_dict()
        d["num_records"] = "500"
        with pytest.raises(SchemaError, match="integer"):
            Manifest.from_dict(d)

    def test_bool_not_int(self):
        d = _valid_manifest_dict()
        d["num_records"] = True
        with pytest.raises(SchemaError, match="integer"):
            Manifest.from_dict(d)


# ===================================================================
# SCHEMAS – Policy
# ===================================================================

class TestPolicy:
    def test_valid(self):
        p = Policy(fx_date_rule="invoice_date", period_boundary="inclusive_start_exclusive_end", rounding="ROUND_HALF_EVEN")
        assert p.fx_date_rule == "invoice_date"
        assert p.amount_tolerance_minor == 0  # default

    def test_with_tolerance(self):
        p = Policy(fx_date_rule="invoice_date", period_boundary="x", rounding="x", amount_tolerance_minor=100)
        assert p.amount_tolerance_minor == 100

    def test_negative_tolerance(self):
        with pytest.raises(SchemaError, match=">= 0"):
            Policy(fx_date_rule="x", period_boundary="x", rounding="x", amount_tolerance_minor=-1)

    def test_empty_field(self):
        with pytest.raises(SchemaError, match="non-empty"):
            Policy(fx_date_rule="", period_boundary="x", rounding="x")


# ===================================================================
# SCHEMAS – LedgerRow
# ===================================================================

class TestLedgerRow:
    def test_valid(self):
        lr = LedgerRow(
            row_id="R001",
            timestamp="2026-06-15T12:00:00+05:00",
            account="AP",
            debit_minor=10000,
            credit_minor=0,
            currency="USD",
            entity_id="E001",
            invoice_id="INV001",
        )
        assert lr.row_id == "R001"
        assert lr.payment_id is None

    def test_null_refs(self):
        lr = LedgerRow.from_dict({
            "row_id": "R002",
            "timestamp": "2026-06-15T12:00:00+05:00",
            "account": "AP",
            "debit_minor": 0,
            "credit_minor": 10000,
            "currency": "USD",
            "entity_id": "E001",
        })
        assert lr.invoice_id is None
        assert lr.payment_id is None


# ===================================================================
# SCHEMAS – Invoice / LineItem
# ===================================================================

class TestInvoice:
    def test_valid_with_line_items(self):
        d = {
            "doc_id": "D001",
            "invoice_id": "INV001",
            "entity_id": "E001",
            "issued_at": "2026-03-01T10:00:00+05:00",
            "due_at": "2026-04-01T10:00:00+05:00",
            "currency": "USD",
            "total_minor": 50000,
            "line_items": [
                {"description": "Widget A", "amount_minor": 30000, "currency": "USD"},
                {"description": "Widget B", "amount_minor": 20000, "currency": "USD"},
            ],
        }
        inv = Invoice.from_dict(d)
        assert len(inv.line_items) == 2
        assert inv.total_minor == 50000

    def test_missing_doc_id(self):
        d = {
            "invoice_id": "INV001",
            "entity_id": "E001",
            "issued_at": "2026-03-01T10:00:00+05:00",
            "due_at": "2026-04-01T10:00:00+05:00",
            "currency": "USD",
            "total_minor": 50000,
        }
        with pytest.raises(SchemaError):
            Invoice.from_dict(d)


# ===================================================================
# SCHEMAS – Payment
# ===================================================================

class TestPayment:
    def test_valid(self):
        p = Payment.from_dict({
            "doc_id": "D100",
            "payment_id": "PAY001",
            "timestamp": "2026-05-01T09:00:00+05:00",
            "currency": "USD",
            "amount_minor": 50000,
            "invoice_refs": ["INV001"],
        })
        assert p.payment_id == "PAY001"
        assert p.invoice_refs == ("INV001",)


# ===================================================================
# SCHEMAS – FXRate
# ===================================================================

class TestFXRate:
    def test_valid(self):
        fx = FXRate(date="2026-06-15", source_currency="EUR", target_currency="USD", rate="1.0850")
        assert fx.rate == "1.0850"

    def test_bad_rate(self):
        with pytest.raises(SchemaError, match="decimal string"):
            FXRate(date="2026-06-15", source_currency="EUR", target_currency="USD", rate="abc")

    def test_bad_date(self):
        with pytest.raises(SchemaError, match="YYYY-MM-DD"):
            FXRate(date="15/06/2026", source_currency="EUR", target_currency="USD", rate="1.08")


# ===================================================================
# SCHEMAS – TruthIssue (closed)
# ===================================================================

class TestTruthIssue:
    def test_valid(self):
        t = TruthIssue(
            issue_id="ISS-001",
            type="duplicate_invoice",
            affected_ids=("INV001", "INV002"),
            expected_amount_minor=50000,
            currency="USD",
            acceptable_evidence_sets=(("D001", "D002"),),
        )
        assert t.type == "duplicate_invoice"

    def test_invalid_type(self):
        with pytest.raises(SchemaError, match="type must be one of"):
            TruthIssue(
                issue_id="ISS-X",
                type="nonexistent_error",
                expected_amount_minor=0,
                currency="USD",
            )


# ===================================================================
# SCHEMAS – ProvenanceEntry (closed)
# ===================================================================

class TestProvenanceEntry:
    def test_valid(self):
        pe = ProvenanceEntry(step=1, action="inject_duplicate_invoice",
                             target_ids=("INV001",), description="Duplicated INV001")
        assert pe.step == 1

    def test_step_not_int(self):
        with pytest.raises(SchemaError, match="integer"):
            ProvenanceEntry(step="one", action="inject_duplicate_invoice")


# ===================================================================
# SCHEMAS – Finding / Submission
# ===================================================================

class TestFinding:
    def test_valid(self):
        f = Finding.from_dict(_valid_finding_dict())
        assert f.type == "duplicate_invoice"

    def test_invalid_type(self):
        d = _valid_finding_dict()
        d["type"] = "made_up_type"
        with pytest.raises(SchemaError, match="type must be one of"):
            Finding.from_dict(d)

    def test_too_many_evidence_ids(self):
        d = _valid_finding_dict()
        d["evidence_ids"] = [f"ev-{i}" for i in range(MAX_EVIDENCE_IDS + 1)]
        with pytest.raises(SchemaError, match="at most 10"):
            Finding.from_dict(d)

    def test_explanation_too_long(self):
        d = _valid_finding_dict()
        d["explanation"] = "x" * (MAX_EXPLANATION_LEN + 1)
        with pytest.raises(SchemaError, match="at most 1000"):
            Finding.from_dict(d)

    def test_claimed_amount_not_int(self):
        d = _valid_finding_dict()
        d["claimed_amount_minor"] = 100.5
        with pytest.raises(SchemaError, match="integer"):
            Finding.from_dict(d)


class TestSubmission:
    def test_valid(self):
        s = validate_submission(_valid_submission_dict())
        assert s.episode_id == "ep-001"
        assert len(s.findings) == 1

    def test_too_many_findings(self):
        raw = _valid_submission_dict(n_findings=MAX_FINDINGS + 1)
        with pytest.raises(SchemaError, match="at most 30"):
            validate_submission(raw)

    def test_exactly_at_limit(self):
        raw = _valid_submission_dict(n_findings=MAX_FINDINGS)
        s = validate_submission(raw)
        assert len(s.findings) == MAX_FINDINGS

    def test_missing_episode_id(self):
        raw = {"findings": []}
        with pytest.raises(SchemaError):
            validate_submission(raw)

    def test_non_dict_input(self):
        with pytest.raises(SchemaError, match="JSON object"):
            validate_submission("not a dict")

    def test_empty_findings_ok(self):
        raw = {"episode_id": "ep-clean", "findings": []}
        s = validate_submission(raw)
        assert len(s.findings) == 0


# ===================================================================
# SCHEMAS – Canonical error types
# ===================================================================

class TestCanonicalTypes:
    def test_exactly_four(self):
        assert len(VALID_ISSUE_TYPES) == 4

    def test_expected_members(self):
        expected = {
            "duplicate_invoice",
            "unallocated_or_partial_payment",
            "wrong_fx_date_or_rate",
            "cutoff_error",
        }
        assert VALID_ISSUE_TYPES == expected


# ===================================================================
# CALCULATOR – valid expressions
# ===================================================================

class TestCalculatorValid:
    def test_addition(self):
        assert calculate("2 + 3") == "5"

    def test_subtraction(self):
        assert calculate("10 - 4") == "6"

    def test_multiplication(self):
        assert calculate("7 * 6") == "42"

    def test_division(self):
        assert calculate("10 / 4") == "2.5"

    def test_decimal_precision(self):
        # 0.1 + 0.2 should be exact in Decimal, not 0.30000000000000004
        result = calculate("0.1 + 0.2")
        assert result == "0.3"

    def test_nested_parentheses(self):
        assert calculate("((2 + 3) * (4 - 1))") == "15"

    def test_unary_minus(self):
        assert calculate("-5 + 3") == "-2"

    def test_unary_plus(self):
        assert calculate("+5") == "5"

    def test_complex_expression(self):
        # (100.50 * 3 - 50) / 2
        result = calculate("(100.50 * 3 - 50) / 2")
        assert result == "125.75"

    def test_large_numbers(self):
        result = calculate("999999999 * 999999999")
        assert result == "999999998000000001"

    def test_negative_result(self):
        assert calculate("3 - 10") == "-7"

    def test_zero(self):
        assert calculate("0") == "0"


# ===================================================================
# CALCULATOR – error cases
# ===================================================================

class TestCalculatorErrors:
    def test_division_by_zero(self):
        with pytest.raises(CalculatorError, match="Division by zero"):
            calculate("1 / 0")

    def test_empty_string(self):
        with pytest.raises(CalculatorError, match="empty"):
            calculate("")

    def test_non_string(self):
        with pytest.raises(CalculatorError, match="string"):
            calculate(42)

    def test_too_long(self):
        expr = "1 + " * 600 + "1"
        with pytest.raises(CalculatorError, match="too long"):
            calculate(expr)


# ===================================================================
# CALCULATOR – code injection attempts
# ===================================================================

class TestCalculatorInjection:
    def test_import_os(self):
        with pytest.raises(CalculatorError):
            calculate("__import__('os').system('ls')")

    def test_eval(self):
        with pytest.raises(CalculatorError):
            calculate("eval('1+1')")

    def test_exec(self):
        with pytest.raises(CalculatorError):
            calculate("exec('print(1)')")

    def test_open_file(self):
        with pytest.raises(CalculatorError):
            calculate("open('/etc/passwd').read()")

    def test_attribute_access(self):
        with pytest.raises(CalculatorError):
            calculate("().__class__.__bases__")

    def test_lambda(self):
        with pytest.raises(CalculatorError):
            calculate("(lambda: 1)()")

    def test_list_comprehension(self):
        with pytest.raises(CalculatorError):
            calculate("[x for x in range(10)]")

    def test_dunder_import(self):
        with pytest.raises(CalculatorError):
            calculate("__import__('subprocess').call(['ls'])")

    def test_variable_name(self):
        with pytest.raises(CalculatorError):
            calculate("x + 1")

    def test_string_literal(self):
        with pytest.raises(CalculatorError):
            calculate("'hello'")

    def test_boolean_literal(self):
        with pytest.raises(CalculatorError):
            calculate("True + 1")

    def test_comparison(self):
        with pytest.raises(CalculatorError):
            calculate("1 < 2")

    def test_power_operator(self):
        with pytest.raises(CalculatorError):
            calculate("2 ** 100")

    def test_modulo(self):
        with pytest.raises(CalculatorError):
            calculate("10 % 3")

    def test_floor_division(self):
        with pytest.raises(CalculatorError):
            calculate("10 // 3")

    def test_bitwise_or(self):
        with pytest.raises(CalculatorError):
            calculate("1 | 2")

    def test_semicolon_statement(self):
        with pytest.raises(CalculatorError):
            calculate("1; import os")

    def test_infinity_literal(self):
        with pytest.raises(CalculatorError):
            calculate("float('inf')")

    def test_nested_dunder(self):
        with pytest.raises(CalculatorError):
            calculate("(1).__class__.__name__")
