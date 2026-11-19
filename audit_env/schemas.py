"""
Strict dataclass models and validators for all GENESIS RL-environment entities.

Covers:
  - Open data: manifest.json, policy.json, ledger.csv rows, invoices.jsonl,
    payments.jsonl, fx.csv rows.
  - Closed evaluator files: truth.json issues, provenance.json entries.
  - Agent submission: findings.json with limits (<=30 findings, <=10
    evidence_ids per finding, <=1000 chars explanation).
  - Canonical error type constants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Canonical error types (section 02 / 05 of SPEC)
# ---------------------------------------------------------------------------

VALID_ISSUE_TYPES: frozenset[str] = frozenset(
    {
        "duplicate_invoice",
        "unallocated_or_partial_payment",
        "wrong_fx_date_or_rate",
        "cutoff_error",
    }
)

# ---------------------------------------------------------------------------
# Submission limits (section 05 of SPEC)
# ---------------------------------------------------------------------------

MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_LEN = 1000

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class SchemaError(Exception):
    """Raised when data violates the GENESIS schema."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


_ISO8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z)$"
)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _check_iso8601(value: str, field_name: str) -> None:
    _require(
        isinstance(value, str) and _ISO8601_RE.match(value) is not None,
        f"{field_name} must be an ISO-8601 datetime with timezone offset, got {value!r}",
    )


def _check_date(value: str, field_name: str) -> None:
    _require(
        isinstance(value, str) and _DATE_RE.match(value) is not None,
        f"{field_name} must be a YYYY-MM-DD date string, got {value!r}",
    )


def _check_str(value: Any, field_name: str) -> None:
    _require(isinstance(value, str) and len(value) > 0,
             f"{field_name} must be a non-empty string, got {value!r}")


def _check_int(value: Any, field_name: str) -> None:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{field_name} must be an integer, got {value!r}")


def _check_optional_str(value: Any, field_name: str) -> None:
    if value is not None:
        _check_str(value, field_name)


# ---------------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Manifest:
    """Episode manifest (section 03)."""

    schema_version: str
    episode_id: str
    base_currency: str
    period_start: str  # ISO-8601
    period_end: str    # ISO-8601
    timezone: str
    num_records: int
    num_documents: int
    rounding_rule: str  # e.g. "ROUND_HALF_EVEN"

    def __post_init__(self) -> None:
        _check_str(self.schema_version, "schema_version")
        _check_str(self.episode_id, "episode_id")
        _check_str(self.base_currency, "base_currency")
        _check_iso8601(self.period_start, "period_start")
        _check_iso8601(self.period_end, "period_end")
        _check_str(self.timezone, "timezone")
        _check_int(self.num_records, "num_records")
        _check_int(self.num_documents, "num_documents")
        _check_str(self.rounding_rule, "rounding_rule")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        try:
            return cls(**{k: d[k] for k in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid manifest data: {exc}") from exc

# ---------------------------------------------------------------------------
# policy.json
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    """Audit policy controlling FX date selection, period boundaries, rounding."""

    fx_date_rule: str         # e.g. "invoice_date" | "payment_date"
    period_boundary: str      # e.g. "inclusive_start_exclusive_end"
    rounding: str             # e.g. "ROUND_HALF_EVEN"
    amount_tolerance_minor: int = 0  # absolute tolerance for amount matching

    def __post_init__(self) -> None:
        _check_str(self.fx_date_rule, "fx_date_rule")
        _check_str(self.period_boundary, "period_boundary")
        _check_str(self.rounding, "rounding")
        _check_int(self.amount_tolerance_minor, "amount_tolerance_minor")
        _require(
            self.amount_tolerance_minor >= 0,
            f"amount_tolerance_minor must be >= 0, got {self.amount_tolerance_minor}",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Policy":
        try:
            fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
            return cls(**fields)
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid policy data: {exc}") from exc

# ---------------------------------------------------------------------------
# ledger.csv rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LedgerRow:
    """Single row from ledger.csv (section 03)."""

    row_id: str
    timestamp: str  # ISO-8601
    account: str
    debit_minor: int
    credit_minor: int
    currency: str
    entity_id: str
    invoice_id: Optional[str] = None
    payment_id: Optional[str] = None

    def __post_init__(self) -> None:
        _check_str(self.row_id, "row_id")
        _check_iso8601(self.timestamp, "timestamp")
        _check_str(self.account, "account")
        _check_int(self.debit_minor, "debit_minor")
        _check_int(self.credit_minor, "credit_minor")
        _check_str(self.currency, "currency")
        _check_str(self.entity_id, "entity_id")
        _check_optional_str(self.invoice_id, "invoice_id")
        _check_optional_str(self.payment_id, "payment_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LedgerRow":
        try:
            return cls(
                row_id=d["row_id"],
                timestamp=d["timestamp"],
                account=d["account"],
                debit_minor=d["debit_minor"],
                credit_minor=d["credit_minor"],
                currency=d["currency"],
                entity_id=d["entity_id"],
                invoice_id=d.get("invoice_id"),
                payment_id=d.get("payment_id"),
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid ledger row: {exc}") from exc

# ---------------------------------------------------------------------------
# invoices.jsonl
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineItem:
    """Single line item inside an invoice."""

    description: str
    amount_minor: int
    currency: str

    def __post_init__(self) -> None:
        _check_str(self.description, "description")
        _check_int(self.amount_minor, "amount_minor")
        _check_str(self.currency, "currency")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LineItem":
        try:
            return cls(**{k: d[k] for k in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid line item: {exc}") from exc


@dataclass(frozen=True)
class Invoice:
    """Single invoice record (section 03)."""

    doc_id: str
    invoice_id: str
    entity_id: str
    issued_at: str   # ISO-8601
    due_at: str      # ISO-8601
    currency: str
    total_minor: int
    line_items: tuple[LineItem, ...] = ()

    def __post_init__(self) -> None:
        _check_str(self.doc_id, "doc_id")
        _check_str(self.invoice_id, "invoice_id")
        _check_str(self.entity_id, "entity_id")
        _check_iso8601(self.issued_at, "issued_at")
        _check_iso8601(self.due_at, "due_at")
        _check_str(self.currency, "currency")
        _check_int(self.total_minor, "total_minor")
        for li in self.line_items:
            _require(isinstance(li, LineItem),
                     f"line_items must contain LineItem instances, got {type(li)}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["line_items"] = [li.to_dict() for li in self.line_items]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Invoice":
        try:
            items = tuple(
                LineItem.from_dict(li) for li in d.get("line_items", [])
            )
            return cls(
                doc_id=d["doc_id"],
                invoice_id=d["invoice_id"],
                entity_id=d["entity_id"],
                issued_at=d["issued_at"],
                due_at=d["due_at"],
                currency=d["currency"],
                total_minor=d["total_minor"],
                line_items=items,
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid invoice: {exc}") from exc

# ---------------------------------------------------------------------------
# payments.jsonl
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Payment:
    """Single payment record (section 03)."""

    doc_id: str
    payment_id: str
    timestamp: str   # ISO-8601
    currency: str
    amount_minor: int
    invoice_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_str(self.doc_id, "doc_id")
        _check_str(self.payment_id, "payment_id")
        _check_iso8601(self.timestamp, "timestamp")
        _check_str(self.currency, "currency")
        _check_int(self.amount_minor, "amount_minor")
        for ref in self.invoice_refs:
            _require(isinstance(ref, str),
                     f"invoice_refs must contain strings, got {type(ref)}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["invoice_refs"] = list(self.invoice_refs)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Payment":
        try:
            refs = tuple(d.get("invoice_refs", []))
            return cls(
                doc_id=d["doc_id"],
                payment_id=d["payment_id"],
                timestamp=d["timestamp"],
                currency=d["currency"],
                amount_minor=d["amount_minor"],
                invoice_refs=refs,
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid payment: {exc}") from exc

# ---------------------------------------------------------------------------
# fx.csv rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FXRate:
    """Single FX rate entry (section 03).

    ``rate`` is kept as a string to preserve decimal precision; the calculator
    or Decimal should be used for arithmetic.
    """

    date: str  # YYYY-MM-DD
    source_currency: str
    target_currency: str
    rate: str  # decimal string

    def __post_init__(self) -> None:
        _check_date(self.date, "date")
        _check_str(self.source_currency, "source_currency")
        _check_str(self.target_currency, "target_currency")
        _check_str(self.rate, "rate")
        # Validate rate is a valid decimal string
        try:
            from decimal import Decimal, InvalidOperation
            Decimal(self.rate)
        except (InvalidOperation, ValueError):
            raise SchemaError(f"rate must be a valid decimal string, got {self.rate!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FXRate":
        try:
            return cls(**{k: d[k] for k in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid FX rate: {exc}") from exc

# ---------------------------------------------------------------------------
# truth.json  (closed — evaluator only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TruthIssue:
    """Single ground-truth issue (section 03/05, evaluator-only)."""

    issue_id: str
    type: str  # one of VALID_ISSUE_TYPES
    affected_ids: tuple[str, ...] = ()
    expected_amount_minor: int = 0
    currency: str = ""
    acceptable_evidence_sets: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        _check_str(self.issue_id, "issue_id")
        _require(
            self.type in VALID_ISSUE_TYPES,
            f"type must be one of {sorted(VALID_ISSUE_TYPES)}, got {self.type!r}",
        )
        for aid in self.affected_ids:
            _require(isinstance(aid, str),
                     f"affected_ids must contain strings, got {type(aid)}")
        _check_int(self.expected_amount_minor, "expected_amount_minor")
        _check_str(self.currency, "currency")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["affected_ids"] = list(self.affected_ids)
        d["acceptable_evidence_sets"] = [list(s) for s in self.acceptable_evidence_sets]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TruthIssue":
        try:
            return cls(
                issue_id=d["issue_id"],
                type=d["type"],
                affected_ids=tuple(d.get("affected_ids", [])),
                expected_amount_minor=d.get("expected_amount_minor", 0),
                currency=d.get("currency", ""),
                acceptable_evidence_sets=tuple(
                    tuple(s) for s in d.get("acceptable_evidence_sets", [])
                ),
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid truth issue: {exc}") from exc

# ---------------------------------------------------------------------------
# provenance.json  (closed — evaluator only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceEntry:
    """Record of a mutation applied during episode generation (evaluator-only)."""

    step: int
    action: str  # e.g. "inject_duplicate_invoice"
    target_ids: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        _check_int(self.step, "step")
        _check_str(self.action, "action")
        for tid in self.target_ids:
            _require(isinstance(tid, str),
                     f"target_ids must contain strings, got {type(tid)}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_ids"] = list(self.target_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProvenanceEntry":
        try:
            return cls(
                step=d["step"],
                action=d["action"],
                target_ids=tuple(d.get("target_ids", [])),
                description=d.get("description", ""),
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid provenance entry: {exc}") from exc

# ---------------------------------------------------------------------------
# findings.json  (agent submission — section 05)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """Single finding in the agent's submission."""

    type: str  # one of VALID_ISSUE_TYPES
    affected_ids: tuple[str, ...] = ()
    claimed_amount_minor: int = 0
    currency: str = ""
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        _require(
            self.type in VALID_ISSUE_TYPES,
            f"type must be one of {sorted(VALID_ISSUE_TYPES)}, got {self.type!r}",
        )
        for aid in self.affected_ids:
            _require(isinstance(aid, str),
                     f"affected_ids must contain strings, got {type(aid)}")
        _check_int(self.claimed_amount_minor, "claimed_amount_minor")
        _check_str(self.currency, "currency")
        _require(
            len(self.evidence_ids) <= MAX_EVIDENCE_IDS,
            f"evidence_ids must have at most {MAX_EVIDENCE_IDS} entries, "
            f"got {len(self.evidence_ids)}",
        )
        for eid in self.evidence_ids:
            _require(isinstance(eid, str),
                     f"evidence_ids must contain strings, got {type(eid)}")
        _require(
            len(self.explanation) <= MAX_EXPLANATION_LEN,
            f"explanation must be at most {MAX_EXPLANATION_LEN} characters, "
            f"got {len(self.explanation)}",
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["affected_ids"] = list(self.affected_ids)
        d["evidence_ids"] = list(self.evidence_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Finding":
        try:
            return cls(
                type=d["type"],
                affected_ids=tuple(d.get("affected_ids", [])),
                claimed_amount_minor=d.get("claimed_amount_minor", 0),
                currency=d.get("currency", ""),
                evidence_ids=tuple(d.get("evidence_ids", [])),
                explanation=d.get("explanation", ""),
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid finding: {exc}") from exc


@dataclass(frozen=True)
class Submission:
    """Full agent submission (section 05)."""

    episode_id: str
    findings: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        _check_str(self.episode_id, "episode_id")
        _require(
            len(self.findings) <= MAX_FINDINGS,
            f"findings must have at most {MAX_FINDINGS} entries, "
            f"got {len(self.findings)}",
        )
        for f in self.findings:
            _require(isinstance(f, Finding),
                     f"findings must contain Finding instances, got {type(f)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Submission":
        try:
            findings = tuple(
                Finding.from_dict(f) for f in d.get("findings", [])
            )
            return cls(
                episode_id=d["episode_id"],
                findings=findings,
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"Invalid submission: {exc}") from exc


# ---------------------------------------------------------------------------
# Convenience: validate a raw dict as a submission, returning errors or None
# ---------------------------------------------------------------------------

def validate_submission(raw: Any) -> Submission:
    """Parse and validate a raw dict/JSON-like object as a Submission.

    Raises SchemaError with a human-readable message on any violation.
    Returns a valid Submission dataclass on success.
    """
    _require(isinstance(raw, dict), "submission must be a JSON object (dict)")
    return Submission.from_dict(raw)
