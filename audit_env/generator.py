"""
Causal episode generator for the GENESIS RL-environment.

Implements sections 02, 03 and 07 of SPEC.md:
  Stage A – Build a consistent, balanced company history (ledger, invoices,
            payments, FX rates) where all accounting identities hold exactly.
  Stage B – Causally inject 0–12 errors (4 types + 2 compound scenarios),
            recording each mutation in provenance.json and truth.json.

All randomness flows through a local ``random.Random(seed)`` instance.
No global random, uuid4 or system time is used anywhere, ensuring full
deterministic reproducibility.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random as _random_module
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# EpisodeData container
# =====================================================================

@dataclass
class EpisodeData:
    """All generated data for one episode."""

    manifest: Manifest
    policy: Policy
    ledger: list[LedgerRow]
    invoices: list[Invoice]
    payments: list[Payment]
    fx_rates: list[FXRate]
    truth_issues: list[TruthIssue]
    provenance: list[ProvenanceEntry]


# =====================================================================
# Constants
# =====================================================================

_SCHEMA_VERSION = "1.0"

_CURRENCIES = ["USD", "EUR", "GBP", "KZT", "JPY", "CHF"]

_MINOR_EXPONENTS: dict[str, int] = {
    "USD": 2, "EUR": 2, "GBP": 2, "KZT": 2,
    "JPY": 0, "CHF": 2,
}

_ENTITY_NAMES = [
    "AlphaTech", "BetaLogistics", "GammaCorp", "DeltaSupply",
    "EpsilonServices", "ZetaManufacturing", "EtaConsulting",
    "ThetaTrading", "IotaDigital", "KappaFinance",
    "LambdaIndustries", "MuEngineering", "NuRetail",
    "XiConstruction", "OmicronPharma", "PiMedia",
]

_ITEM_DESCRIPTIONS = [
    "Office supplies", "Cloud hosting", "Consulting services",
    "Raw materials", "Equipment lease", "Software license",
    "Maintenance contract", "Transport services", "Insurance premium",
    "Marketing campaign", "Facility rent", "Staff training",
    "Security audit", "Legal services", "Telecommunications",
    "Warehouse storage",
]

_ACCOUNTS = [
    "accounts_payable", "accounts_receivable", "cash",
    "revenue", "cost_of_goods", "fx_gain_loss",
]

_TIMEZONES = {
    "Asia/Almaty": "+05:00",
    "Europe/London": "+00:00",
    "America/New_York": "-05:00",
    "Europe/Berlin": "+01:00",
    "Asia/Tokyo": "+09:00",
}

# =====================================================================
# Helpers
# =====================================================================

def _tz_offset(tz_name: str) -> str:
    return _TIMEZONES.get(tz_name, "+00:00")


def _make_ts(d: date, hour: int, minute: int, tz_offset: str) -> str:
    """Create ISO-8601 timestamp string."""
    return f"{d.isoformat()}T{hour:02d}:{minute:02d}:00{tz_offset}"


def _date_range(start: date, end: date) -> list[date]:
    """Inclusive date range."""
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def _decimal_round(value: Decimal, exponent: int) -> int:
    """Round a Decimal to *exponent* decimal places and return as int in minor units."""
    if exponent == 0:
        return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
    q = Decimal(10) ** (-exponent)
    rounded = value.quantize(q, rounding=ROUND_HALF_EVEN)
    return int(rounded * Decimal(10) ** exponent)


def _to_minor(amount: Decimal, currency: str) -> int:
    """Convert a Decimal amount to integer minor units for *currency*."""
    exp = _MINOR_EXPONENTS.get(currency, 2)
    factor = Decimal(10) ** exp
    return int((amount * factor).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def _from_minor(minor: int, currency: str) -> Decimal:
    """Convert integer minor units back to a Decimal amount."""
    exp = _MINOR_EXPONENTS.get(currency, 2)
    return Decimal(minor) / Decimal(10) ** exp


# =====================================================================
# ID generators (all purely from the local RNG)
# =====================================================================

class _IDGen:
    """Sequential, deterministic ID generator."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def next(self, prefix: str) -> str:
        c = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = c
        return f"{prefix}-{c:04d}"


# =====================================================================
# FX rate generation
# =====================================================================

def _generate_fx_rates(
    rng: _random_module.Random,
    base_currency: str,
    foreign_currencies: list[str],
    start: date,
    end: date,
) -> list[FXRate]:
    """Generate daily FX rates for each foreign currency pair."""
    rates: list[FXRate] = []
    days = _date_range(start, end)

    # Base rates (mid-market style).
    base_rates: dict[str, Decimal] = {}
    for fc in foreign_currencies:
        if fc == "JPY":
            base_rates[fc] = Decimal(str(round(rng.uniform(100, 160), 4)))
        elif fc == "KZT":
            base_rates[fc] = Decimal(str(round(rng.uniform(400, 500), 4)))
        else:
            base_rates[fc] = Decimal(str(round(rng.uniform(0.7, 1.5), 6)))

    for d in days:
        for fc in foreign_currencies:
            # Daily random walk ±0.5 %
            drift = Decimal(str(round(rng.uniform(-0.005, 0.005), 6)))
            base_rates[fc] = (base_rates[fc] * (Decimal("1") + drift)).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_EVEN
            )
            rate_str = str(base_rates[fc])
            rates.append(FXRate(
                date=d.isoformat(),
                source_currency=base_currency,
                target_currency=fc,
                rate=rate_str,
            ))
    return rates


def _get_rate(fx_rates: list[FXRate], dt: date, src: str, tgt: str) -> Decimal:
    """Look up the FX rate for a specific date and pair."""
    if src == tgt:
        return Decimal("1")
    for r in fx_rates:
        if r.date == dt.isoformat() and r.source_currency == src and r.target_currency == tgt:
            return Decimal(r.rate)
    # Try reverse
    for r in fx_rates:
        if r.date == dt.isoformat() and r.source_currency == tgt and r.target_currency == src:
            return Decimal("1") / Decimal(r.rate)
    raise ValueError(f"No FX rate for {src}->{tgt} on {dt}")


# =====================================================================
# Stage A – Clean company history
# =====================================================================

def _generate_clean_history(
    rng: _random_module.Random,
    ids: _IDGen,
    base_currency: str,
    foreign_currencies: list[str],
    fx_rates: list[FXRate],
    period_start: date,
    period_end: date,
    tz_offset: str,
    target_docs: int,
    include_distractors: bool,
    policy: Policy,
) -> Tuple[list[Invoice], list[Payment], list[LedgerRow]]:
    """Build clean invoices, payments and balanced ledger entries."""

    invoices: list[Invoice] = []
    payments: list[Payment] = []
    ledger: list[LedgerRow] = []
    entities = [ids.next("ENT") for _ in range(min(8, max(3, target_docs // 8)))]

    days_in_period = (period_end - period_start).days
    if days_in_period < 1:
        days_in_period = 1

    # Determine how many invoices to create (each produces ~4-10 ledger rows)
    num_invoices = max(10, target_docs // 2)
    num_simple_payments = 0

    # Distractor counters
    num_split_payments = max(2, target_docs // 15) if include_distractors else 0
    num_refunds = max(2, target_docs // 20) if include_distractors else 0
    num_recurring = max(2, target_docs // 20) if include_distractors else 0

    # Adjust simple invoices downward to leave room for distractors
    num_invoices = num_invoices - num_split_payments - num_refunds - num_recurring
    num_invoices = max(5, num_invoices)

    def _random_date() -> date:
        offset = rng.randint(0, days_in_period)
        return period_start + timedelta(days=offset)

    def _random_ts(d: date) -> str:
        return _make_ts(d, rng.randint(8, 17), rng.randint(0, 59), tz_offset)

    def _pick_currency() -> str:
        if rng.random() < 0.6 or not foreign_currencies:
            return base_currency
        return rng.choice(foreign_currencies)

    def _make_line_items(currency: str, total_minor: int) -> tuple[LineItem, ...]:
        n = rng.randint(1, 3)
        if n == 1:
            return (LineItem(
                description=rng.choice(_ITEM_DESCRIPTIONS),
                amount_minor=total_minor,
                currency=currency,
            ),)
        parts: list[int] = []
        remaining = total_minor
        for i in range(n - 1):
            part = rng.randint(1, max(1, remaining - (n - i - 1)))
            parts.append(part)
            remaining -= part
        parts.append(remaining)
        return tuple(
            LineItem(
                description=rng.choice(_ITEM_DESCRIPTIONS),
                amount_minor=p,
                currency=currency,
            )
            for p in parts
        )

    def _create_invoice(
        entity_id: str, inv_date: date, currency: str, total_minor: int,
    ) -> Invoice:
        doc_id = ids.next("DOC")
        inv_id = ids.next("INV")
        due_date = inv_date + timedelta(days=rng.choice([15, 30, 45, 60]))
        due_date = min(due_date, period_end + timedelta(days=60))
        inv = Invoice(
            doc_id=doc_id,
            invoice_id=inv_id,
            entity_id=entity_id,
            issued_at=_random_ts(inv_date),
            due_at=_random_ts(due_date),
            currency=currency,
            total_minor=total_minor,
            line_items=_make_line_items(currency, total_minor),
        )
        invoices.append(inv)
        return inv

    def _create_payment(
        entity_id: str, pay_date: date, currency: str, amount_minor: int,
        invoice_refs: tuple[str, ...],
    ) -> Payment:
        doc_id = ids.next("DOC")
        pay_id = ids.next("PAY")
        pay = Payment(
            doc_id=doc_id,
            payment_id=pay_id,
            timestamp=_random_ts(pay_date),
            currency=currency,
            amount_minor=amount_minor,
            invoice_refs=invoice_refs,
        )
        payments.append(pay)
        return pay

    def _add_ledger_pair(
        ts: str, debit_acc: str, credit_acc: str,
        amount_minor: int, currency: str, entity_id: str,
        invoice_id: Optional[str] = None, payment_id: Optional[str] = None,
    ) -> Tuple[LedgerRow, LedgerRow]:
        r1 = LedgerRow(
            row_id=ids.next("ROW"),
            timestamp=ts,
            account=debit_acc,
            debit_minor=amount_minor,
            credit_minor=0,
            currency=currency,
            entity_id=entity_id,
            invoice_id=invoice_id,
            payment_id=payment_id,
        )
        r2 = LedgerRow(
            row_id=ids.next("ROW"),
            timestamp=ts,
            account=credit_acc,
            debit_minor=0,
            credit_minor=amount_minor,
            currency=currency,
            entity_id=entity_id,
            invoice_id=invoice_id,
            payment_id=payment_id,
        )
        ledger.extend([r1, r2])
        return r1, r2

    def _fx_convert(
        amount_minor: int, src_currency: str, tgt_currency: str,
        rate_date: date,
    ) -> int:
        """Convert *amount_minor* from src to tgt using the rate on rate_date."""
        if src_currency == tgt_currency:
            return amount_minor
        rate = _get_rate(fx_rates, rate_date, tgt_currency, src_currency)
        # rate = how many src per 1 tgt => tgt_amount = src_amount / rate
        result = Decimal(amount_minor) / rate
        return int(result.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))

    def _get_fx_date(inv: Invoice, pay_date: date) -> date:
        """Determine which date to use for FX based on policy."""
        if policy.fx_date_rule == "payment_date":
            return pay_date
        # Default: invoice_date
        return date.fromisoformat(inv.issued_at[:10])

    # ----- Simple invoices with single full payments -----
    for _ in range(num_invoices):
        entity_id = rng.choice(entities)
        currency = _pick_currency()
        total = rng.randint(10000, 500000)
        inv_date = _random_date()
        inv = _create_invoice(entity_id, inv_date, currency, total)

        # Ledger: debit accounts_receivable, credit revenue
        _add_ledger_pair(
            inv.issued_at, "accounts_receivable", "revenue",
            total, currency, entity_id, invoice_id=inv.invoice_id,
        )

        # Payment in full
        pay_lag = rng.randint(5, 40)
        pay_date = inv_date + timedelta(days=pay_lag)
        pay_date = min(pay_date, period_end)

        # Payment may be in base_currency even if invoice in foreign
        pay_currency = currency
        pay_amount = total
        fx_date = _get_fx_date(inv, pay_date)

        if currency != base_currency and rng.random() < 0.3:
            pay_currency = base_currency
            pay_amount = _fx_convert(total, currency, base_currency, fx_date)

        pay = _create_payment(
            entity_id, pay_date, pay_currency, pay_amount,
            invoice_refs=(inv.invoice_id,),
        )

        # Ledger: debit cash, credit accounts_receivable
        _add_ledger_pair(
            pay.timestamp, "cash", "accounts_receivable",
            pay_amount, pay_currency, entity_id,
            payment_id=pay.payment_id, invoice_id=inv.invoice_id,
        )

        # If FX conversion, add FX gain/loss entry
        if pay_currency != currency:
            diff = pay_amount - total  # simplification
            if diff != 0:
                if diff > 0:
                    _add_ledger_pair(
                        pay.timestamp, "cash", "fx_gain_loss",
                        abs(diff), base_currency, entity_id,
                        payment_id=pay.payment_id,
                    )
                else:
                    _add_ledger_pair(
                        pay.timestamp, "fx_gain_loss", "cash",
                        abs(diff), base_currency, entity_id,
                        payment_id=pay.payment_id,
                    )

    # ----- DISTRACTOR 1: Split payments (1 invoice → 2-3 payments summing to 100%) -----
    for _ in range(num_split_payments):
        entity_id = rng.choice(entities)
        currency = base_currency  # keep simple for distractors
        total = rng.randint(50000, 200000)
        inv_date = _random_date()
        inv = _create_invoice(entity_id, inv_date, currency, total)

        _add_ledger_pair(
            inv.issued_at, "accounts_receivable", "revenue",
            total, currency, entity_id, invoice_id=inv.invoice_id,
        )

        num_parts = rng.randint(2, 3)
        remaining = total
        for j in range(num_parts):
            if j == num_parts - 1:
                part_amount = remaining
            else:
                part_amount = rng.randint(1, max(1, remaining - (num_parts - j - 1)))
                remaining -= part_amount

            pay_date = inv_date + timedelta(days=(j + 1) * rng.randint(5, 15))
            pay_date = min(pay_date, period_end)
            pay = _create_payment(
                entity_id, pay_date, currency, part_amount,
                invoice_refs=(inv.invoice_id,),
            )
            _add_ledger_pair(
                pay.timestamp, "cash", "accounts_receivable",
                part_amount, currency, entity_id,
                payment_id=pay.payment_id, invoice_id=inv.invoice_id,
            )

    # ----- DISTRACTOR 2: Refunds / credit notes -----
    for _ in range(num_refunds):
        entity_id = rng.choice(entities)
        currency = base_currency
        total = rng.randint(10000, 100000)
        inv_date = _random_date()

        # Original invoice
        inv = _create_invoice(entity_id, inv_date, currency, total)
        _add_ledger_pair(
            inv.issued_at, "accounts_receivable", "revenue",
            total, currency, entity_id, invoice_id=inv.invoice_id,
        )

        # Payment
        pay_date = inv_date + timedelta(days=rng.randint(5, 20))
        pay_date = min(pay_date, period_end)
        pay = _create_payment(
            entity_id, pay_date, currency, total,
            invoice_refs=(inv.invoice_id,),
        )
        _add_ledger_pair(
            pay.timestamp, "cash", "accounts_receivable",
            total, currency, entity_id,
            payment_id=pay.payment_id, invoice_id=inv.invoice_id,
        )

        # Credit note (refund): mirror entry
        refund_date = pay_date + timedelta(days=rng.randint(3, 15))
        refund_date = min(refund_date, period_end)
        refund_amount = rng.randint(1, total)  # partial or full refund
        refund_inv = _create_invoice(
            entity_id, refund_date, currency, -refund_amount,
        )

        # Reverse ledger entries for refund
        _add_ledger_pair(
            refund_inv.issued_at, "revenue", "accounts_receivable",
            refund_amount, currency, entity_id,
            invoice_id=refund_inv.invoice_id,
        )

        refund_pay = _create_payment(
            entity_id, refund_date, currency, -refund_amount,
            invoice_refs=(refund_inv.invoice_id,),
        )
        _add_ledger_pair(
            refund_pay.timestamp, "accounts_receivable", "cash",
            refund_amount, currency, entity_id,
            payment_id=refund_pay.payment_id, invoice_id=refund_inv.invoice_id,
        )

    # ----- DISTRACTOR 3: Recurring payments (same amount, different dates) -----
    for _ in range(num_recurring):
        entity_id = rng.choice(entities)
        currency = base_currency
        monthly_amount = rng.choice([25000, 50000, 75000, 100000])
        desc = rng.choice(["Facility rent", "Software license", "Insurance premium"])

        # Generate 3-6 monthly invoices with identical amounts
        num_months = rng.randint(3, min(6, max(1, days_in_period // 30)))
        for m in range(num_months):
            inv_date = period_start + timedelta(days=30 * m + rng.randint(0, 5))
            if inv_date > period_end:
                break
            inv = Invoice(
                doc_id=ids.next("DOC"),
                invoice_id=ids.next("INV"),
                entity_id=entity_id,
                issued_at=_random_ts(inv_date),
                due_at=_random_ts(inv_date + timedelta(days=15)),
                currency=currency,
                total_minor=monthly_amount,
                line_items=(LineItem(
                    description=desc,
                    amount_minor=monthly_amount,
                    currency=currency,
                ),),
            )
            invoices.append(inv)

            _add_ledger_pair(
                inv.issued_at, "accounts_receivable", "revenue",
                monthly_amount, currency, entity_id,
                invoice_id=inv.invoice_id,
            )

            pay_date = inv_date + timedelta(days=rng.randint(5, 15))
            pay_date = min(pay_date, period_end)
            pay = _create_payment(
                entity_id, pay_date, currency, monthly_amount,
                invoice_refs=(inv.invoice_id,),
            )
            _add_ledger_pair(
                pay.timestamp, "cash", "accounts_receivable",
                monthly_amount, currency, entity_id,
                payment_id=pay.payment_id, invoice_id=inv.invoice_id,
            )

    return invoices, payments, ledger


# =====================================================================
# Stage B – Error injection
# =====================================================================

def _inject_duplicate_invoice(
    rng: _random_module.Random,
    ids: _IDGen,
    invoices: list[Invoice],
    ledger: list[LedgerRow],
    tz_offset: str,
    provenance: list[ProvenanceEntry],
    truth: list[TruthIssue],
    step: int,
) -> None:
    """Inject a duplicate invoice error."""
    # Pick a source invoice to duplicate
    candidates = [inv for inv in invoices if inv.total_minor > 0]
    if not candidates:
        return
    source = rng.choice(candidates)

    # Create duplicate with new IDs but same entity, amount, line_items
    dup_doc_id = ids.next("DOC")
    dup_inv_id = ids.next("INV")

    # Shift date by 1-3 days
    src_date = date.fromisoformat(source.issued_at[:10])
    dup_date = src_date + timedelta(days=rng.randint(1, 3))
    dup_ts = _make_ts(dup_date, rng.randint(8, 17), rng.randint(0, 59), tz_offset)
    due_date = dup_date + timedelta(days=30)

    dup_inv = Invoice(
        doc_id=dup_doc_id,
        invoice_id=dup_inv_id,
        entity_id=source.entity_id,
        issued_at=dup_ts,
        due_at=_make_ts(due_date, 17, 0, tz_offset),
        currency=source.currency,
        total_minor=source.total_minor,
        line_items=source.line_items,  # same line items
    )
    invoices.append(dup_inv)

    # Duplicate ledger entries
    r1 = LedgerRow(
        row_id=ids.next("ROW"),
        timestamp=dup_ts,
        account="accounts_receivable",
        debit_minor=source.total_minor,
        credit_minor=0,
        currency=source.currency,
        entity_id=source.entity_id,
        invoice_id=dup_inv_id,
    )
    r2 = LedgerRow(
        row_id=ids.next("ROW"),
        timestamp=dup_ts,
        account="revenue",
        debit_minor=0,
        credit_minor=source.total_minor,
        currency=source.currency,
        entity_id=source.entity_id,
        invoice_id=dup_inv_id,
    )
    ledger.extend([r1, r2])

    issue_id = ids.next("ISS")
    truth.append(TruthIssue(
        issue_id=issue_id,
        type="duplicate_invoice",
        affected_ids=(source.invoice_id, dup_inv_id),
        expected_amount_minor=source.total_minor,
        currency=source.currency,
        acceptable_evidence_sets=(
            (source.doc_id, dup_doc_id),
            (source.doc_id, dup_doc_id, r1.row_id, r2.row_id),
        ),
    ))
    provenance.append(ProvenanceEntry(
        step=step,
        action="inject_duplicate_invoice",
        target_ids=(source.invoice_id, dup_inv_id),
        description=f"Duplicated {source.invoice_id} as {dup_inv_id}",
    ))


def _inject_unallocated_payment(
    rng: _random_module.Random,
    ids: _IDGen,
    invoices: list[Invoice],
    payments: list[Payment],
    ledger: list[LedgerRow],
    tz_offset: str,
    period_end: date,
    provenance: list[ProvenanceEntry],
    truth: list[TruthIssue],
    step: int,
) -> None:
    """Inject an unallocated or partial payment error."""
    candidates = [inv for inv in invoices if inv.total_minor > 10000]
    if not candidates:
        return
    target_inv = rng.choice(candidates)

    # Partial payment: pay only 40-80% of the invoice
    pct = rng.randint(40, 80)
    partial_amount = int(Decimal(target_inv.total_minor) * Decimal(pct) / Decimal(100))
    shortfall = target_inv.total_minor - partial_amount

    pay_date = date.fromisoformat(target_inv.issued_at[:10]) + timedelta(days=rng.randint(5, 30))
    pay_date = min(pay_date, period_end)
    pay_ts = _make_ts(pay_date, rng.randint(8, 17), rng.randint(0, 59), tz_offset)

    pay_doc_id = ids.next("DOC")
    pay_id = ids.next("PAY")
    pay = Payment(
        doc_id=pay_doc_id,
        payment_id=pay_id,
        timestamp=pay_ts,
        currency=target_inv.currency,
        amount_minor=partial_amount,
        invoice_refs=(target_inv.invoice_id,),
    )
    payments.append(pay)

    r1 = LedgerRow(
        row_id=ids.next("ROW"),
        timestamp=pay_ts,
        account="cash",
        debit_minor=partial_amount,
        credit_minor=0,
        currency=target_inv.currency,
        entity_id=target_inv.entity_id,
        invoice_id=target_inv.invoice_id,
        payment_id=pay_id,
    )
    r2 = LedgerRow(
        row_id=ids.next("ROW"),
        timestamp=pay_ts,
        account="accounts_receivable",
        debit_minor=0,
        credit_minor=partial_amount,
        currency=target_inv.currency,
        entity_id=target_inv.entity_id,
        invoice_id=target_inv.invoice_id,
        payment_id=pay_id,
    )
    ledger.extend([r1, r2])

    issue_id = ids.next("ISS")
    truth.append(TruthIssue(
        issue_id=issue_id,
        type="unallocated_or_partial_payment",
        affected_ids=(target_inv.invoice_id, pay_id),
        expected_amount_minor=shortfall,
        currency=target_inv.currency,
        acceptable_evidence_sets=(
            (target_inv.doc_id, pay_doc_id),
            (target_inv.doc_id, pay_doc_id, r1.row_id, r2.row_id),
        ),
    ))
    provenance.append(ProvenanceEntry(
        step=step,
        action="inject_unallocated_or_partial_payment",
        target_ids=(target_inv.invoice_id, pay_id),
        description=(
            f"Partial payment {pay_id} of {partial_amount} for "
            f"invoice {target_inv.invoice_id} (total {target_inv.total_minor}), "
            f"shortfall {shortfall}"
        ),
    ))


def _inject_wrong_fx(
    rng: _random_module.Random,
    ids: _IDGen,
    invoices: list[Invoice],
    payments: list[Payment],
    ledger: list[LedgerRow],
    fx_rates: list[FXRate],
    base_currency: str,
    foreign_currencies: list[str],
    tz_offset: str,
    period_start: date,
    period_end: date,
    policy: Policy,
    provenance: list[ProvenanceEntry],
    truth: list[TruthIssue],
    step: int,
) -> None:
    """Inject a wrong FX date/rate error."""
    if not foreign_currencies:
        return

    # Pick a foreign-currency invoice
    fc_invoices = [inv for inv in invoices
                   if inv.currency != base_currency and inv.total_minor > 0]
    if not fc_invoices:
        # Create one
        fc = rng.choice(foreign_currencies)
        entity_id = invoices[0].entity_id if invoices else ids.next("ENT")
        inv_date = period_start + timedelta(days=rng.randint(10, max(10, (period_end - period_start).days - 10)))
        inv_ts = _make_ts(inv_date, 10, 0, tz_offset)
        doc_id = ids.next("DOC")
        inv_id = ids.next("INV")
        total = rng.randint(50000, 200000)
        inv = Invoice(
            doc_id=doc_id, invoice_id=inv_id, entity_id=entity_id,
            issued_at=inv_ts,
            due_at=_make_ts(inv_date + timedelta(days=30), 17, 0, tz_offset),
            currency=fc, total_minor=total,
            line_items=(LineItem(description="Imported goods", amount_minor=total, currency=fc),),
        )
        invoices.append(inv)
        # Add ledger for original invoice
        ledger.append(LedgerRow(
            row_id=ids.next("ROW"), timestamp=inv_ts,
            account="accounts_receivable", debit_minor=total, credit_minor=0,
            currency=fc, entity_id=entity_id, invoice_id=inv_id,
        ))
        ledger.append(LedgerRow(
            row_id=ids.next("ROW"), timestamp=inv_ts,
            account="revenue", debit_minor=0, credit_minor=total,
            currency=fc, entity_id=entity_id, invoice_id=inv_id,
        ))
        fc_invoices = [inv]

    source_inv = rng.choice(fc_invoices)
    inv_date = date.fromisoformat(source_inv.issued_at[:10])
    pay_date = inv_date + timedelta(days=rng.randint(5, 25))
    pay_date = min(pay_date, period_end)

    # Correct FX date per policy
    correct_fx_date = inv_date if policy.fx_date_rule == "invoice_date" else pay_date

    # Wrong FX date: use a date that is off by several days
    wrong_offset = rng.choice([-5, -3, -2, 2, 3, 5])
    wrong_fx_date = correct_fx_date + timedelta(days=wrong_offset)
    # Clamp to available dates
    if wrong_fx_date < period_start:
        wrong_fx_date = period_start
    if wrong_fx_date > period_end:
        wrong_fx_date = period_end

    # Compute amounts
    correct_rate = _get_rate(fx_rates, correct_fx_date, base_currency, source_inv.currency)
    wrong_rate = _get_rate(fx_rates, wrong_fx_date, base_currency, source_inv.currency)

    correct_base_amount = int(
        (Decimal(source_inv.total_minor) * correct_rate).quantize(
            Decimal("1"), rounding=ROUND_HALF_EVEN
        )
    )
    wrong_base_amount = int(
        (Decimal(source_inv.total_minor) * wrong_rate).quantize(
            Decimal("1"), rounding=ROUND_HALF_EVEN
        )
    )

    discrepancy = abs(wrong_base_amount - correct_base_amount)
    if discrepancy == 0:
        # Force a discrepancy by tweaking the rate slightly
        wrong_base_amount += rng.randint(100, 500)
        discrepancy = abs(wrong_base_amount - correct_base_amount)

    # Create the payment using the WRONG converted amount
    pay_ts = _make_ts(pay_date, rng.randint(8, 17), rng.randint(0, 59), tz_offset)
    pay_doc_id = ids.next("DOC")
    pay_id = ids.next("PAY")
    pay = Payment(
        doc_id=pay_doc_id, payment_id=pay_id,
        timestamp=pay_ts, currency=base_currency,
        amount_minor=wrong_base_amount,
        invoice_refs=(source_inv.invoice_id,),
    )
    payments.append(pay)

    # Ledger entries with wrong FX amount
    r1 = LedgerRow(
        row_id=ids.next("ROW"), timestamp=pay_ts,
        account="cash", debit_minor=wrong_base_amount, credit_minor=0,
        currency=base_currency, entity_id=source_inv.entity_id,
        invoice_id=source_inv.invoice_id, payment_id=pay_id,
    )
    r2 = LedgerRow(
        row_id=ids.next("ROW"), timestamp=pay_ts,
        account="accounts_receivable", debit_minor=0, credit_minor=wrong_base_amount,
        currency=base_currency, entity_id=source_inv.entity_id,
        invoice_id=source_inv.invoice_id, payment_id=pay_id,
    )
    ledger.extend([r1, r2])

    # Find FX rate rows for evidence
    correct_fx_row_date = correct_fx_date.isoformat()
    wrong_fx_row_date = wrong_fx_date.isoformat()

    issue_id = ids.next("ISS")
    truth.append(TruthIssue(
        issue_id=issue_id,
        type="wrong_fx_date_or_rate",
        affected_ids=(source_inv.invoice_id, pay_id),
        expected_amount_minor=discrepancy,
        currency=base_currency,
        acceptable_evidence_sets=(
            (source_inv.doc_id, pay_doc_id),
            (source_inv.doc_id, pay_doc_id, r1.row_id, r2.row_id),
        ),
    ))
    provenance.append(ProvenanceEntry(
        step=step,
        action="inject_wrong_fx_date_or_rate",
        target_ids=(source_inv.invoice_id, pay_id),
        description=(
            f"Used FX rate from {wrong_fx_date} instead of {correct_fx_date} "
            f"for {source_inv.invoice_id}, discrepancy={discrepancy} {base_currency}"
        ),
    ))


def _inject_cutoff_error(
    rng: _random_module.Random,
    ids: _IDGen,
    invoices: list[Invoice],
    ledger: list[LedgerRow],
    tz_offset: str,
    period_start: date,
    period_end: date,
    provenance: list[ProvenanceEntry],
    truth: list[TruthIssue],
    step: int,
) -> None:
    """Inject a cutoff error — out-of-period transaction included in ledger."""
    entity_id = invoices[0].entity_id if invoices else ids.next("ENT")
    currency = invoices[0].currency if invoices else "USD"

    # Create an invoice dated AFTER period_end but recorded in the ledger
    out_of_period_date = period_end + timedelta(days=rng.randint(1, 10))
    out_ts = _make_ts(out_of_period_date, rng.randint(8, 17), rng.randint(0, 59), tz_offset)

    total = rng.randint(20000, 150000)
    doc_id = ids.next("DOC")
    inv_id = ids.next("INV")

    inv = Invoice(
        doc_id=doc_id, invoice_id=inv_id, entity_id=entity_id,
        issued_at=out_ts,
        due_at=_make_ts(out_of_period_date + timedelta(days=30), 17, 0, tz_offset),
        currency=currency, total_minor=total,
        line_items=(LineItem(description="Next period service", amount_minor=total, currency=currency),),
    )
    invoices.append(inv)

    # Ledger entries — the cutoff error is that these appear in the period ledger
    # but the invoice date is outside the period
    r1 = LedgerRow(
        row_id=ids.next("ROW"), timestamp=out_ts,
        account="accounts_receivable", debit_minor=total, credit_minor=0,
        currency=currency, entity_id=entity_id, invoice_id=inv_id,
    )
    r2 = LedgerRow(
        row_id=ids.next("ROW"), timestamp=out_ts,
        account="revenue", debit_minor=0, credit_minor=total,
        currency=currency, entity_id=entity_id, invoice_id=inv_id,
    )
    ledger.extend([r1, r2])

    issue_id = ids.next("ISS")
    truth.append(TruthIssue(
        issue_id=issue_id,
        type="cutoff_error",
        affected_ids=(inv_id,),
        expected_amount_minor=total,
        currency=currency,
        acceptable_evidence_sets=(
            (doc_id,),
            (doc_id, r1.row_id, r2.row_id),
        ),
    ))
    provenance.append(ProvenanceEntry(
        step=step,
        action="inject_cutoff_error",
        target_ids=(inv_id,),
        description=(
            f"Invoice {inv_id} dated {out_of_period_date} (after period_end {period_end}) "
            f"included in period ledger"
        ),
    ))


# ----- Compound scenarios -----

def _inject_compound_fx_cutoff(
    rng: _random_module.Random,
    ids: _IDGen,
    invoices: list[Invoice],
    payments: list[Payment],
    ledger: list[LedgerRow],
    fx_rates: list[FXRate],
    base_currency: str,
    foreign_currencies: list[str],
    tz_offset: str,
    period_start: date,
    period_end: date,
    policy: Policy,
    provenance: list[ProvenanceEntry],
    truth: list[TruthIssue],
    step: int,
) -> None:
    """Compound scenario 1: partial payment in foreign currency at period boundary
    with wrong FX date due to timezone confusion."""
    if not foreign_currencies:
        return

    fc = rng.choice(foreign_currencies)
    entity_id = ids.next("ENT")

    # Invoice near end of period
    inv_date = period_end - timedelta(days=rng.randint(2, 5))
    inv_ts = _make_ts(inv_date, 23, 30, tz_offset)  # late evening
    total = rng.randint(80000, 300000)
    doc_id = ids.next("DOC")
    inv_id = ids.next("INV")
    inv = Invoice(
        doc_id=doc_id, invoice_id=inv_id, entity_id=entity_id,
        issued_at=inv_ts,
        due_at=_make_ts(inv_date + timedelta(days=30), 17, 0, tz_offset),
        currency=fc, total_minor=total,
        line_items=(LineItem(description="Cross-border service", amount_minor=total, currency=fc),),
    )
    invoices.append(inv)
    ledger.append(LedgerRow(
        row_id=ids.next("ROW"), timestamp=inv_ts,
        account="accounts_receivable", debit_minor=total, credit_minor=0,
        currency=fc, entity_id=entity_id, invoice_id=inv_id,
    ))
    ledger.append(LedgerRow(
        row_id=ids.next("ROW"), timestamp=inv_ts,
        account="revenue", debit_minor=0, credit_minor=total,
        currency=fc, entity_id=entity_id, invoice_id=inv_id,
    ))

    # Partial payment at period boundary (next day = after period in some TZ)
    pay_date = period_end  # exactly at boundary
    pay_ts = _make_ts(pay_date, 23, 59, tz_offset)  # very last minute
    partial_pct = rng.randint(50, 80)
    partial_fc_amount = int(Decimal(total) * Decimal(partial_pct) / Decimal(100))
    shortfall = total - partial_fc_amount

    # Use WRONG fx date (next day, which is after period)
    correct_fx_date = inv_date if policy.fx_date_rule == "invoice_date" else pay_date
    wrong_fx_date = correct_fx_date + timedelta(days=1)
    if wrong_fx_date > period_end:
        wrong_fx_date = period_end  # clamp to available rates
    wrong_rate = _get_rate(fx_rates, wrong_fx_date, base_currency, fc)
    correct_rate = _get_rate(fx_rates, correct_fx_date, base_currency, fc)
    wrong_base_amount = int(
        (Decimal(partial_fc_amount) * wrong_rate).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
    )
    correct_base_amount = int(
        (Decimal(partial_fc_amount) * correct_rate).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
    )
    fx_discrepancy = abs(wrong_base_amount - correct_base_amount)

    pay_doc_id = ids.next("DOC")
    pay_id = ids.next("PAY")
    pay = Payment(
        doc_id=pay_doc_id, payment_id=pay_id,
        timestamp=pay_ts, currency=base_currency,
        amount_minor=wrong_base_amount,
        invoice_refs=(inv_id,),
    )
    payments.append(pay)

    r1 = LedgerRow(
        row_id=ids.next("ROW"), timestamp=pay_ts,
        account="cash", debit_minor=wrong_base_amount, credit_minor=0,
        currency=base_currency, entity_id=entity_id,
        invoice_id=inv_id, payment_id=pay_id,
    )
    r2 = LedgerRow(
        row_id=ids.next("ROW"), timestamp=pay_ts,
        account="accounts_receivable", debit_minor=0, credit_minor=wrong_base_amount,
        currency=base_currency, entity_id=entity_id,
        invoice_id=inv_id, payment_id=pay_id,
    )
    ledger.extend([r1, r2])

    # This creates TWO truth issues from one compound scenario:
    # 1) Partial payment shortfall
    issue_id_1 = ids.next("ISS")
    truth.append(TruthIssue(
        issue_id=issue_id_1,
        type="unallocated_or_partial_payment",
        affected_ids=(inv_id, pay_id),
        expected_amount_minor=shortfall,
        currency=fc,
        acceptable_evidence_sets=(
            (doc_id, pay_doc_id),
            (doc_id, pay_doc_id, r1.row_id, r2.row_id),
        ),
    ))

    # 2) Wrong FX rate (if discrepancy > 0)
    if fx_discrepancy > 0:
        issue_id_2 = ids.next("ISS")
        truth.append(TruthIssue(
            issue_id=issue_id_2,
            type="wrong_fx_date_or_rate",
            affected_ids=(inv_id, pay_id),
            expected_amount_minor=fx_discrepancy,
            currency=base_currency,
            acceptable_evidence_sets=(
                (doc_id, pay_doc_id),
                (doc_id, pay_doc_id, r1.row_id, r2.row_id),
            ),
        ))

    provenance.append(ProvenanceEntry(
        step=step,
        action="inject_compound_fx_cutoff",
        target_ids=(inv_id, pay_id),
        description=(
            f"Compound: partial payment ({partial_pct}%) in {fc} at period boundary "
            f"with wrong FX date ({wrong_fx_date} instead of {correct_fx_date})"
        ),
    ))


def _inject_compound_duplicate_split(
    rng: _random_module.Random,
    ids: _IDGen,
    invoices: list[Invoice],
    payments: list[Payment],
    ledger: list[LedgerRow],
    tz_offset: str,
    period_start: date,
    period_end: date,
    provenance: list[ProvenanceEntry],
    truth: list[TruthIssue],
    step: int,
) -> None:
    """Compound scenario 2: duplicate invoice partially covered by split payment
    so that entity-level sum reconciliation hides the duplication."""
    entity_id = ids.next("ENT")
    currency = "USD"
    total = rng.randint(60000, 200000)
    inv_date = period_start + timedelta(days=rng.randint(5, max(5, (period_end - period_start).days // 2)))

    # Original invoice
    orig_doc_id = ids.next("DOC")
    orig_inv_id = ids.next("INV")
    inv_ts = _make_ts(inv_date, 10, 0, tz_offset)
    orig_inv = Invoice(
        doc_id=orig_doc_id, invoice_id=orig_inv_id, entity_id=entity_id,
        issued_at=inv_ts,
        due_at=_make_ts(inv_date + timedelta(days=30), 17, 0, tz_offset),
        currency=currency, total_minor=total,
        line_items=(LineItem(description="Consulting services", amount_minor=total, currency=currency),),
    )
    invoices.append(orig_inv)
    ledger.append(LedgerRow(
        row_id=ids.next("ROW"), timestamp=inv_ts,
        account="accounts_receivable", debit_minor=total, credit_minor=0,
        currency=currency, entity_id=entity_id, invoice_id=orig_inv_id,
    ))
    ledger.append(LedgerRow(
        row_id=ids.next("ROW"), timestamp=inv_ts,
        account="revenue", debit_minor=0, credit_minor=total,
        currency=currency, entity_id=entity_id, invoice_id=orig_inv_id,
    ))

    # Duplicate invoice (new IDs, same details, slight date shift)
    dup_doc_id = ids.next("DOC")
    dup_inv_id = ids.next("INV")
    dup_date = inv_date + timedelta(days=rng.randint(1, 3))
    dup_ts = _make_ts(dup_date, 11, 0, tz_offset)
    dup_inv = Invoice(
        doc_id=dup_doc_id, invoice_id=dup_inv_id, entity_id=entity_id,
        issued_at=dup_ts,
        due_at=_make_ts(dup_date + timedelta(days=30), 17, 0, tz_offset),
        currency=currency, total_minor=total,
        line_items=orig_inv.line_items,
    )
    invoices.append(dup_inv)
    dup_r1 = LedgerRow(
        row_id=ids.next("ROW"), timestamp=dup_ts,
        account="accounts_receivable", debit_minor=total, credit_minor=0,
        currency=currency, entity_id=entity_id, invoice_id=dup_inv_id,
    )
    dup_r2 = LedgerRow(
        row_id=ids.next("ROW"), timestamp=dup_ts,
        account="revenue", debit_minor=0, credit_minor=total,
        currency=currency, entity_id=entity_id, invoice_id=dup_inv_id,
    )
    ledger.extend([dup_r1, dup_r2])

    # Split payments: pay the original in 2 parts (correctly),
    # and partially pay the duplicate (this hides the duplication in sum checks)
    part1_amount = int(Decimal(total) * Decimal("0.6"))
    part2_amount = total - part1_amount

    for i, (amount, inv_id, doc_ref) in enumerate([
        (part1_amount, orig_inv_id, orig_doc_id),
        (part2_amount, orig_inv_id, orig_doc_id),
    ]):
        pay_date = inv_date + timedelta(days=10 + i * 10)
        pay_date = min(pay_date, period_end)
        pay_ts = _make_ts(pay_date, 14, 0, tz_offset)
        p_doc = ids.next("DOC")
        p_id = ids.next("PAY")
        pay = Payment(
            doc_id=p_doc, payment_id=p_id,
            timestamp=pay_ts, currency=currency, amount_minor=amount,
            invoice_refs=(inv_id,),
        )
        payments.append(pay)
        ledger.append(LedgerRow(
            row_id=ids.next("ROW"), timestamp=pay_ts,
            account="cash", debit_minor=amount, credit_minor=0,
            currency=currency, entity_id=entity_id,
            invoice_id=inv_id, payment_id=p_id,
        ))
        ledger.append(LedgerRow(
            row_id=ids.next("ROW"), timestamp=pay_ts,
            account="accounts_receivable", debit_minor=0, credit_minor=amount,
            currency=currency, entity_id=entity_id,
            invoice_id=inv_id, payment_id=p_id,
        ))

    # Truth: the duplicate invoice
    issue_id = ids.next("ISS")
    truth.append(TruthIssue(
        issue_id=issue_id,
        type="duplicate_invoice",
        affected_ids=(orig_inv_id, dup_inv_id),
        expected_amount_minor=total,
        currency=currency,
        acceptable_evidence_sets=(
            (orig_doc_id, dup_doc_id),
            (orig_doc_id, dup_doc_id, dup_r1.row_id, dup_r2.row_id),
        ),
    ))
    provenance.append(ProvenanceEntry(
        step=step,
        action="inject_compound_duplicate_split",
        target_ids=(orig_inv_id, dup_inv_id),
        description=(
            f"Compound: duplicate invoice {dup_inv_id} of {orig_inv_id} "
            f"partially hidden by split payment pattern on original"
        ),
    ))


# =====================================================================
# Main generator
# =====================================================================

_DEFAULT_CONFIG: dict[str, Any] = {
    "num_errors": 4,
    "target_ledger_rows": 400,
    "target_docs": 60,
    "include_compound": True,
    "include_distractors": True,
    "episode_id": None,       # auto-generated from seed
    "timezone": "Asia/Almaty",
    "base_currency": None,    # random if None
    "num_foreign_currencies": 2,
    "period_months": 6,
}


def generate_episode(
    seed: int,
    config: dict[str, Any] | None = None,
) -> EpisodeData:
    """Generate a complete deterministic episode.

    Parameters
    ----------
    seed : int
        Controls all randomness.  Same seed + config → identical output.
    config : dict, optional
        Override keys from ``_DEFAULT_CONFIG``.

    Returns
    -------
    EpisodeData
        All open and closed data for the episode.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    rng = _random_module.Random(seed)
    ids = _IDGen()

    # ── Parameters ───────────────────────────────────────────────────
    num_errors: int = max(0, min(12, cfg["num_errors"]))
    target_docs: int = max(20, min(200, cfg["target_docs"]))
    tz_name: str = cfg["timezone"]
    tz_offset = _tz_offset(tz_name)

    base_currency: str = cfg["base_currency"] or rng.choice(["USD", "EUR", "GBP"])
    n_fc: int = cfg["num_foreign_currencies"]
    available_fc = [c for c in _CURRENCIES if c != base_currency]
    rng.shuffle(available_fc)
    foreign_currencies = available_fc[:n_fc]

    episode_id: str = cfg["episode_id"] or f"ep-{seed:06d}"

    period_months: int = cfg["period_months"]
    year = 2026
    period_start = date(year, 1, 1)
    period_end = date(year, min(12, period_months), 28)

    # ── Policy ───────────────────────────────────────────────────────
    fx_date_rule = rng.choice(["invoice_date", "payment_date"])
    policy = Policy(
        fx_date_rule=fx_date_rule,
        period_boundary="inclusive_start_exclusive_end",
        rounding="ROUND_HALF_EVEN",
        amount_tolerance_minor=100,
    )

    # ── FX rates ─────────────────────────────────────────────────────
    # Extend rates a bit beyond period for cutoff scenarios
    fx_start = period_start - timedelta(days=5)
    fx_end = period_end + timedelta(days=15)
    fx_rates = _generate_fx_rates(rng, base_currency, foreign_currencies,
                                  fx_start, fx_end)

    # ── Stage A: clean history ───────────────────────────────────────
    invoices, payments, ledger = _generate_clean_history(
        rng, ids, base_currency, foreign_currencies, fx_rates,
        period_start, period_end, tz_offset, target_docs,
        cfg["include_distractors"], policy,
    )

    # ── Stage B: inject errors ───────────────────────────────────────
    truth_issues: list[TruthIssue] = []
    provenance_log: list[ProvenanceEntry] = []

    if num_errors > 0:
        # Build a plan of which error types to inject
        error_types = ["duplicate_invoice", "unallocated_or_partial_payment",
                       "wrong_fx_date_or_rate", "cutoff_error"]

        # Reserve slots for compound scenarios if requested
        compound_count = 0
        remaining_errors = num_errors
        if cfg["include_compound"] and remaining_errors >= 4:
            compound_count = min(2, remaining_errors // 2)
            remaining_errors -= compound_count * 2  # each compound produces ~2 issues

        # Distribute remaining errors across the 4 types
        error_plan: list[str] = []
        for i in range(remaining_errors):
            error_plan.append(error_types[i % len(error_types)])
        rng.shuffle(error_plan)

        step_counter = 0

        # Inject compound scenarios first
        if compound_count >= 1:
            step_counter += 1
            _inject_compound_fx_cutoff(
                rng, ids, invoices, payments, ledger, fx_rates,
                base_currency, foreign_currencies, tz_offset,
                period_start, period_end, policy,
                provenance_log, truth_issues, step_counter,
            )
        if compound_count >= 2:
            step_counter += 1
            _inject_compound_duplicate_split(
                rng, ids, invoices, payments, ledger, tz_offset,
                period_start, period_end,
                provenance_log, truth_issues, step_counter,
            )

        # Inject individual errors
        for error_type in error_plan:
            step_counter += 1
            if error_type == "duplicate_invoice":
                _inject_duplicate_invoice(
                    rng, ids, invoices, ledger, tz_offset,
                    provenance_log, truth_issues, step_counter,
                )
            elif error_type == "unallocated_or_partial_payment":
                _inject_unallocated_payment(
                    rng, ids, invoices, payments, ledger, tz_offset,
                    period_end, provenance_log, truth_issues, step_counter,
                )
            elif error_type == "wrong_fx_date_or_rate":
                _inject_wrong_fx(
                    rng, ids, invoices, payments, ledger, fx_rates,
                    base_currency, foreign_currencies, tz_offset,
                    period_start, period_end, policy,
                    provenance_log, truth_issues, step_counter,
                )
            elif error_type == "cutoff_error":
                _inject_cutoff_error(
                    rng, ids, invoices, ledger, tz_offset,
                    period_start, period_end,
                    provenance_log, truth_issues, step_counter,
                )

    # ── Shuffle records so injected errors are NOT clustered at the end ──
    rng.shuffle(ledger)
    rng.shuffle(invoices)
    rng.shuffle(payments)

    # ── Manifest ─────────────────────────────────────────────────────
    manifest = Manifest(
        schema_version=_SCHEMA_VERSION,
        episode_id=episode_id,
        base_currency=base_currency,
        period_start=_make_ts(period_start, 0, 0, tz_offset),
        period_end=_make_ts(period_end, 23, 59, tz_offset),
        timezone=tz_name,
        num_records=len(ledger),
        num_documents=len(invoices) + len(payments),
        rounding_rule="ROUND_HALF_EVEN",
    )

    return EpisodeData(
        manifest=manifest,
        policy=policy,
        ledger=ledger,
        invoices=invoices,
        payments=payments,
        fx_rates=fx_rates,
        truth_issues=truth_issues,
        provenance=provenance_log,
    )


# =====================================================================
# Save / Load
# =====================================================================

def save_episode(episode: EpisodeData, directory: str | Path) -> None:
    """Save episode to *directory*, separating open and closed files.

    Open files (agent-visible):
        manifest.json, policy.json, ledger.csv, invoices.jsonl,
        payments.jsonl, fx.csv

    Closed files (evaluator-only, in ``closed/`` subdirectory):
        truth.json, provenance.json
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    closed = d / "closed"
    closed.mkdir(exist_ok=True)

    # manifest.json
    with open(d / "manifest.json", "w") as f:
        json.dump(episode.manifest.to_dict(), f, indent=2)

    # policy.json
    with open(d / "policy.json", "w") as f:
        json.dump(episode.policy.to_dict(), f, indent=2)

    # ledger.csv
    fieldnames = [
        "row_id", "timestamp", "account", "debit_minor", "credit_minor",
        "currency", "entity_id", "invoice_id", "payment_id",
    ]
    with open(d / "ledger.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode.ledger:
            rd = row.to_dict()
            # Replace None with empty string for CSV
            rd = {k: ("" if v is None else v) for k, v in rd.items()}
            writer.writerow(rd)

    # invoices.jsonl
    with open(d / "invoices.jsonl", "w") as f:
        for inv in episode.invoices:
            f.write(json.dumps(inv.to_dict()) + "\n")

    # payments.jsonl
    with open(d / "payments.jsonl", "w") as f:
        for pay in episode.payments:
            f.write(json.dumps(pay.to_dict()) + "\n")

    # fx.csv
    with open(d / "fx.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "source_currency", "target_currency", "rate"])
        writer.writeheader()
        for rate in episode.fx_rates:
            writer.writerow(rate.to_dict())

    # Closed: truth.json
    with open(closed / "truth.json", "w") as f:
        json.dump([t.to_dict() for t in episode.truth_issues], f, indent=2)

    # Closed: provenance.json
    with open(closed / "provenance.json", "w") as f:
        json.dump([p.to_dict() for p in episode.provenance], f, indent=2)


def load_episode(directory: str | Path) -> EpisodeData:
    """Load an episode from *directory* (inverse of ``save_episode``)."""
    d = Path(directory)
    closed = d / "closed"

    # manifest
    with open(d / "manifest.json") as f:
        manifest = Manifest.from_dict(json.load(f))

    # policy
    with open(d / "policy.json") as f:
        policy = Policy.from_dict(json.load(f))

    # ledger
    ledger: list[LedgerRow] = []
    with open(d / "ledger.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["debit_minor"] = int(row["debit_minor"])
            row["credit_minor"] = int(row["credit_minor"])
            row["invoice_id"] = row["invoice_id"] if row["invoice_id"] else None
            row["payment_id"] = row["payment_id"] if row["payment_id"] else None
            ledger.append(LedgerRow.from_dict(row))

    # invoices
    invoices: list[Invoice] = []
    with open(d / "invoices.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                invoices.append(Invoice.from_dict(json.loads(line)))

    # payments
    payments_list: list[Payment] = []
    with open(d / "payments.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                payments_list.append(Payment.from_dict(json.loads(line)))

    # fx
    fx_rates: list[FXRate] = []
    with open(d / "fx.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fx_rates.append(FXRate.from_dict(row))

    # truth
    truth_issues: list[TruthIssue] = []
    truth_path = closed / "truth.json"
    if truth_path.exists():
        with open(truth_path) as f:
            for item in json.load(f):
                truth_issues.append(TruthIssue.from_dict(item))

    # provenance
    provenance_log: list[ProvenanceEntry] = []
    prov_path = closed / "provenance.json"
    if prov_path.exists():
        with open(prov_path) as f:
            for item in json.load(f):
                provenance_log.append(ProvenanceEntry.from_dict(item))

    return EpisodeData(
        manifest=manifest,
        policy=policy,
        ledger=ledger,
        invoices=invoices,
        payments=payments_list,
        fx_rates=fx_rates,
        truth_issues=truth_issues,
        provenance=provenance_log,
    )
