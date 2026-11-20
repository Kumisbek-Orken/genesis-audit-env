"""
Baseline policies for the GENESIS RL-environment (SPEC §§08-09).

Provides four policies and a unified ``run_episode`` driver:

Policy classes
--------------
RandomPolicy(seed)         – random exploration + random/empty submit.
WeakHeuristicPolicy()      – shallow first-page heuristic (errors on distractors).
StrongBaselinePolicy()     – full rule-based deterministic auditor (all 4 error types,
                             correct distractor handling, ≤100 steps).
OraclePolicy()             – upper-bound benchmark: reads truth_issues directly
                             and achieves F1 == 1.0.

Unified interface
-----------------
run_episode(env, policy, *, seed, config, episode_data) -> Trajectory

All policies are callable:  policy(obs, info, env) -> dict | None
Returning None causes run_episode to submit empty findings immediately.
"""

from __future__ import annotations

import random as _random_module
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Dict, List, Optional, Protocol

from audit_env.env import AuditEnv
from audit_env.generator import EpisodeData
from audit_env.schemas import VALID_ISSUE_TYPES


# =====================================================================
# Trajectory container
# =====================================================================

@dataclass
class StepRecord:
    """One recorded step in a trajectory."""
    step: int
    action: dict
    observation: dict
    reward: float
    terminated: bool
    truncated: bool


@dataclass
class Trajectory:
    """Full episode trajectory (no closed truth data in observations)."""
    episode_id: str
    steps: List[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0
    final_obs: dict = field(default_factory=dict)
    final_info: dict = field(default_factory=dict)


# =====================================================================
# Unified run_episode driver
# =====================================================================

class _PolicyProtocol(Protocol):
    def __call__(self, obs: dict, info: dict, env: AuditEnv) -> Optional[dict]: ...


def run_episode(
    env: AuditEnv,
    policy: _PolicyProtocol,
    *,
    seed: Optional[int] = None,
    config: Optional[dict] = None,
    episode_data: Optional[EpisodeData] = None,
) -> Trajectory:
    """Drive a full episode with *policy* and return the trajectory.

    Parameters
    ----------
    env : AuditEnv
        The environment (eval_mode may be True or False).
    policy : callable
        ``policy(obs, info, env) -> action_dict | None``.
        Returning None immediately submits empty findings.
    seed, config, episode_data
        Forwarded to ``env.reset()``.

    Returns
    -------
    Trajectory
        Steps list, total_reward, final obs/info.
    """
    obs, info = env.reset(seed=seed, config=config, episode_data=episode_data)
    episode_id: str = info.get("episode_id", "unknown")
    traj = Trajectory(episode_id=episode_id)

    terminated = False
    truncated = False
    reward = 0.0

    while not terminated and not truncated:
        action = policy(obs, info, env)
        if action is None:
            action = {
                "tool": "submit",
                "findings": {"episode_id": episode_id, "findings": []},
            }

        obs, reward, terminated, truncated, info = env.step(action)
        traj.steps.append(StepRecord(
            step=len(traj.steps) + 1,
            action=action,
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        ))

    traj.total_reward = reward
    traj.final_obs = obs
    traj.final_info = info
    return traj


# =====================================================================
# 1. RandomPolicy
# =====================================================================

class RandomPolicy:
    """Random exploration followed by a random or empty submit.

    Makes 1–5 valid tool calls (list_records / get_record /
    get_document / calculator) then submits either a handful of random
    findings or an empty list.

    State is fully re-initialised on the first call after each reset so
    the same instance can be reused across episodes.
    """

    def __init__(self, seed: int = 0) -> None:
        self._init_seed = seed
        self._rng: Optional[_random_module.Random] = None
        self._explore_budget = 0
        self._steps_done = 0
        self._episode_id = ""
        self._tables = ("ledger", "invoices", "payments", "fx_rates")

    # Called once per episode by run_episode / external loop
    def _reset(self, info: dict, obs: dict) -> None:
        self._rng = _random_module.Random(self._init_seed + len(obs))
        self._explore_budget = self._rng.randint(1, 5)
        self._steps_done = 0
        self._episode_id = info.get(
            "episode_id",
            obs.get("manifest", {}).get("episode_id", "unknown"),
        )

    def __call__(self, obs: dict, info: dict, env: AuditEnv) -> dict:
        # First call: initialise
        if self._rng is None or obs.get("type") is None:
            self._reset(info, obs)

        self._steps_done += 1
        remaining = obs.get("remaining_steps", 1)

        # Submit when exploration budget exhausted or step limit near
        if self._steps_done > self._explore_budget or remaining <= 2:
            return self._make_submit()

        # Random exploration
        choice = self._rng.choice(
            ["list_records", "list_records", "get_document", "calculator"]
        )
        if choice == "list_records":
            return {
                "tool": "list_records",
                "table": self._rng.choice(self._tables),
                "cursor": 0,
                "limit": 50,
            }
        elif choice == "get_document":
            return {
                "tool": "get_document",
                "doc_id": f"DOC-{self._rng.randint(1, 50):04d}",
            }
        else:
            a = self._rng.randint(1, 10000)
            b = self._rng.randint(1, 10000)
            return {"tool": "calculator", "expression": f"{a} + {b}"}

    def _make_submit(self) -> dict:
        rng = self._rng
        if rng.random() < 0.5:
            return {
                "tool": "submit",
                "findings": {"episode_id": self._episode_id, "findings": []},
            }
        num = rng.randint(1, 5)
        findings = []
        for _ in range(num):
            findings.append({
                "type": rng.choice(list(VALID_ISSUE_TYPES)),
                "affected_ids": [
                    f"INV-{rng.randint(1, 30):04d}",
                ],
                "claimed_amount_minor": rng.randint(1000, 100_000),
                "currency": rng.choice(["USD", "EUR", "GBP"]),
                "evidence_ids": [f"DOC-{rng.randint(1, 30):04d}"],
                "explanation": "Random heuristic finding",
            })
        return {
            "tool": "submit",
            "findings": {"episode_id": self._episode_id, "findings": findings},
        }

    # Allow reuse across episodes: reset internal state
    def reset(self) -> None:
        self._rng = None
        self._steps_done = 0


# =====================================================================
# 2. WeakHeuristicPolicy
# =====================================================================

class WeakHeuristicPolicy:
    """Shallow first-page heuristic auditor.

    Downloads the first 50 invoices and first 50 payments then flags:
    - Payments with no invoice_refs → unallocated.
    - Invoices with the same (entity, amount, currency) on the first page
      → naive duplicate detection (WRONG on distractors like recurring
        subscriptions and legitimate split payments).

    Intentionally does NOT handle split payments, refunds (negative
    total_minor), FX conversions, or cutoff errors — it will produce
    false positives and miss many real issues.
    """

    def __init__(self) -> None:
        self._phase = 0
        self._episode_id = ""
        self._invoices: list[dict] = []
        self._payments: list[dict] = []

    def __call__(self, obs: dict, info: dict, env: AuditEnv) -> dict:
        if self._phase == 0:
            self._episode_id = info.get(
                "episode_id",
                obs.get("manifest", {}).get("episode_id", "unknown"),
            )
            self._invoices = []
            self._payments = []
            self._phase = 1
            return {"tool": "list_records", "table": "invoices",
                    "cursor": 0, "limit": 50}

        if self._phase == 1:
            if obs.get("type") == "list_records":
                self._invoices = obs.get("records", [])
            self._phase = 2
            return {"tool": "list_records", "table": "payments",
                    "cursor": 0, "limit": 50}

        if self._phase == 2:
            if obs.get("type") == "list_records":
                self._payments = obs.get("records", [])
            self._phase = 99
            return self._build_submit()

        # Safety fallback
        return {
            "tool": "submit",
            "findings": {"episode_id": self._episode_id, "findings": []},
        }

    def _build_submit(self) -> dict:
        findings: list[dict] = []

        # Flag payments with empty invoice_refs
        for pay in self._payments:
            if not pay.get("invoice_refs"):
                findings.append({
                    "type": "unallocated_or_partial_payment",
                    "affected_ids": [pay.get("payment_id", "")],
                    "claimed_amount_minor": abs(pay.get("amount_minor", 0)),
                    "currency": pay.get("currency", "USD"),
                    "evidence_ids": [pay.get("doc_id", "")],
                    "explanation": "Payment has no invoice references",
                })

        # Naive duplicate detection: same (entity, total, currency) on first page
        seen: dict[tuple, dict] = {}
        for inv in self._invoices:
            if inv.get("total_minor", 0) <= 0:
                continue
            key = (
                inv.get("entity_id", ""),
                inv.get("total_minor", 0),
                inv.get("currency", ""),
            )
            if key in seen:
                prev = seen[key]
                findings.append({
                    "type": "duplicate_invoice",
                    "affected_ids": [
                        prev.get("invoice_id", ""),
                        inv.get("invoice_id", ""),
                    ],
                    "claimed_amount_minor": inv.get("total_minor", 0),
                    "currency": inv.get("currency", "USD"),
                    "evidence_ids": [
                        prev.get("doc_id", ""),
                        inv.get("doc_id", ""),
                    ],
                    "explanation": "Same entity, amount, currency — naive duplicate flag",
                })
            else:
                seen[key] = inv

        return {
            "tool": "submit",
            "findings": {
                "episode_id": self._episode_id,
                "findings": findings[:30],
            },
        }

    def reset(self) -> None:
        self._phase = 0
        self._invoices = []
        self._payments = []


# =====================================================================
# 3. StrongBaselinePolicy
# =====================================================================

class StrongBaselinePolicy:
    """Full deterministic rule-based auditor.

    Works ONLY through AuditEnv observations (no access to closed files).

    Strategy (guaranteed ≤ 100 steps):
      Phase 1 – Paginate ALL tables via list_records (50 rows/page).
                invoices, payments, ledger, fx_rates.
                Worst case: ~200 records each → 4 pages each → 16 steps.
                Default generator target_docs=60 → ≤4 steps per table.
      Phase 2 – single submit step.
      Total: typically 8–20 steps, never >60 for the default config.

    Error detection:
      A. duplicate_invoice   – same entity, amount, currency AND matching
                               line-item signatures; skip recurring subscriptions
                               (≥3 fully-paid invoices with ~monthly gaps).
      B. unallocated_or_partial_payment – same-currency paid total < invoice
                               total; skip cross-currency invoices (FX check)
                               and invoices with matching refunds.
      C. wrong_fx_date_or_rate – cross-currency payments; re-compute
                               correct amount using fx_date_rule from policy_summary.
      D. cutoff_error        – invoice dated AFTER period_end appears in ledger.
    """

    def __init__(self) -> None:
        self._phase = "init"
        self._episode_id = ""
        self._manifest: dict = {}
        self._policy_summary: dict = {}

        self._invoices: list[dict] = []
        self._payments: list[dict] = []
        self._ledger: list[dict] = []
        self._fx_rates: list[dict] = []

        # Pagination
        _TABLE_ORDER = ["invoices", "payments", "ledger", "fx_rates"]
        self._tables_to_load: list[str] = list(_TABLE_ORDER)
        self._tbl_idx: int = 0
        self._next_cursor: Optional[int] = 0

    def __call__(self, obs: dict, info: dict, env: AuditEnv) -> dict:
        # ── Initialisation ──────────────────────────────────────────
        if self._phase == "init":
            self._episode_id = info.get(
                "episode_id",
                obs.get("manifest", {}).get("episode_id", "unknown"),
            )
            self._manifest = obs.get("manifest", {})
            self._policy_summary = obs.get("policy_summary", {})
            self._invoices = []
            self._payments = []
            self._ledger = []
            self._fx_rates = []
            self._tbl_idx = 0
            self._next_cursor = 0
            self._phase = "paginate"
            return self._page_action()

        # ── Pagination ──────────────────────────────────────────────
        if self._phase == "paginate":
            if obs.get("type") == "list_records":
                self._store(obs.get("table", ""), obs.get("records", []))
                self._next_cursor = obs.get("next_cursor")  # None when exhausted
            else:
                self._next_cursor = None  # error response → skip to next table

            if self._next_cursor is not None:
                # More pages for the current table
                return self._page_action()

            # Move to next table
            self._tbl_idx += 1
            self._next_cursor = 0
            if self._tbl_idx < len(self._tables_to_load):
                return self._page_action()

            # All tables loaded
            self._phase = "submit"
            return self._build_submit()

        # Fallback
        return {
            "tool": "submit",
            "findings": {"episode_id": self._episode_id, "findings": []},
        }

    def _page_action(self) -> dict:
        return {
            "tool": "list_records",
            "table": self._tables_to_load[self._tbl_idx],
            "cursor": self._next_cursor or 0,
            "limit": 50,
        }

    def _store(self, table: str, records: list[dict]) -> None:
        if table == "invoices":
            self._invoices.extend(records)
        elif table == "payments":
            self._payments.extend(records)
        elif table == "ledger":
            self._ledger.extend(records)
        elif table == "fx_rates":
            self._fx_rates.extend(records)

    # ── Analysis ────────────────────────────────────────────────────

    def _build_submit(self) -> dict:
        findings: list[dict] = []

        inv_by_id: dict[str, dict] = {
            inv["invoice_id"]: inv for inv in self._invoices
        }

        # Build FX lookup: (date, src_currency, tgt_currency) → Decimal
        fx_lookup: dict[tuple, Decimal] = {}
        for fx in self._fx_rates:
            fx_lookup[(fx["date"], fx["source_currency"], fx["target_currency"])] = Decimal(fx["rate"])

        base_ccy = self._manifest.get("base_currency", "USD")
        period_end = self._manifest.get("period_end", "")[:10]
        fx_rule = self._policy_summary.get("fx_date_rule", "invoice_date")
        tolerance = 100  # default from policy

        # A. Duplicate invoices
        self._detect_duplicates(findings)

        # B. Partial / unallocated payments
        self._check_allocation(findings, inv_by_id)

        # C. Wrong FX rate
        self._check_fx(findings, inv_by_id, fx_lookup, base_ccy, fx_rule, tolerance)

        # D. Cutoff errors
        self._check_cutoff(findings, inv_by_id, period_end)

        return {
            "tool": "submit",
            "findings": {
                "episode_id": self._episode_id,
                "findings": findings[:30],
            },
        }

    # ── A: duplicates ────────────────────────────────────────────────

    def _detect_duplicates(self, findings: list[dict]) -> None:
        groups: dict[tuple, list[dict]] = {}
        for inv in self._invoices:
            if inv.get("total_minor", 0) <= 0:
                continue
            key = (inv["entity_id"], inv["total_minor"], inv["currency"])
            groups.setdefault(key, []).append(inv)

        for key, group in groups.items():
            if len(group) < 2:
                continue

            # Sub-group by line-item signature
            def _sig(inv: dict) -> frozenset:
                return frozenset(
                    (it.get("description", ""), it.get("amount_minor", 0), it.get("currency", ""))
                    for it in inv.get("line_items", [])
                )

            sig_groups: dict[frozenset, list[dict]] = {}
            for inv in group:
                sig_groups.setdefault(_sig(inv), []).append(inv)

            for sig, invs in sig_groups.items():
                if len(invs) < 2:
                    continue
                if self._is_recurring(invs):
                    continue
                # Report all pairs (but cap total findings later)
                for i in range(len(invs)):
                    for j in range(i + 1, len(invs)):
                        a, b = invs[i], invs[j]
                        findings.append({
                            "type": "duplicate_invoice",
                            "affected_ids": [a["invoice_id"], b["invoice_id"]],
                            "claimed_amount_minor": a["total_minor"],
                            "currency": a["currency"],
                            "evidence_ids": [a["doc_id"], b["doc_id"]],
                            "explanation": (
                                f"Duplicate: {a['invoice_id']} and {b['invoice_id']} "
                                f"same entity/amount/currency/line-items"
                            ),
                        })

    def _is_recurring(self, invs: list[dict]) -> bool:
        """True when ≥3 same-amount invoices are ~monthly spaced AND fully paid."""
        if len(invs) < 3:
            return False

        # Build paid-per-invoice map
        paid_per: dict[str, int] = {}
        for pay in self._payments:
            for ref in pay.get("invoice_refs", []):
                paid_per.setdefault(ref, 0)
                paid_per[ref] += pay.get("amount_minor", 0)

        fully_paid = all(
            paid_per.get(inv["invoice_id"], 0) >= inv["total_minor"]
            for inv in invs
        )
        if not fully_paid:
            return False

        dates = sorted(inv["issued_at"][:10] for inv in invs)
        gaps = [
            (date.fromisoformat(dates[i]) - date.fromisoformat(dates[i - 1])).days
            for i in range(1, len(dates))
        ]
        return all(20 <= g <= 42 for g in gaps)

    # ── B: payment allocation ────────────────────────────────────────

    def _check_allocation(
        self,
        findings: list[dict],
        inv_by_id: dict[str, dict],
    ) -> None:
        # Find invoices that have cross-currency payments → exclude from allocation check
        cross_currency_invs: set[str] = set()
        for pay in self._payments:
            for ref in pay.get("invoice_refs", []):
                inv = inv_by_id.get(ref)
                if inv and pay["currency"] != inv["currency"]:
                    cross_currency_invs.add(ref)

        # Build refund set: entities with negative-total invoices
        refund_entities: set[str] = {
            inv["entity_id"]
            for inv in self._invoices
            if inv.get("total_minor", 0) < 0
        }

        # Accumulate same-currency payments per invoice_id
        paid: dict[str, int] = {}
        pay_docs: dict[str, list[str]] = {}
        pay_ids_map: dict[str, list[str]] = {}
        for pay in self._payments:
            for ref in pay.get("invoice_refs", []):
                inv = inv_by_id.get(ref)
                if inv is None:
                    continue
                if pay["currency"] != inv["currency"]:
                    continue
                paid.setdefault(ref, 0)
                paid[ref] += pay["amount_minor"]
                pay_docs.setdefault(ref, []).append(pay["doc_id"])
                pay_ids_map.setdefault(ref, []).append(pay["payment_id"])

        for inv_id, inv in inv_by_id.items():
            if inv["total_minor"] <= 0:
                continue
            if inv_id in cross_currency_invs:
                continue
            # Skip if entity has refunds (legitimate credit note scenario)
            if inv["entity_id"] in refund_entities:
                continue

            total_paid = paid.get(inv_id, 0)
            if total_paid >= inv["total_minor"]:
                continue
            pids = pay_ids_map.get(inv_id, [])
            pdocs = pay_docs.get(inv_id, [])
            if not pids:
                continue  # no payments at all — may be pending; don't flag

            shortfall = inv["total_minor"] - total_paid
            evidence = ([inv["doc_id"]] + pdocs)[:10]
            findings.append({
                "type": "unallocated_or_partial_payment",
                "affected_ids": [inv_id] + pids,
                "claimed_amount_minor": shortfall,
                "currency": inv["currency"],
                "evidence_ids": evidence,
                "explanation": (
                    f"Invoice {inv_id} total={inv['total_minor']} "
                    f"but only {total_paid} paid in same currency (shortfall {shortfall})"
                ),
            })

    # ── C: FX rate check ────────────────────────────────────────────

    def _get_rate(
        self,
        fx_lookup: dict[tuple, Decimal],
        dt: str,
        base: str,
        foreign: str,
    ) -> Optional[Decimal]:
        r = fx_lookup.get((dt, base, foreign))
        if r is not None:
            return r
        r = fx_lookup.get((dt, foreign, base))
        if r is not None:
            return Decimal("1") / r
        return None

    def _check_fx(
        self,
        findings: list[dict],
        inv_by_id: dict[str, dict],
        fx_lookup: dict[tuple, Decimal],
        base_ccy: str,
        fx_rule: str,
        tolerance: int,
    ) -> None:
        for pay in self._payments:
            for ref in pay.get("invoice_refs", []):
                inv = inv_by_id.get(ref)
                if inv is None:
                    continue
                if pay["currency"] == inv["currency"]:
                    continue  # same currency — handled by allocation check

                # Determine correct FX date per policy
                fx_date = (
                    pay["timestamp"][:10]
                    if fx_rule == "payment_date"
                    else inv["issued_at"][:10]
                )

                rate = self._get_rate(fx_lookup, fx_date, base_ccy, inv["currency"])
                if rate is None:
                    continue

                expected = int(
                    (Decimal(inv["total_minor"]) * rate).quantize(
                        Decimal("1"), rounding=ROUND_HALF_EVEN
                    )
                )
                discrepancy = abs(pay["amount_minor"] - expected)
                if discrepancy > tolerance:
                    findings.append({
                        "type": "wrong_fx_date_or_rate",
                        "affected_ids": [inv["invoice_id"], pay["payment_id"]],
                        "claimed_amount_minor": discrepancy,
                        "currency": pay["currency"],
                        "evidence_ids": [inv["doc_id"], pay["doc_id"]],
                        "explanation": (
                            f"FX mismatch: payment {pay['payment_id']} "
                            f"= {pay['amount_minor']} but expected {expected} "
                            f"(discrepancy {discrepancy}) using rate on {fx_date}"
                        ),
                    })

    # ── D: cutoff errors ─────────────────────────────────────────────

    def _check_cutoff(
        self,
        findings: list[dict],
        inv_by_id: dict[str, dict],
        period_end: str,
    ) -> None:
        if not period_end:
            return
        ledger_inv_ids: set[str] = {
            row["invoice_id"]
            for row in self._ledger
            if row.get("invoice_id")
        }
        row_ids_per_inv: dict[str, list[str]] = {}
        for row in self._ledger:
            iid = row.get("invoice_id")
            if iid:
                row_ids_per_inv.setdefault(iid, []).append(row["row_id"])

        for inv_id in ledger_inv_ids:
            inv = inv_by_id.get(inv_id)
            if inv is None:
                continue
            inv_date = inv["issued_at"][:10]
            if inv_date > period_end:
                rows = row_ids_per_inv.get(inv_id, [])[:8]
                evidence = ([inv["doc_id"]] + rows)[:10]
                findings.append({
                    "type": "cutoff_error",
                    "affected_ids": [inv_id],
                    "claimed_amount_minor": abs(inv["total_minor"]),
                    "currency": inv["currency"],
                    "evidence_ids": evidence,
                    "explanation": (
                        f"Invoice {inv_id} issued {inv_date} is after "
                        f"period end {period_end} but recorded in ledger"
                    ),
                })

    def reset(self) -> None:
        self._phase = "init"


# =====================================================================
# 4. OraclePolicy
# =====================================================================

class OraclePolicy:
    """Upper-bound benchmark.

    Reads ``env._episode.truth_issues`` directly (closed data) and
    submits a perfect answer on the very first step, achieving F1 == 1.0.

    Not a valid auditing agent — it exists solely to verify the scorer.
    """

    def __call__(self, obs: dict, info: dict, env: AuditEnv) -> dict:
        ep = env._episode
        episode_id = ep.manifest.episode_id

        findings: list[dict] = []
        for issue in ep.truth_issues:
            evidence = (
                list(issue.acceptable_evidence_sets[0])
                if issue.acceptable_evidence_sets
                else []
            )
            findings.append({
                "type": issue.type,
                "affected_ids": list(issue.affected_ids),
                "claimed_amount_minor": issue.expected_amount_minor,
                "currency": issue.currency,
                "evidence_ids": evidence[:10],
                "explanation": f"Oracle: {issue.issue_id}",
            })

        return {
            "tool": "submit",
            "findings": {"episode_id": episode_id, "findings": findings},
        }
