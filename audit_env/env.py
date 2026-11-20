"""
RL environment with reset/step contract — section 04 of SPEC.md.

Implements a Gymnasium-style JSON-compatible interface (without inheriting
from gymnasium.Env) for the financial audit task.

Actions
-------
list_records(table, cursor=None, limit=50)   – paginated read (max 50 rows)
get_record(table, id)                        – single record by ID
get_document(doc_id)                         – invoice or payment by doc_id
calculator(expression)                       – safe arithmetic
submit(findings)                             – terminal action

Observation contract
--------------------
Only requested records, result/error and remaining steps.  No shell, SQL,
filesystem or internet access.  No truth, provenance, seed or hints in
observations — even in error messages.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from audit_env.calculator import CalculatorError, calculate
from audit_env.generator import EpisodeData, generate_episode
from audit_env.matcher import evaluate_submission
from audit_env.schemas import Policy, TruthIssue


_MAX_STEPS = 100
_PAGE_SIZE = 50

_VALID_TABLES = ("ledger", "invoices", "payments", "fx_rates")

_TASK_DESCRIPTION = (
    "You are an AI auditor. Examine the company's financial records for the "
    "reporting period described in the manifest. Identify any accounting "
    "errors and submit your findings with supporting evidence before your "
    "step budget runs out."
)

_TOOL_SPEC = {
    "list_records": {
        "description": "Paginated listing of a table. Returns up to 50 rows.",
        "params": {"table": "str (ledger|invoices|payments|fx_rates)",
                   "cursor": "int or null (0-based offset, default 0)",
                   "limit": "int (1..50, default 50)"},
    },
    "get_record": {
        "description": "Fetch a single record by its primary ID.",
        "params": {"table": "str", "id": "str"},
    },
    "get_document": {
        "description": "Fetch an invoice or payment document by doc_id.",
        "params": {"doc_id": "str"},
    },
    "calculator": {
        "description": "Evaluate a safe arithmetic expression (decimals, +, -, *, /, parentheses).",
        "params": {"expression": "str"},
    },
    "submit": {
        "description": "Submit findings and end the episode.",
        "params": {"findings": "object conforming to findings.json schema"},
    },
}


class AuditEnv:
    """RL environment for the financial audit task (SPEC §04).

    Follows the Gymnasium ``reset`` / ``step`` contract without inheriting
    from ``gymnasium.Env``.  All state is fully isolated between episodes.
    """

    def __init__(self, eval_mode: bool = True) -> None:
        self.eval_mode = eval_mode
        self._episode: Optional[EpisodeData] = None
        self._step_count: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        # Indexing structures built on reset
        self._ledger_index: dict[str, dict] = {}
        self._invoice_index: dict[str, dict] = {}
        self._payment_index: dict[str, dict] = {}
        self._doc_index: dict[str, dict] = {}
        self._fx_list: list[dict] = []
        # Ordered lists for pagination
        self._tables: dict[str, list[dict]] = {}
        # Evaluation result (available after submit)
        self.last_eval_result: Optional[dict] = None

    # ── reset ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        config: Optional[dict] = None,
        episode_data: Optional[EpisodeData] = None,
    ) -> tuple[dict, dict]:
        """Start a new episode.

        Parameters
        ----------
        seed : int, optional
            RNG seed for the generator.  Ignored when *episode_data* is given.
        config : dict, optional
            Generator config overrides.  Ignored when *episode_data* is given.
        episode_data : EpisodeData, optional
            Pre-built episode (e.g. loaded from disk).  Overrides seed/config.

        Returns
        -------
        (observation, info)
        """
        # ── generate or accept episode data ──
        if episode_data is not None:
            self._episode = episode_data
        else:
            if seed is None:
                seed = 0
            self._episode = generate_episode(seed, config)

        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self.last_eval_result = None

        # ── build indexes ──
        ep = self._episode

        # Ledger
        ledger_dicts = [r.to_dict() for r in ep.ledger]
        self._ledger_index = {d["row_id"]: d for d in ledger_dicts}

        # Invoices
        inv_dicts = [inv.to_dict() for inv in ep.invoices]
        self._invoice_index = {d["invoice_id"]: d for d in inv_dicts}
        doc_map: dict[str, dict] = {}
        for d in inv_dicts:
            doc_map[d["doc_id"]] = d

        # Payments
        pay_dicts = [p.to_dict() for p in ep.payments]
        self._payment_index = {d["payment_id"]: d for d in pay_dicts}
        for d in pay_dicts:
            doc_map[d["doc_id"]] = d
        self._doc_index = doc_map

        # FX
        fx_dicts = [fx.to_dict() for fx in ep.fx_rates]

        # Pagination tables (stable order — whatever order the generator set)
        self._tables = {
            "ledger": ledger_dicts,
            "invoices": inv_dicts,
            "payments": pay_dicts,
            "fx_rates": fx_dicts,
        }

        # ── build observation ──
        manifest_dict = ep.manifest.to_dict()
        policy_summary = {
            "fx_date_rule": ep.policy.fx_date_rule,
            "period_boundary": ep.policy.period_boundary,
            "rounding": ep.policy.rounding,
        }

        observation = {
            "task": _TASK_DESCRIPTION,
            "manifest": manifest_dict,
            "policy_summary": policy_summary,
            "tools": _TOOL_SPEC,
            "remaining_steps": _MAX_STEPS,
        }

        info: dict[str, Any] = {
            "episode_id": ep.manifest.episode_id,
        }

        return observation, info

    # ── step ──────────────────────────────────────────────────────────

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """Execute one action.

        Parameters
        ----------
        action : dict
            Must have ``"tool"`` (str) and tool-specific params.

        Returns
        -------
        (observation, reward, terminated, truncated, info)
        """
        if self._episode is None:
            raise RuntimeError("Must call reset() before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("Episode already finished; call reset()")

        self._step_count += 1
        remaining = _MAX_STEPS - self._step_count

        # ── parse action ──
        if not isinstance(action, dict) or "tool" not in action:
            return self._error_obs("Invalid action: must be a dict with 'tool' key.", remaining)

        tool_name = action.get("tool")

        # ── dispatch ──
        if tool_name == "list_records":
            return self._do_list_records(action, remaining)
        elif tool_name == "get_record":
            return self._do_get_record(action, remaining)
        elif tool_name == "get_document":
            return self._do_get_document(action, remaining)
        elif tool_name == "calculator":
            return self._do_calculator(action, remaining)
        elif tool_name == "submit":
            return self._do_submit(action, remaining)
        else:
            return self._error_obs(
                f"Unknown tool '{tool_name}'. Available: "
                f"list_records, get_record, get_document, calculator, submit.",
                remaining,
            )

    # ── tool implementations ──────────────────────────────────────────

    def _do_list_records(self, action: dict, remaining: int):
        table = action.get("table")
        if table not in _VALID_TABLES:
            return self._error_obs(
                f"Unknown table '{table}'. Valid: {', '.join(_VALID_TABLES)}.",
                remaining,
            )

        cursor = action.get("cursor", 0)
        if cursor is None:
            cursor = 0
        if not isinstance(cursor, int) or cursor < 0:
            return self._error_obs(
                f"Invalid cursor: must be a non-negative integer, got {cursor!r}.",
                remaining,
            )

        limit = action.get("limit", _PAGE_SIZE)
        if not isinstance(limit, int) or limit < 1:
            limit = _PAGE_SIZE
        limit = min(limit, _PAGE_SIZE)

        data = self._tables[table]
        page = data[cursor: cursor + limit]
        next_cursor = cursor + limit if cursor + limit < len(data) else None

        obs = {
            "type": "list_records",
            "table": table,
            "records": page,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "total": len(data),
            "remaining_steps": remaining,
        }

        # Check if we hit the step limit AFTER this non-submit action
        if remaining <= 0:
            return self._truncate(obs)
        return obs, 0.0, False, False, {}

    def _do_get_record(self, action: dict, remaining: int):
        table = action.get("table")
        record_id = action.get("id")

        if table not in _VALID_TABLES:
            return self._error_obs(
                f"Unknown table '{table}'. Valid: {', '.join(_VALID_TABLES)}.",
                remaining,
            )

        if not isinstance(record_id, str):
            return self._error_obs("'id' must be a string.", remaining)

        # Look up by primary key depending on table
        record = None
        if table == "ledger":
            record = self._ledger_index.get(record_id)
        elif table == "invoices":
            record = self._invoice_index.get(record_id)
        elif table == "payments":
            record = self._payment_index.get(record_id)
        elif table == "fx_rates":
            # FX has no single-key index; search by composite
            for fx in self._tables["fx_rates"]:
                if fx.get("date") == record_id:
                    record = fx
                    break

        if record is None:
            return self._error_obs(
                f"Record '{record_id}' not found in table '{table}'.",
                remaining,
            )

        obs = {
            "type": "get_record",
            "table": table,
            "record": record,
            "remaining_steps": remaining,
        }
        if remaining <= 0:
            return self._truncate(obs)
        return obs, 0.0, False, False, {}

    def _do_get_document(self, action: dict, remaining: int):
        doc_id = action.get("doc_id")
        if not isinstance(doc_id, str):
            return self._error_obs("'doc_id' must be a string.", remaining)

        doc = self._doc_index.get(doc_id)
        if doc is None:
            return self._error_obs(
                f"Document '{doc_id}' not found.",
                remaining,
            )

        obs = {
            "type": "get_document",
            "document": doc,
            "remaining_steps": remaining,
        }
        if remaining <= 0:
            return self._truncate(obs)
        return obs, 0.0, False, False, {}

    def _do_calculator(self, action: dict, remaining: int):
        expression = action.get("expression")
        if not isinstance(expression, str):
            return self._error_obs("'expression' must be a string.", remaining)

        try:
            result = calculate(expression)
        except CalculatorError as exc:
            return self._error_obs(f"Calculator error: {exc}", remaining)

        obs = {
            "type": "calculator",
            "expression": expression,
            "result": result,
            "remaining_steps": remaining,
        }
        if remaining <= 0:
            return self._truncate(obs)
        return obs, 0.0, False, False, {}

    def _do_submit(self, action: dict, remaining: int):
        """Terminal action — evaluate the submission."""
        ep = self._episode
        findings_payload = action.get("findings")

        eval_result = evaluate_submission(
            findings_payload,
            ep.truth_issues,
            ep.policy,
            inspection_steps=self._step_count,
            expected_episode_id=ep.manifest.episode_id,
        )

        self._terminated = True
        self.last_eval_result = eval_result

        reward = eval_result["reward"]

        if self.eval_mode:
            # Closed evaluation — reveal only reward, not diagnostics
            obs: dict[str, Any] = {
                "type": "submit",
                "status": "valid" if eval_result["valid"] else "invalid",
                "remaining_steps": remaining,
            }
            info: dict[str, Any] = {}
        else:
            # Open training mode — return metrics to the policy
            obs = {
                "type": "submit",
                "status": "valid" if eval_result["valid"] else "invalid",
                "reward": reward,
                "precision": eval_result["precision"],
                "recall": eval_result["recall"],
                "f1": eval_result["f1"],
                "remaining_steps": remaining,
            }
            info = {
                "tp": eval_result["tp"],
                "fp": eval_result["fp"],
                "fn": eval_result["fn"],
            }

        return obs, reward, True, False, info

    # ── helpers ───────────────────────────────────────────────────────

    def _error_obs(self, message: str, remaining: int):
        """Return an error observation consuming one step."""
        obs = {
            "type": "error",
            "error": message,
            "remaining_steps": remaining,
        }
        if remaining <= 0:
            return self._truncate(obs)
        return obs, 0.0, False, False, {}

    def _truncate(self, last_obs: dict):
        """Handle step-limit truncation (§04: truncated=True, empty final answer)."""
        self._truncated = True

        # Compute truncation reward: submit with empty findings
        ep = self._episode
        eval_result = evaluate_submission(
            {"episode_id": ep.manifest.episode_id, "findings": []},
            ep.truth_issues,
            ep.policy,
            inspection_steps=self._step_count,
            expected_episode_id=ep.manifest.episode_id,
        )
        self.last_eval_result = eval_result
        reward = eval_result["reward"]

        last_obs["truncated"] = True
        return last_obs, reward, False, True, {}
