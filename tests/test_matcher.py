"""
Exhaustive tests for audit_env.matcher — section 05 of SPEC.md.

Covers every check listed in the spec and in the task brief:
  - Perfect answer beats silence on erroneous books.
  - Silence on clean books (F1=1) beats false accusations.
  - Duplicate findings do not help (extra copies are FP).
  - "Flag everything" attack yields poor / negative reward.
  - Step-cost (inspection_steps) differentiates equal-F1 answers.
  - Guessed IDs, incomplete evidence, wrong currency, tolerance
    boundary, and structurally-invalid submissions.
"""

from __future__ import annotations

import pytest

from audit_env.schemas import (
    Finding,
    Policy,
    SchemaError,
    Submission,
    TruthIssue,
    VALID_ISSUE_TYPES,
    MAX_FINDINGS,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_LEN,
)
from audit_env.matcher import evaluate_submission


# ===================================================================
# Fixtures / helpers
# ===================================================================

def _policy(tolerance: int = 0) -> Policy:
    return Policy(
        fx_date_rule="invoice_date",
        period_boundary="inclusive_start_exclusive_end",
        rounding="ROUND_HALF_EVEN",
        amount_tolerance_minor=tolerance,
    )


def _truth(
    issue_id: str = "ISS-1",
    type_: str = "duplicate_invoice",
    affected_ids: tuple[str, ...] = ("INV-001", "INV-002"),
    expected_amount: int = 50000,
    currency: str = "USD",
    evidence_sets: tuple[tuple[str, ...], ...] = (("DOC-1", "DOC-2"),),
) -> TruthIssue:
    return TruthIssue(
        issue_id=issue_id,
        type=type_,
        affected_ids=affected_ids,
        expected_amount_minor=expected_amount,
        currency=currency,
        acceptable_evidence_sets=evidence_sets,
    )


def _finding_dict(
    type_: str = "duplicate_invoice",
    affected_ids: list[str] | None = None,
    claimed_amount: int = 50000,
    currency: str = "USD",
    evidence_ids: list[str] | None = None,
    explanation: str = "Duplicate detected.",
) -> dict:
    return {
        "type": type_,
        "affected_ids": affected_ids if affected_ids is not None else ["INV-001", "INV-002"],
        "claimed_amount_minor": claimed_amount,
        "currency": currency,
        "evidence_ids": evidence_ids if evidence_ids is not None else ["DOC-1", "DOC-2"],
        "explanation": explanation,
    }


def _submission(
    episode_id: str = "ep-001",
    findings: list[dict] | None = None,
) -> dict:
    return {
        "episode_id": episode_id,
        "findings": findings if findings is not None else [],
    }


# ===================================================================
# 1. Perfect correct answer > silence on erroneous books
# ===================================================================

class TestPerfectVsSilence:
    """SPEC: 'Full correct answer with evidence beats silence on erroneous books.'"""

    def test_perfect_beats_silence(self):
        truth = [_truth()]
        policy = _policy()

        # Perfect answer
        perfect = _submission(findings=[_finding_dict()])
        r_perfect = evaluate_submission(perfect, truth, policy, inspection_steps=10)

        # Silence (no findings)
        silence = _submission(findings=[])
        r_silence = evaluate_submission(silence, truth, policy, inspection_steps=10)

        assert r_perfect["valid"]
        assert r_silence["valid"]
        assert r_perfect["reward"] > r_silence["reward"]
        assert r_perfect["f1"] == 1.0
        assert r_silence["f1"] == 0.0

    def test_perfect_tp_fp_fn(self):
        truth = [_truth()]
        result = evaluate_submission(
            _submission(findings=[_finding_dict()]),
            truth, _policy(), inspection_steps=0,
        )
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["f1"] == 1.0
        # reward = 2*1 - 1 - 0 = 1.0
        assert result["reward"] == 1.0

    def test_silence_on_errors_gives_negative(self):
        truth = [_truth()]
        result = evaluate_submission(
            _submission(findings=[]),
            truth, _policy(), inspection_steps=0,
        )
        assert result["tp"] == 0
        assert result["fn"] == 1
        # F1 = 0 → reward = 2*0 - 1 = -1.0
        assert result["reward"] == -1.0


# ===================================================================
# 2. Silence on clean books (F1=1) > false accusations
# ===================================================================

class TestCleanBooks:
    """SPEC: 'Silence is better than false accusations on clean books.'"""

    def test_silence_on_clean_gives_f1_one(self):
        result = evaluate_submission(
            _submission(findings=[]),
            truth_issues=[],  # clean
            policy=_policy(),
            inspection_steps=0,
        )
        assert result["valid"]
        assert result["f1"] == 1.0
        assert result["reward"] == 1.0

    def test_false_accusation_on_clean(self):
        result = evaluate_submission(
            _submission(findings=[_finding_dict()]),
            truth_issues=[],
            policy=_policy(),
            inspection_steps=0,
        )
        assert result["valid"]
        assert result["f1"] == 0.0
        # 2*0 - 1 = -1.0
        assert result["reward"] == -1.0

    def test_silence_beats_false_accusation(self):
        policy = _policy()
        r_silence = evaluate_submission(
            _submission(findings=[]), [], policy, 10,
        )
        r_false = evaluate_submission(
            _submission(findings=[_finding_dict()]), [], policy, 10,
        )
        assert r_silence["reward"] > r_false["reward"]


# ===================================================================
# 3. Duplicate findings do not help
# ===================================================================

class TestDuplicateFindings:
    """SPEC: 'Duplicates do not yield benefit.'"""

    def test_duplicate_counts_as_fp(self):
        truth = [_truth()]
        # Submit TWO identical correct findings
        findings = [_finding_dict(), _finding_dict()]
        result = evaluate_submission(
            _submission(findings=findings),
            truth, _policy(), 0,
        )
        assert result["tp"] == 1
        assert result["fp"] == 1  # second copy is FP
        assert result["fn"] == 0
        # F1 = 2*1/(2*1+1+0) = 2/3
        assert abs(result["f1"] - 2 / 3) < 1e-9

    def test_duplicate_reward_less_than_single(self):
        truth = [_truth()]
        policy = _policy()
        r_single = evaluate_submission(
            _submission(findings=[_finding_dict()]),
            truth, policy, 0,
        )
        r_dup = evaluate_submission(
            _submission(findings=[_finding_dict(), _finding_dict()]),
            truth, policy, 0,
        )
        assert r_single["reward"] > r_dup["reward"]


# ===================================================================
# 4. "Flag everything" attack gives bad reward
# ===================================================================

class TestFlagAllAttack:
    """SPEC: 'Marking all rows/types yields a bad result.'"""

    def test_flag_all_negative_reward(self):
        # 1 real issue, 30 findings (spam)
        truth = [_truth()]
        spam = [
            _finding_dict(
                affected_ids=[f"FAKE-{i}"],
                evidence_ids=[f"EV-{i}"],
            )
            for i in range(MAX_FINDINGS)
        ]
        # Ensure one finding actually matches
        spam[0] = _finding_dict()
        result = evaluate_submission(
            _submission(findings=spam),
            truth, _policy(), 0,
        )
        assert result["tp"] == 1
        assert result["fp"] == MAX_FINDINGS - 1
        assert result["fn"] == 0
        # Precision is very low → F1 ≈ 2/(2+29) ≈ 0.0645 → reward ≈ 2*0.0645-1 ≈ -0.871
        assert result["reward"] < 0

    def test_all_wrong_types_gives_zero_tp(self):
        truth = [_truth(type_="cutoff_error")]
        spam = [_finding_dict(type_="duplicate_invoice") for _ in range(10)]
        result = evaluate_submission(
            _submission(findings=spam), truth, _policy(), 0,
        )
        assert result["tp"] == 0
        assert result["fp"] == 10
        assert result["fn"] == 1
        assert result["reward"] < 0


# ===================================================================
# 5. Step cost differentiates equal-F1 answers
# ===================================================================

class TestStepCost:
    """SPEC: 'Saving steps slightly differentiates otherwise equal answers.'"""

    def test_fewer_steps_higher_reward(self):
        truth = [_truth()]
        policy = _policy()
        payload = _submission(findings=[_finding_dict()])

        r_fast = evaluate_submission(payload, truth, policy, inspection_steps=5)
        r_slow = evaluate_submission(payload, truth, policy, inspection_steps=50)

        assert r_fast["f1"] == r_slow["f1"] == 1.0
        assert r_fast["reward"] > r_slow["reward"]

    def test_step_cost_magnitude(self):
        # 10 steps → cost = 0.02, 100 steps → cost = 0.2
        truth = [_truth()]
        r10 = evaluate_submission(
            _submission(findings=[_finding_dict()]),
            truth, _policy(), 10,
        )
        r100 = evaluate_submission(
            _submission(findings=[_finding_dict()]),
            truth, _policy(), 100,
        )
        # F1 = 1 for both, reward = 1 - 0.002*steps
        assert abs(r10["reward"] - (1.0 - 0.002 * 10)) < 1e-9
        assert abs(r100["reward"] - (1.0 - 0.002 * 100)) < 1e-9

    def test_reward_clipped_at_minus_one(self):
        # Even extreme steps don't go below -1
        result = evaluate_submission(
            _submission(findings=[]),
            [_truth()], _policy(), 9999,
        )
        assert result["reward"] == -1.0


# ===================================================================
# 6. Guessed / wrong affected_ids
# ===================================================================

class TestGuessedIDs:
    """SPEC: 'Check guessed IDs.'"""

    def test_wrong_affected_ids_is_fp(self):
        truth = [_truth(affected_ids=("INV-001", "INV-002"))]
        # Agent guesses wrong IDs
        finding = _finding_dict(affected_ids=["INV-001", "INV-999"])
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_subset_of_affected_ids_is_fp(self):
        truth = [_truth(affected_ids=("INV-001", "INV-002"))]
        finding = _finding_dict(affected_ids=["INV-001"])  # missing INV-002
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 0

    def test_superset_of_affected_ids_is_fp(self):
        truth = [_truth(affected_ids=("INV-001",))]
        finding = _finding_dict(affected_ids=["INV-001", "INV-002"])
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 0

    def test_order_does_not_matter(self):
        """affected_ids matching is set-based, order-insensitive."""
        truth = [_truth(affected_ids=("INV-002", "INV-001"))]
        finding = _finding_dict(affected_ids=["INV-001", "INV-002"])
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 1


# ===================================================================
# 7. Incomplete / insufficient evidence
# ===================================================================

class TestEvidence:
    """SPEC: 'Check unconfirmed evidence.'"""

    def test_missing_evidence_is_fp(self):
        truth = [_truth(evidence_sets=(("DOC-1", "DOC-2"),))]
        # Agent provides only DOC-1 (not full set)
        finding = _finding_dict(evidence_ids=["DOC-1"])
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 0
        assert result["fp"] == 1

    def test_superset_of_evidence_is_ok(self):
        truth = [_truth(evidence_sets=(("DOC-1",),))]
        finding = _finding_dict(evidence_ids=["DOC-1", "DOC-99"])
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 1

    def test_alternative_evidence_set(self):
        """Truth has two acceptable evidence sets; either one suffices."""
        truth = [_truth(evidence_sets=(("DOC-1", "DOC-2"), ("DOC-3",)))]
        # Agent uses second alternative
        finding = _finding_dict(evidence_ids=["DOC-3"])
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 1

    def test_no_required_evidence(self):
        """If truth has no acceptable_evidence_sets, any evidence is fine."""
        truth = [_truth(evidence_sets=())]
        finding = _finding_dict(evidence_ids=[])
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 1

    def test_empty_evidence_against_required(self):
        truth = [_truth(evidence_sets=(("DOC-1",),))]
        finding = _finding_dict(evidence_ids=[])
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 0


# ===================================================================
# 8. Wrong currency
# ===================================================================

class TestWrongCurrency:
    """SPEC: 'Check currencies.'"""

    def test_wrong_currency_is_fp(self):
        truth = [_truth(currency="USD")]
        finding = _finding_dict(currency="EUR")
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(), 0,
        )
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1


# ===================================================================
# 9. Amount tolerance boundary
# ===================================================================

class TestAmountTolerance:
    """Boundary values for the amount tolerance parameter."""

    def test_exact_match_zero_tolerance(self):
        truth = [_truth(expected_amount=50000)]
        finding = _finding_dict(claimed_amount=50000)
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(tolerance=0), 0,
        )
        assert result["tp"] == 1

    def test_off_by_one_zero_tolerance(self):
        truth = [_truth(expected_amount=50000)]
        finding = _finding_dict(claimed_amount=50001)
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(tolerance=0), 0,
        )
        assert result["tp"] == 0

    def test_within_tolerance(self):
        truth = [_truth(expected_amount=50000)]
        finding = _finding_dict(claimed_amount=50100)
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(tolerance=100), 0,
        )
        assert result["tp"] == 1

    def test_at_tolerance_boundary(self):
        truth = [_truth(expected_amount=50000)]
        finding = _finding_dict(claimed_amount=50100)
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(tolerance=100), 0,
        )
        assert result["tp"] == 1  # exactly at boundary → TP

    def test_just_outside_tolerance(self):
        truth = [_truth(expected_amount=50000)]
        finding = _finding_dict(claimed_amount=50101)
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(tolerance=100), 0,
        )
        assert result["tp"] == 0

    def test_negative_direction_tolerance(self):
        truth = [_truth(expected_amount=50000)]
        finding = _finding_dict(claimed_amount=49900)
        result = evaluate_submission(
            _submission(findings=[finding]),
            truth, _policy(tolerance=100), 0,
        )
        assert result["tp"] == 1


# ===================================================================
# 10. Invalid / malformed submissions → reward = -1
# ===================================================================

class TestInvalidSubmission:
    """SPEC: 'Invalid structure terminates episode with reward = -1.'"""

    def test_not_a_dict(self):
        result = evaluate_submission("garbage", [_truth()], _policy(), 0)
        assert result["valid"] is False
        assert result["reward"] == -1.0
        assert result["error"] is not None

    def test_missing_episode_id(self):
        result = evaluate_submission(
            {"findings": []}, [_truth()], _policy(), 0,
        )
        assert result["valid"] is False
        assert result["reward"] == -1.0

    def test_too_many_findings(self):
        payload = _submission(findings=[_finding_dict() for _ in range(MAX_FINDINGS + 1)])
        result = evaluate_submission(payload, [_truth()], _policy(), 0)
        assert result["valid"] is False
        assert result["reward"] == -1.0

    def test_too_many_evidence_ids(self):
        f = _finding_dict(evidence_ids=[f"E-{i}" for i in range(MAX_EVIDENCE_IDS + 1)])
        result = evaluate_submission(
            _submission(findings=[f]),
            [_truth()], _policy(), 0,
        )
        assert result["valid"] is False
        assert result["reward"] == -1.0

    def test_explanation_too_long(self):
        f = _finding_dict(explanation="x" * (MAX_EXPLANATION_LEN + 1))
        result = evaluate_submission(
            _submission(findings=[f]),
            [_truth()], _policy(), 0,
        )
        assert result["valid"] is False
        assert result["reward"] == -1.0

    def test_invalid_type_in_finding(self):
        f = _finding_dict(type_="nonexistent_type")
        result = evaluate_submission(
            _submission(findings=[f]),
            [_truth()], _policy(), 0,
        )
        assert result["valid"] is False
        assert result["reward"] == -1.0

    def test_claimed_amount_not_int(self):
        payload = {
            "episode_id": "ep-001",
            "findings": [{
                "type": "duplicate_invoice",
                "affected_ids": ["A"],
                "claimed_amount_minor": 100.5,  # float — invalid
                "currency": "USD",
                "evidence_ids": [],
                "explanation": "x",
            }],
        }
        result = evaluate_submission(payload, [_truth()], _policy(), 0)
        assert result["valid"] is False
        assert result["reward"] == -1.0


# ===================================================================
# 11. Episode-ID mismatch
# ===================================================================

class TestEpisodeIdMismatch:
    def test_mismatch_gives_minus_one(self):
        result = evaluate_submission(
            _submission(episode_id="ep-wrong"),
            [_truth()], _policy(), 0,
            expected_episode_id="ep-001",
        )
        assert result["valid"] is False
        assert result["reward"] == -1.0
        assert "mismatch" in result["error"]

    def test_match_passes(self):
        result = evaluate_submission(
            _submission(episode_id="ep-001", findings=[_finding_dict()]),
            [_truth()], _policy(), 0,
            expected_episode_id="ep-001",
        )
        assert result["valid"]
        assert result["tp"] == 1

    def test_no_expected_id_skips_check(self):
        result = evaluate_submission(
            _submission(episode_id="anything"),
            [], _policy(), 0,
        )
        assert result["valid"]


# ===================================================================
# 12. Multi-issue matching
# ===================================================================

class TestMultiIssueMatching:
    """Greedy 1-to-1 bipartite matching over multiple truth issues."""

    def test_two_issues_two_findings(self):
        t1 = _truth(issue_id="ISS-1", type_="duplicate_invoice",
                     affected_ids=("A",), expected_amount=100, currency="USD",
                     evidence_sets=(("E1",),))
        t2 = _truth(issue_id="ISS-2", type_="cutoff_error",
                     affected_ids=("B",), expected_amount=200, currency="USD",
                     evidence_sets=(("E2",),))
        f1 = _finding_dict(type_="duplicate_invoice", affected_ids=["A"],
                           claimed_amount=100, evidence_ids=["E1"])
        f2 = _finding_dict(type_="cutoff_error", affected_ids=["B"],
                           claimed_amount=200, evidence_ids=["E2"])
        result = evaluate_submission(
            _submission(findings=[f1, f2]),
            [t1, t2], _policy(), 0,
        )
        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["f1"] == 1.0

    def test_partial_match(self):
        t1 = _truth(issue_id="ISS-1", type_="duplicate_invoice",
                     affected_ids=("A",), expected_amount=100, currency="USD",
                     evidence_sets=(("E1",),))
        t2 = _truth(issue_id="ISS-2", type_="cutoff_error",
                     affected_ids=("B",), expected_amount=200, currency="USD",
                     evidence_sets=(("E2",),))
        # Agent only finds first issue
        f1 = _finding_dict(type_="duplicate_invoice", affected_ids=["A"],
                           claimed_amount=100, evidence_ids=["E1"])
        result = evaluate_submission(
            _submission(findings=[f1]),
            [t1, t2], _policy(), 0,
        )
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 1
        # F1 = 2*1/(2+0+1) = 2/3
        assert abs(result["f1"] - 2 / 3) < 1e-9

    def test_greedy_order_matters(self):
        """First truth issue grabs the first matching finding."""
        # Both truth issues have same type/currency, different affected_ids
        t1 = _truth(issue_id="ISS-1", affected_ids=("A",), expected_amount=100,
                     evidence_sets=(("E1",),))
        t2 = _truth(issue_id="ISS-2", affected_ids=("B",), expected_amount=100,
                     evidence_sets=(("E1",),))
        # Findings match t2 first by content, but greedy goes by truth order
        f_b = _finding_dict(affected_ids=["B"], claimed_amount=100, evidence_ids=["E1"])
        f_a = _finding_dict(affected_ids=["A"], claimed_amount=100, evidence_ids=["E1"])
        result = evaluate_submission(
            _submission(findings=[f_b, f_a]),
            [t1, t2], _policy(), 0,
        )
        # t1 looks for affected_ids={"A"} → skips f_b (idx=0), matches f_a (idx=1)
        # t2 looks for affected_ids={"B"} → matches f_b (idx=0)
        assert result["tp"] == 2
        assert result["matches"] == [("ISS-1", 1), ("ISS-2", 0)]

    def test_conflicting_candidates_maximum_matching(self):
        """Kuhn's matching resolves ambiguous conflicts to achieve maximum TP.

        T1 can match either F1 or F2.
        T2 can only match F1.
        Naive greedy in order [T1, T2] would take F1 for T1, leaving T2 unmatched (TP=1).
        Kuhn's algorithm reassigns T1 to F2 and gives F1 to T2, achieving TP=2.
        """
        # Both truth issues have same type, affected_ids, currency, but different tolerances/amounts
        t1 = _truth(issue_id="ISS-1", affected_ids=("INV-1",), expected_amount=100, evidence_sets=(("E1",),))
        t2 = _truth(issue_id="ISS-2", affected_ids=("INV-1",), expected_amount=130, evidence_sets=(("E1",),))

        # f1 has amount 125 (within policy tolerance of 30 for both: |125-100|=25 <= 30, |125-130|=5 <= 30)
        # f2 has amount 80 (|80-100|=20 <= 30, but |80-130|=50 > 30, so matches only T1)
        f1 = _finding_dict(affected_ids=["INV-1"], claimed_amount=125, evidence_ids=["E1"])
        f2 = _finding_dict(affected_ids=["INV-1"], claimed_amount=80, evidence_ids=["E1"])

        policy = _policy(tolerance=30)

        # Submission presents [f1, f2] in that order
        result = evaluate_submission(
            _submission(findings=[f1, f2]),
            [t1, t2],
            policy,
            inspection_steps=0,
        )

        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["f1"] == 1.0
        assert set(result["matches"]) == {("ISS-1", 1), ("ISS-2", 0)}


# ===================================================================
# 13. Reward formula correctness
# ===================================================================

class TestRewardFormula:
    def test_perfect_zero_steps(self):
        # 2*1 - 1 - 0 = 1.0
        result = evaluate_submission(
            _submission(findings=[_finding_dict()]),
            [_truth()], _policy(), 0,
        )
        assert result["reward"] == 1.0

    def test_perfect_fifty_steps(self):
        # 2*1 - 1 - 0.002*50 = 0.9
        result = evaluate_submission(
            _submission(findings=[_finding_dict()]),
            [_truth()], _policy(), 50,
        )
        assert abs(result["reward"] - 0.9) < 1e-9

    def test_clipped_at_plus_one(self):
        # Clean books: F1=1, 0 steps → 2*1-1-0 = 1.0 (already at max)
        result = evaluate_submission(
            _submission(findings=[]),
            [], _policy(), 0,
        )
        assert result["reward"] == 1.0

    def test_clipped_at_minus_one(self):
        # F1=0, many steps → should be -1 not worse
        result = evaluate_submission(
            _submission(findings=[]),
            [_truth()], _policy(), 500,
        )
        assert result["reward"] == -1.0

    def test_intermediate_f1(self):
        # 1 TP, 1 FP, 0 FN → P=0.5, R=1, F1=2/3
        # reward = 2*(2/3) - 1 - 0 = 1/3
        t = _truth()
        findings = [
            _finding_dict(),
            _finding_dict(affected_ids=["FAKE"], evidence_ids=["EF"]),
        ]
        result = evaluate_submission(
            _submission(findings=findings),
            [t], _policy(), 0,
        )
        assert abs(result["reward"] - 1 / 3) < 1e-9


# ===================================================================
# 14. Matches list correctness
# ===================================================================

class TestMatchesList:
    def test_matches_contain_truth_id_and_index(self):
        result = evaluate_submission(
            _submission(findings=[_finding_dict()]),
            [_truth(issue_id="ISS-42")], _policy(), 0,
        )
        assert result["matches"] == [("ISS-42", 0)]

    def test_no_matches_on_clean(self):
        result = evaluate_submission(
            _submission(findings=[]), [], _policy(), 0,
        )
        assert result["matches"] == []
