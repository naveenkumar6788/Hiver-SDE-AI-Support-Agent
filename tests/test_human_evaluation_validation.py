"""
tests/test_human_evaluation_validation.py

Comprehensive test suite for the independent human evaluation and
human-vs-LLM agreement pipeline.

Verifies:
1. Missing human ratings fail validation.
2. Invalid rating values (outside 1-5, floats, non-numeric) fail validation.
3. Invalid intents (not in the 12 allowed intents) fail validation.
4. Duplicate case IDs fail validation.
5. Missing overall comments fail validation.
6. Incomplete annotations (< 50 cases) fail validation.
7. Successful validation on a valid 50-case annotation file.
8. Safety check: No fake agreement is calculated when human ratings are absent.
9. Statistical agreement calculation when complete, valid ratings are provided.
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.validate_human_annotations import (
    validate_human_annotations,
    ALLOWED_INTENTS,
    REQUIRED_COLUMNS,
    RATING_DIMENSIONS_NEW,
)
from src.evaluation.human_agreement import (
    run_human_agreement,
    check_human_rating_status,
    compute_dimension_agreement,
)


def _make_valid_50_df() -> pd.DataFrame:
    """Helper to construct a valid 50-case human evaluation DataFrame."""
    records = []
    intents = sorted(ALLOWED_INTENTS)
    for i in range(50):
        records.append({
            "case_id": f"case_{i + 1:02d}",
            "customer_message": f"Sample customer message {i + 1}",
            "system_intent": intents[i % len(intents)],
            "historical_customer_message": f"Historical question {i + 1}",
            "historical_response": f"Historical response {i + 1}",
            "system_reply": f"Draft support response {i + 1}",
            "system_decision": "ESCALATE" if i == 0 else "AUTO_HANDLE",
            "system_escalation_reason": "account_security" if i == 0 else "",
            "human_intent": intents[i % len(intents)],
            "intent_correct": "YES",
            "correctness_1_5": 4,
            "groundedness_1_5": 4,
            "relevance_1_5": 4,
            "completeness_1_5": 3,
            "tone_1_5": 5,
            "unsupported_claims_1_5": 5,
            "escalation_appropriateness_1_5": 5,
            "evidence_supported": "YES",
            "overall_comment": f"Valid thorough review of case {i + 1}.",
        })
    return pd.DataFrame(records, columns=REQUIRED_COLUMNS).astype(object)


class TestHumanEvaluationValidation:
    """Tests for validate_human_annotations."""

    def test_successful_validation(self):
        """A complete, properly formatted 50-case file must pass validation."""
        df = _make_valid_50_df()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            is_valid, errors = validate_human_annotations(f.name)
            assert is_valid is True
            assert len(errors) == 0

    def test_missing_human_ratings_fails_validation(self):
        """Empty or NaN ratings must trigger validation failure."""
        df = _make_valid_50_df()
        df.loc[0, "correctness_1_5"] = np.nan
        df.loc[1, "tone_1_5"] = ""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            is_valid, errors = validate_human_annotations(f.name)
            assert is_valid is False
            assert any("Missing rating for correctness_1_5" in e for e in errors)
            assert any("Missing rating for tone_1_5" in e for e in errors)

    def test_invalid_rating_values_fail_validation(self):
        """Ratings outside integer range 1-5 must fail validation."""
        df = _make_valid_50_df()
        df.loc[0, "correctness_1_5"] = 0
        df.loc[1, "groundedness_1_5"] = 6
        df.loc[2, "relevance_1_5"] = 3.5
        df.loc[3, "completeness_1_5"] = "invalid"

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            is_valid, errors = validate_human_annotations(f.name)
            assert is_valid is False
            assert any("Invalid rating 0" in e or "Invalid rating 0.0" in e for e in errors)
            assert any("Invalid rating 6" in e or "Invalid rating 6.0" in e for e in errors)
            assert any("Invalid rating 3.5" in e for e in errors)
            assert any("Non-numeric rating 'invalid'" in e for e in errors)

    def test_invalid_intent_fails_validation(self):
        """Human intent not belonging to the 12 taxonomy intents must fail."""
        df = _make_valid_50_df()
        df.loc[0, "human_intent"] = "unknown_bug_topic"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            is_valid, errors = validate_human_annotations(f.name)
            assert is_valid is False
            assert any("Invalid human_intent 'unknown_bug_topic'" in e for e in errors)

    def test_duplicate_case_ids_fail_validation(self):
        """Duplicate case_id entries must fail validation."""
        df = _make_valid_50_df()
        df.loc[1, "case_id"] = "case_01"  # Duplicate case_01
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            is_valid, errors = validate_human_annotations(f.name)
            assert is_valid is False
            assert any("Duplicate case_id found" in e for e in errors)

    def test_missing_overall_comment_fails_validation(self):
        """Missing or empty overall comments must fail validation."""
        df = _make_valid_50_df()
        df.loc[0, "overall_comment"] = ""
        df.loc[1, "overall_comment"] = np.nan
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            is_valid, errors = validate_human_annotations(f.name)
            assert is_valid is False
            assert any("Missing overall_comment" in e for e in errors)

    def test_incomplete_annotations_fails_validation(self):
        """A file with fewer than 50 cases must fail validation."""
        df = _make_valid_50_df().iloc[:45]  # Only 45 cases
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            is_valid, errors = validate_human_annotations(f.name)
            assert is_valid is False
            assert any("Expected exactly 50 cases, but found 45" in e for e in errors)

    def test_invalid_intent_correct_and_evidence_supported_values(self):
        """Fields expecting YES/NO must reject other strings."""
        df = _make_valid_50_df()
        df.loc[0, "intent_correct"] = "MAYBE"
        df.loc[1, "evidence_supported"] = "UNKNOWN"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            is_valid, errors = validate_human_annotations(f.name)
            assert is_valid is False
            assert any("Invalid intent_correct 'MAYBE'" in e for e in errors)
            assert any("Invalid evidence_supported 'UNKNOWN'" in e for e in errors)


class TestHumanAgreementSafetyAndCalculation:
    """Tests for human_agreement.py safety invariants and statistics."""

    def test_no_fake_agreement_when_ratings_are_absent(self):
        """When human ratings are empty, status must be NOT_YET_VERIFIED and zero agreement calculated."""
        # The default human_evaluation_template.csv is unannotated
        result = run_human_agreement(
            human_ratings_path="golden_set/human_evaluation_template.csv",
        )
        assert result["status"] == "NOT_YET_VERIFIED"
        assert "incomplete" in result["reason"].lower()

    def test_check_human_rating_status_lifecycle(self):
        """Verifies check_human_rating_status identifies NOT_YET_VERIFIED, INCOMPLETE, and VERIFIED."""
        df = _make_valid_50_df()

        # Complete
        status, completed, total = check_human_rating_status(df)
        assert status == "VERIFIED"
        assert completed == 50

        # Incomplete (only 20 rated)
        df_incomplete = df.copy()
        for col in RATING_DIMENSIONS_NEW:
            df_incomplete.loc[20:, col] = np.nan
        status, completed, total = check_human_rating_status(df_incomplete)
        assert status == "INCOMPLETE"
        assert completed == 20

        # Empty
        df_empty = df.copy()
        for col in RATING_DIMENSIONS_NEW:
            df_empty[col] = np.nan
        status, completed, total = check_human_rating_status(df_empty)
        assert status == "NOT_YET_VERIFIED"
        assert completed == 0

    def test_agreement_calculation_with_valid_paired_cases(self):
        """When 50 complete valid human ratings exist, statistical agreement is computed correctly."""
        df_human = _make_valid_50_df()

        # Mock judge DataFrame with 50 cases
        judge_rows = []
        for i in range(50):
            judge_rows.append({
                "tweet_id": 1000 + i,
                "case_id": f"case_{i + 1:02d}",
                "customer_message": f"Sample message {i + 1}",
                "predicted_intent": df_human.loc[i, "human_intent"],
                "escalation_decision": df_human.loc[i, "system_decision"],
                "correctness_score": 4,
                "groundedness_score": 4,
                "relevance_score": 4,
                "completeness_score": 3,
                "tone_score": 5,
                "unsupported_claims_score": 5,
                "escalation_appropriateness_score": 5,
            })
        df_judge = pd.DataFrame(judge_rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            h_path = Path(tmpdir) / "human.csv"
            j_path = Path(tmpdir) / "judge.csv"
            sum_path = Path(tmpdir) / "summary.csv"
            rep_path = Path(tmpdir) / "report.md"
            res_path = Path(tmpdir) / "results.csv"

            df_human.to_csv(h_path, index=False)
            df_judge.to_csv(j_path, index=False)

            res = run_human_agreement(
                human_ratings_path=str(h_path),
                judge_results_path=str(j_path),
                summary_output_path=str(sum_path),
                report_output_path=str(rep_path),
                results_output_path=str(res_path),
                cases_path=str(h_path),
            )

            assert res["status"] == "VERIFIED"
            assert res["paired_cases"] == 50
            assert sum_path.exists()
            assert rep_path.exists()
            assert res_path.exists()

            df_summary = pd.read_csv(sum_path)
            assert len(df_summary) == 7
            assert "correctness" in df_summary["dimension"].values
            assert "escalation_appropriateness" in df_summary["dimension"].values

            # Correctness mean should be 4.0
            corr_row = df_summary[df_summary["dimension"] == "correctness"].iloc[0]
            assert float(corr_row["human_mean"]) == 4.0
            assert float(corr_row["llm_mean"]) == 4.0
            assert float(corr_row["mean_absolute_difference"]) == 0.0

    def test_actual_human_evaluation_50_agreement_pipeline(self):
        """Tests the agreement pipeline with the actual golden_set/human_evaluation_50.csv."""
        actual_human_path = Path(__file__).resolve().parents[1] / "golden_set" / "human_evaluation_50.csv"
        actual_judge_path = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "llm_judge_results.csv"

        if not actual_human_path.exists() or not actual_judge_path.exists():
            raise unittest.SkipTest("Actual human evaluation or LLM judge results not found.")

        with tempfile.TemporaryDirectory() as tmpdir:
            sum_path = Path(tmpdir) / "agreement_summary.csv"
            rep_path = Path(tmpdir) / "agreement_report.md"
            cases_path = Path(tmpdir) / "agreement_cases.csv"

            res = run_human_agreement(
                human_ratings_path=str(actual_human_path),
                judge_results_path=str(actual_judge_path),
                summary_output_path=str(sum_path),
                report_output_path=str(rep_path),
                results_output_path=str(cases_path),
            )

            assert res["status"] == "VERIFIED"
            # Exactly 6 cases succeeded in current LLM judge run
            assert res["paired_cases"] == 6
            assert sum_path.exists()
            assert rep_path.exists()

            df_cases_out = Path(PROJECT_ROOT) / "evaluation" / "results" / "human_llm_agreement_cases.csv"
            assert df_cases_out.exists()
            df_cases = pd.read_csv(df_cases_out)
            assert len(df_cases) == 6
            assert "correctness_diff" in df_cases.columns
            assert "max_abs_diff" in df_cases.columns
            assert "overall_comment" in df_cases.columns
            assert "judge_reason" in df_cases.columns


if __name__ == "__main__":
    print("Run tests with: .\\venv\\Scripts\\python.exe -m pytest tests/test_human_evaluation_validation.py -v")

