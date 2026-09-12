"""
src/evaluation/test_human_agreement.py

Unit tests for Human-vs-LLM agreement pipeline.
Verifies all 3 lifecycle states:
1. Empty human ratings (0/50) -> NOT_YET_VERIFIED
2. Partial human ratings (1-49/50) -> INCOMPLETE
3. Complete human ratings (50/50) -> VERIFIED
"""

import tempfile
import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.human_agreement import (
    RATING_DIMENSIONS,
    check_human_rating_status,
    compute_dimension_agreement,
    run_human_agreement,
)
from src.evaluation.validate_human_annotations import validate_annotation_file


class TestHumanAgreement(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create dummy judge results
        judge_rows = []
        for i in range(50):
            judge_rows.append({
                "tweet_id": 1000 + i,
                "evaluation_id": f"eval_{i+1:02d}",
                "customer_message": f"Message {i}",
                "generated_reply": f"Reply {i}",
                "gold_intent": "ios_update",
                "predicted_intent": "ios_update",
                "evidence_used": True,
                "correctness_score": 3,
                "groundedness_score": 4,
                "relevance_score": 3,
                "completeness_score": 2,
                "tone_score": 4,
                "unsupported_claims_score": 5,
                "overall_score": 3.5,
                "judge_decision": "NEEDS_IMPROVEMENT",
                "judge_reason": "Safe but generic",
            })
        self.df_judge = pd.DataFrame(judge_rows)
        self.judge_path = self.dir_path / "judge_results.csv"
        self.df_judge.to_csv(self.judge_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_empty_human_ratings_status(self):
        """Test Case A: All human ratings are empty/NaN."""
        df_empty = self.df_judge.copy()
        for _, h_col, j_col in RATING_DIMENSIONS:
            df_empty[h_col] = np.nan
            df_empty[j_col] = df_empty["correctness_score"]

        status, completed, total = check_human_rating_status(df_empty)
        self.assertEqual(status, "NOT_YET_VERIFIED")
        self.assertEqual(completed, 0)
        self.assertEqual(total, 50)

    def test_02_partial_human_ratings_status(self):
        """Test Case B: Only a subset of rows (e.g. 20/50) are rated."""
        df_partial = self.df_judge.copy()
        for _, h_col, j_col in RATING_DIMENSIONS:
            df_partial[h_col] = np.nan
            df_partial[j_col] = df_partial["correctness_score"]

        # Populate first 20 rows
        for i in range(20):
            for _, h_col, _ in RATING_DIMENSIONS:
                df_partial.at[i, h_col] = 4

        status, completed, total = check_human_rating_status(df_partial)
        self.assertEqual(status, "INCOMPLETE")
        self.assertEqual(completed, 20)
        self.assertEqual(total, 50)

    def test_03_complete_human_ratings_status(self):
        """Test Case C: All 50 rows are rated with valid 1-5 scores."""
        df_complete = self.df_judge.copy()
        for _, h_col, j_col in RATING_DIMENSIONS:
            df_complete[h_col] = 4
            df_complete[j_col] = df_complete["correctness_score"]

        status, completed, total = check_human_rating_status(df_complete)
        self.assertEqual(status, "VERIFIED")
        self.assertEqual(completed, 50)
        self.assertEqual(total, 50)

    def test_04_dimension_agreement_computation(self):
        """Test statistical agreement computation on known vector inputs."""
        human = pd.Series([1, 2, 3, 4, 5])
        judge = pd.Series([1, 2, 3, 4, 5])  # Perfect agreement

        metrics = compute_dimension_agreement(human, judge)
        self.assertEqual(metrics["exact_agreement"], 1.0)
        self.assertEqual(metrics["within_1_point_agreement"], 1.0)
        self.assertEqual(metrics["mean_absolute_difference"], 0.0)
        self.assertEqual(metrics["spearman_correlation"], "1.0000")
        self.assertEqual(metrics["weighted_cohen_kappa"], "1.0000")

    def test_05_dimension_agreement_with_offset(self):
        """Test agreement when scores differ by exactly 1 point."""
        human = pd.Series([2, 3, 4, 5, 5])
        judge = pd.Series([1, 2, 3, 4, 4])

        metrics = compute_dimension_agreement(human, judge)
        self.assertEqual(metrics["exact_agreement"], 0.0)
        self.assertEqual(metrics["within_1_point_agreement"], 1.0)
        self.assertEqual(metrics["mean_absolute_difference"], 1.0)

    def test_06_run_pipeline_empty_ratings(self):
        """Test full pipeline execution when ratings file is empty."""
        human_path = self.dir_path / "human_empty.csv"
        blind_path = self.dir_path / "blind_empty.csv"
        comp_path = self.dir_path / "comp_out.csv"
        sum_path = self.dir_path / "sum_out.csv"

        df_empty = self.df_judge.copy()
        for _, h_col, j_col in RATING_DIMENSIONS:
            df_empty[h_col] = np.nan
            df_empty[j_col] = df_empty["correctness_score"]

        df_empty.to_csv(human_path, index=False)
        df_empty.to_csv(blind_path, index=False)

        res = run_human_agreement(
            judge_results_path=str(self.judge_path),
            human_ratings_path=str(human_path),
            blind_ratings_path=str(blind_path),
            comparison_output_path=str(comp_path),
            summary_output_path=str(sum_path),
        )
        self.assertEqual(res["status"], "NOT_YET_VERIFIED")
        self.assertEqual(res["completed"], 0)

        # Check that output CSV has NOT_YET_VERIFIED status
        sum_df = pd.read_csv(sum_path)
        self.assertTrue((sum_df["status"] == "NOT_YET_VERIFIED").all())

    def test_07_run_pipeline_partial_ratings(self):
        """Test full pipeline execution when ratings file has partial ratings."""
        human_path = self.dir_path / "human_partial.csv"
        blind_path = self.dir_path / "blind_partial.csv"
        comp_path = self.dir_path / "comp_out.csv"
        sum_path = self.dir_path / "sum_out.csv"

        df_partial = self.df_judge.copy()
        for _, h_col, j_col in RATING_DIMENSIONS:
            df_partial[h_col] = np.nan
            df_partial[j_col] = df_partial["correctness_score"]

        # Fill 15 rows
        for i in range(15):
            for _, h_col, _ in RATING_DIMENSIONS:
                df_partial.at[i, h_col] = 3

        df_partial.to_csv(human_path, index=False)
        df_partial.to_csv(blind_path, index=False)

        res = run_human_agreement(
            judge_results_path=str(self.judge_path),
            human_ratings_path=str(human_path),
            blind_ratings_path=str(blind_path),
            comparison_output_path=str(comp_path),
            summary_output_path=str(sum_path),
        )
        self.assertEqual(res["status"], "INCOMPLETE")
        self.assertEqual(res["completed"], 15)

        sum_df = pd.read_csv(sum_path)
        self.assertTrue((sum_df["status"] == "INCOMPLETE").all())

    def test_08_run_pipeline_complete_ratings(self):
        """Test full pipeline execution when ratings file has complete ratings."""
        human_path = self.dir_path / "human_complete.csv"
        blind_path = self.dir_path / "blind_complete.csv"
        comp_path = self.dir_path / "comp_out.csv"
        sum_path = self.dir_path / "sum_out.csv"

        df_complete = self.df_judge.copy()
        for idx, row in df_complete.iterrows():
            for _, h_col, j_col in RATING_DIMENSIONS:
                # Slight variation
                score = (idx % 5) + 1
                df_complete.at[idx, h_col] = score
                df_complete.at[idx, j_col] = max(1, min(5, score + (1 if idx % 2 == 0 else -1)))

        df_complete.to_csv(human_path, index=False)
        df_complete.to_csv(blind_path, index=False)

        res = run_human_agreement(
            judge_results_path=str(self.judge_path),
            human_ratings_path=str(human_path),
            blind_ratings_path=str(blind_path),
            comparison_output_path=str(comp_path),
            summary_output_path=str(sum_path),
        )

        self.assertEqual(res["status"], "VERIFIED")
        self.assertEqual(res["completed"], 50)

        sum_df = pd.read_csv(sum_path)
        self.assertTrue((sum_df["status"] == "VERIFIED").all())
        self.assertEqual(len(sum_df), 7)  # 6 dimensions + 1 overall

    def test_09_validate_annotation_file_empty(self):
        """Test validator on completely unrated form (no annotation_source)."""
        form_path = self.dir_path / "form_empty.csv"
        df_form = pd.DataFrame([{
            "evaluation_id": f"eval_{i+1:02d}",
            "customer_message": f"Message {i}",
            "expected_intent": "ios_update",
            "generated_reply": f"Reply {i}",
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "",
            "evidence_text": "",
            # No annotation_source column -> triggers scientific integrity check
            "human_correctness": np.nan,
            "human_groundedness": np.nan,
            "human_relevance": np.nan,
            "human_completeness": np.nan,
            "human_tone": np.nan,
            "human_unsupported_claims": np.nan,
            "human_notes": np.nan,
        } for i in range(50)])
        df_form.to_csv(form_path, index=False)

        status, details = validate_annotation_file(str(form_path))
        self.assertEqual(status, "NOT_YET_VERIFIED")
        # Missing annotation_source triggers early return with issues key only
        self.assertIn("issues", details)
        self.assertTrue(len(details["issues"]) > 0)

    def test_10_validate_annotation_file_complete(self):
        """Test validator on completely and validly rated form with annotation_source='human'."""
        form_path = self.dir_path / "form_complete.csv"
        df_form = pd.DataFrame([{
            "evaluation_id": f"eval_{i+1:02d}",
            "annotation_source": "human",  # Required: marks genuine human annotation
            "customer_message": f"Message {i}",
            "expected_intent": "ios_update",
            "generated_reply": f"Reply {i}",
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "",
            "evidence_text": "",
            "human_correctness": (i % 5) + 1,
            "human_groundedness": 5,
            "human_relevance": (i % 5) + 1,
            "human_completeness": 4,
            "human_tone": 5,
            "human_unsupported_claims": 5,
            "human_notes": "Note",
        } for i in range(50)])
        df_form.to_csv(form_path, index=False)

        status, details = validate_annotation_file(str(form_path))
        self.assertEqual(status, "VERIFIED")
        self.assertEqual(details["complete_count"], 50)

    def test_11_validate_annotation_file_invalid_scores(self):
        """Test validator catches scores out of range (e.g. 6 or string)."""
        form_path = self.dir_path / "form_invalid.csv"
        df_form = pd.DataFrame([{
            "evaluation_id": f"eval_{i+1:02d}",
            "annotation_source": "human",  # Must pass source check so validator reaches score check
            "customer_message": f"Message {i}",
            "expected_intent": "ios_update",
            "generated_reply": f"Reply {i}",
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "",
            "evidence_text": "",
            "human_correctness": 6 if i == 0 else 4,  # Invalid 6
            "human_groundedness": 5,
            "human_relevance": 4,
            "human_completeness": 4,
            "human_tone": 5,
            "human_unsupported_claims": 5,
            "human_notes": "",
        } for i in range(50)])
        df_form.to_csv(form_path, index=False)

        status, details = validate_annotation_file(str(form_path))
        self.assertEqual(status, "NOT_YET_VERIFIED")
        self.assertTrue(len(details["invalid_rows"]) > 0)

    def test_12_validate_annotation_file_missing_and_duplicate_ids(self):
        """Test validator catches missing cases and duplicate IDs."""
        form_path = self.dir_path / "form_dup.csv"
        df_form = pd.DataFrame([{
            "evaluation_id": "eval_01" if i == 1 else f"eval_{i+1:02d}",  # Duplicate eval_01
            "customer_message": f"Message {i}",
            "expected_intent": "ios_update",
            "generated_reply": f"Reply {i}",
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "",
            "evidence_text": "",
            "human_correctness": 4,
            "human_groundedness": 5,
            "human_relevance": 4,
            "human_completeness": 4,
            "human_tone": 5,
            "human_unsupported_claims": 5,
            "human_notes": "",
        } for i in range(50)])
        df_form.to_csv(form_path, index=False)

        status, details = validate_annotation_file(str(form_path))
        self.assertEqual(status, "NOT_YET_VERIFIED")
        # annotation_source missing -> early return, issues contains the annotation_source error
        self.assertIn("issues", details)
        self.assertTrue(len(details["issues"]) > 0)


if __name__ == "__main__":
    unittest.main()
