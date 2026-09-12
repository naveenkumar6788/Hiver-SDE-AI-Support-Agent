"""
src/evaluation/human_agreement.py

Human-vs-LLM Judge Agreement Evaluation for AppleSupport Customer Support Responses.

Evaluates 50 representative cases from the golden evaluation set.
Strict validation behavior:
1. Empty human ratings (0/50):
   Status -> HUMAN AGREEMENT: NOT YET VERIFIED
   Reason: No independent human response-quality ratings are available.
2. Partial human ratings (1-49/50):
   Status -> HUMAN AGREEMENT: INCOMPLETE
   Reason: Agreement calculation withheld until the required human evaluation set is complete.
3. Complete human ratings (50/50):
   Status -> HUMAN AGREEMENT: VERIFIED
   Computes exact agreement, agreement within ±1 point, mean absolute difference,
   Spearman correlation, Pearson correlation, and quadratic weighted Cohen's kappa.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RATING_DIMENSIONS = [
    ("correctness", "human_correctness", "llm_judge_correctness"),
    ("groundedness", "human_groundedness", "llm_judge_groundedness"),
    ("relevance", "human_relevance", "llm_judge_relevance"),
    ("completeness", "human_completeness", "llm_judge_completeness"),
    ("tone", "human_tone", "llm_judge_tone"),
    ("unsupported_claims", "human_unsupported_claims", "llm_judge_unsupported_claims"),
]


def check_human_rating_status(df: pd.DataFrame) -> Tuple[str, int, int]:
    """
    Checks human rating completion status.
    Returns (status_label, completed_count, total_count).
    Status label is one of: 'NOT_YET_VERIFIED', 'INCOMPLETE', 'VERIFIED'
    """
    human_cols = [h_col for _, h_col, _ in RATING_DIMENSIONS]
    total_cases = len(df)

    # Count rows where all human dimensions are non-null and valid (1-5)
    valid_mask = pd.Series(True, index=df.index)
    for col in human_cols:
        if col not in df.columns:
            return "NOT_YET_VERIFIED", 0, total_cases
        valid_mask = valid_mask & df[col].notna() & df[col].between(1, 5)

    completed_count = int(valid_mask.sum())

    if completed_count == 0:
        return "NOT_YET_VERIFIED", 0, total_cases
    elif completed_count < total_cases:
        return "INCOMPLETE", completed_count, total_cases
    else:
        return "VERIFIED", completed_count, total_cases


def compute_dimension_agreement(human_series: pd.Series, judge_series: pd.Series) -> Dict[str, Any]:
    """Computes statistical agreement metrics between human and judge ratings."""
    n = len(human_series)
    if n == 0:
        return {
            "sample_size": 0,
            "mean_human_score": "N/A",
            "mean_judge_score": "N/A",
            "exact_agreement": "N/A",
            "within_1_point_agreement": "N/A",
            "mean_absolute_difference": "N/A",
            "spearman_correlation": "N/A",
            "pearson_correlation": "N/A",
            "weighted_cohen_kappa": "N/A",
        }

    h = human_series.to_numpy(dtype=float)
    j = judge_series.to_numpy(dtype=float)

    mean_h = float(np.mean(h))
    mean_j = float(np.mean(j))

    exact = float(np.mean(h == j))
    within_1 = float(np.mean(np.abs(h - j) <= 1.0))
    mad = float(np.mean(np.abs(h - j)))

    # Spearman correlation
    if np.std(h) > 1e-6 and np.std(j) > 1e-6:
        rho, _ = spearmanr(h, j)
        r, _ = pearsonr(h, j)
        rho_str = f"{rho:.4f}" if not np.isnan(rho) else "N/A"
        r_str = f"{r:.4f}" if not np.isnan(r) else "N/A"
    else:
        rho_str = "N/A (zero variance)"
        r_str = "N/A (zero variance)"

    # Weighted Cohen's kappa (quadratic)
    try:
        # Map to integer categories 1-5
        h_int = h.astype(int)
        j_int = j.astype(int)
        kappa = cohen_kappa_score(h_int, j_int, labels=[1, 2, 3, 4, 5], weights="quadratic")
        kappa_str = f"{kappa:.4f}" if not np.isnan(kappa) else "N/A"
    except Exception:
        kappa_str = "N/A"

    return {
        "sample_size": n,
        "mean_human_score": round(mean_h, 4),
        "mean_judge_score": round(mean_j, 4),
        "exact_agreement": round(exact, 4),
        "within_1_point_agreement": round(within_1, 4),
        "mean_absolute_difference": round(mad, 4),
        "spearman_correlation": rho_str,
        "pearson_correlation": r_str,
        "weighted_cohen_kappa": kappa_str,
    }


def run_human_agreement(
    judge_results_path: str = "evaluation/results/judge_results.csv",
    human_ratings_path: str = "golden_set/human_judge_50.csv",
    blind_ratings_path: str = "golden_set/human_judge_50_blind.csv",
    form_ratings_path: str = "golden_set/human_annotation_form.csv",
    comparison_output_path: str = "evaluation/results/human_judge_comparison.csv",
    summary_output_path: str = "evaluation/results/human_agreement_summary.csv",
) -> Dict[str, Any]:
    print("=" * 80)
    print("HUMAN-VS-LLM JUDGE AGREEMENT EVALUATION")
    print("=" * 80)

    human_file = PROJECT_ROOT / human_ratings_path
    blind_file = PROJECT_ROOT / blind_ratings_path
    form_file = PROJECT_ROOT / form_ratings_path
    judge_file = PROJECT_ROOT / judge_results_path

    if not judge_file.exists():
        raise FileNotFoundError(f"Judge results file not found: {judge_file}")

    if not human_file.exists():
        raise FileNotFoundError(f"Human evaluation file not found: {human_file}")

    df_human = pd.read_csv(human_file)

    # If form file exists and has ratings, sync them in (only when using default human_ratings_path)
    if human_ratings_path == "golden_set/human_judge_50.csv" and form_file.exists():
        df_form = pd.read_csv(form_file)

        # SCIENTIFIC INTEGRITY CHECK: only sync if annotation_source is 'human'.
        # Structural CSV validity (valid 1-5 integers) does NOT prove human authorship.
        form_source_ok = False
        if "annotation_source" in df_form.columns:
            non_human_rows = df_form[df_form["annotation_source"] != "human"]
            form_source_ok = len(non_human_rows) == 0

        if form_source_ok:
            for _, h_col, _ in RATING_DIMENSIONS:
                if h_col in df_form.columns and df_form[h_col].notna().any():
                    df_human[h_col] = df_form[h_col]
        else:
            source_vals = (
                df_form["annotation_source"].unique().tolist()
                if "annotation_source" in df_form.columns
                else ["<missing>"]
            )
            print("\n[SCIENTIFIC INTEGRITY] annotation_source in human_annotation_form.csv is not 'human'.")
            print(f"  Found annotation_source values: {source_vals}")
            print("  Ratings present but NOT genuine independent human annotations.")
            print("  Sync blocked. Agreement calculation withheld.\n")
            # Do NOT sync — leave human columns as NaN (will produce NOT_YET_VERIFIED)

    # If blind file exists and has ratings, sync them in (check source too)
    if blind_file.exists():
        df_blind = pd.read_csv(blind_file)
        blind_source_ok = True
        if "annotation_source" in df_blind.columns:
            non_human_blind = df_blind[df_blind["annotation_source"] != "human"]
            blind_source_ok = len(non_human_blind) == 0
        if blind_source_ok:
            for _, h_col, _ in RATING_DIMENSIONS:
                if h_col in df_blind.columns and df_blind[h_col].notna().any():
                    df_human[h_col] = df_blind[h_col]

    status, completed_count, total_count = check_human_rating_status(df_human)

    # -------------------------------------------------------------
    # CASE A: No genuine human ratings
    # -------------------------------------------------------------
    if status == "NOT_YET_VERIFIED":
        print("\nHUMAN AGREEMENT: NOT YET VERIFIED")
        print("Reason: No independent human response-quality ratings are available.")
        print("Agreement calculation withheld in accordance with scientific integrity rules.\n")

        # Save template comparison file
        comp_records = []
        for _, row in df_human.iterrows():
            rec = {
                "evaluation_id": row.get("evaluation_id", ""),
                "tweet_id": row.get("tweet_id", ""),
                "customer_message": row.get("customer_message", ""),
                "generated_reply": row.get("generated_reply", ""),
                "evidence_used": row.get("evidence_used", False),
            }
            for dim_name, h_col, j_col in RATING_DIMENSIONS:
                rec[h_col] = np.nan
                rec[j_col] = row.get(j_col, np.nan)
            rec["status"] = "AWAITING_HUMAN_ANNOTATION"
            comp_records.append(rec)

        df_comp = pd.DataFrame(comp_records)
        df_comp.to_csv(PROJECT_ROOT / comparison_output_path, index=False)

        # Save summary marking NOT_YET_VERIFIED
        summary_records = []
        for dim_name, _, _ in RATING_DIMENSIONS:
            summary_records.append({
                "dimension": dim_name,
                "sample_size": 0,
                "mean_human_score": "N/A",
                "mean_judge_score": "N/A",
                "exact_agreement": "N/A",
                "within_1_point_agreement": "N/A",
                "mean_absolute_difference": "N/A",
                "spearman_correlation": "N/A",
                "pearson_correlation": "N/A",
                "weighted_cohen_kappa": "N/A",
                "status": "NOT_YET_VERIFIED",
            })
        # Overall
        summary_records.append({
            "dimension": "overall",
            "sample_size": 0,
            "mean_human_score": "N/A",
            "mean_judge_score": "N/A",
            "exact_agreement": "N/A",
            "within_1_point_agreement": "N/A",
            "mean_absolute_difference": "N/A",
            "spearman_correlation": "N/A",
            "pearson_correlation": "N/A",
            "weighted_cohen_kappa": "N/A",
            "status": "NOT_YET_VERIFIED",
        })

        df_sum = pd.DataFrame(summary_records)
        df_sum.to_csv(PROJECT_ROOT / summary_output_path, index=False)
        print(f"Updated {summary_output_path} with status NOT_YET_VERIFIED.")
        return {"status": "NOT_YET_VERIFIED", "completed": 0, "total": total_count}

    # -------------------------------------------------------------
    # CASE B: Partial human ratings
    # -------------------------------------------------------------
    elif status == "INCOMPLETE":
        print(f"\nHUMAN AGREEMENT: INCOMPLETE")
        print(f"Human-rated cases: {completed_count}/{total_count}")
        print("Agreement calculation withheld until the required human evaluation set is complete.\n")

        summary_records = []
        for dim_name, _, _ in RATING_DIMENSIONS:
            summary_records.append({
                "dimension": dim_name,
                "sample_size": completed_count,
                "mean_human_score": "N/A",
                "mean_judge_score": "N/A",
                "exact_agreement": "N/A",
                "within_1_point_agreement": "N/A",
                "mean_absolute_difference": "N/A",
                "spearman_correlation": "N/A",
                "pearson_correlation": "N/A",
                "weighted_cohen_kappa": "N/A",
                "status": "INCOMPLETE",
            })
        summary_records.append({
            "dimension": "overall",
            "sample_size": completed_count,
            "mean_human_score": "N/A",
            "mean_judge_score": "N/A",
            "exact_agreement": "N/A",
            "within_1_point_agreement": "N/A",
            "mean_absolute_difference": "N/A",
            "spearman_correlation": "N/A",
            "pearson_correlation": "N/A",
            "weighted_cohen_kappa": "N/A",
            "status": "INCOMPLETE",
        })

        df_sum = pd.DataFrame(summary_records)
        df_sum.to_csv(PROJECT_ROOT / summary_output_path, index=False)
        print(f"Updated {summary_output_path} with status INCOMPLETE.")
        return {"status": "INCOMPLETE", "completed": completed_count, "total": total_count}

    # -------------------------------------------------------------
    # CASE C: Complete genuine human ratings (50/50)
    # -------------------------------------------------------------
    else:
        print(f"\nHUMAN AGREEMENT: VERIFIED")
        print(f"Evaluating agreement across all {completed_count}/{total_count} human-rated cases.\n")

        summary_records = []
        all_exact = []
        all_within_1 = []
        all_mad = []
        all_human_means = []
        all_judge_means = []

        # Calculate agreement per dimension
        for dim_name, h_col, j_col in RATING_DIMENSIONS:
            h_vals = df_human[h_col].astype(float)
            j_vals = df_human[j_col].astype(float)

            dim_metrics = compute_dimension_agreement(h_vals, j_vals)
            dim_metrics["dimension"] = dim_name
            dim_metrics["status"] = "VERIFIED"
            summary_records.append(dim_metrics)

            all_exact.append(dim_metrics["exact_agreement"])
            all_within_1.append(dim_metrics["within_1_point_agreement"])
            all_mad.append(dim_metrics["mean_absolute_difference"])
            all_human_means.append(dim_metrics["mean_human_score"])
            all_judge_means.append(dim_metrics["mean_judge_score"])

        # Overall across all dimensions
        overall_metrics = {
            "dimension": "overall",
            "sample_size": total_count,
            "mean_human_score": round(float(np.mean(all_human_means)), 4),
            "mean_judge_score": round(float(np.mean(all_judge_means)), 4),
            "exact_agreement": round(float(np.mean(all_exact)), 4),
            "within_1_point_agreement": round(float(np.mean(all_within_1)), 4),
            "mean_absolute_difference": round(float(np.mean(all_mad)), 4),
            "spearman_correlation": "N/A (aggregated)",
            "pearson_correlation": "N/A (aggregated)",
            "weighted_cohen_kappa": "N/A (aggregated)",
            "status": "VERIFIED",
        }
        summary_records.append(overall_metrics)

        df_sum = pd.DataFrame(summary_records)
        df_sum.to_csv(PROJECT_ROOT / summary_output_path, index=False)

        # Also write comparison output
        df_human.to_csv(PROJECT_ROOT / comparison_output_path, index=False)

        # Print summary table
        print("-" * 80)
        print(f"{'Dimension':<20} | {'Mean Human':<10} | {'Mean LLM':<10} | {'Exact':<8} | {'Within ±1':<10} | {'Spearman':<10}")
        print("-" * 80)
        for r in summary_records[:-1]:
            print(f"{r['dimension']:<20} | {str(r['mean_human_score']):<10} | {str(r['mean_judge_score']):<10} | {str(r['exact_agreement']):<8} | {str(r['within_1_point_agreement']):<10} | {str(r['spearman_correlation']):<10}")
        print("-" * 80)
        print(f"{'OVERALL':<20} | {str(overall_metrics['mean_human_score']):<10} | {str(overall_metrics['mean_judge_score']):<10} | {str(overall_metrics['exact_agreement']):<8} | {str(overall_metrics['within_1_point_agreement']):<10} | {'N/A':<10}")
        print("-" * 80)

        return {"status": "VERIFIED", "completed": completed_count, "total": total_count, "summary": summary_records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human vs LLM Judge Agreement Evaluation.")
    parser.add_argument("--judge_results", default="evaluation/results/judge_results.csv")
    parser.add_argument("--human_ratings", default="golden_set/human_judge_50.csv")
    parser.add_argument("--blind_ratings", default="golden_set/human_judge_50_blind.csv")
    parser.add_argument("--form_ratings", default="golden_set/human_annotation_form.csv")
    parser.add_argument("--output_comparison", default="evaluation/results/human_judge_comparison.csv")
    parser.add_argument("--output_summary", default="evaluation/results/human_agreement_summary.csv")

    args = parser.parse_args()
    run_human_agreement(
        judge_results_path=args.judge_results,
        human_ratings_path=args.human_ratings,
        blind_ratings_path=args.blind_ratings,
        form_ratings_path=args.form_ratings,
        comparison_output_path=args.output_comparison,
        summary_output_path=args.output_summary,
    )
