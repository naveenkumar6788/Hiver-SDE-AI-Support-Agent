"""
CLI Runner for LLM-as-a-Judge evaluation on generated customer support replies.

Reads:
    evaluation/results/reply_generation_full_results.csv

Generates:
    evaluation/results/llm_judge_results.csv
    evaluation/results/llm_judge_summary.csv

Features:
    - Default evaluation limit: 50 examples
    - Configurable delay between API calls
    - Saves progress after every example
    - Safer Groq rate-limit handling
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

try:
    from dotenv import load_dotenv

    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        load_dotenv(dotenv_path=str(env_file), override=False)
    else:
        load_dotenv(override=False)

except ImportError:
    pass


# ============================================================
# IMPORT JUDGE
# ============================================================

from src.evaluation.llm_judge import LLMJudge


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge on generated customer support replies."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="evaluation/results/reply_generation_full_results.csv",
        help="Path to reply generation results CSV."
    )

    parser.add_argument(
        "--output_results",
        type=str,
        default="evaluation/results/llm_judge_results.csv",
        help="Path to save detailed judge results."
    )

    parser.add_argument(
        "--output_summary",
        type=str,
        default="evaluation/results/llm_judge_summary.csv",
        help="Path to save judge summary."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of examples to judge. Default: 50."
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds between API calls. Default: 5.0 seconds."
    )

    parser.add_argument(
        "--max-retries",
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts per API call on rate limit or network error. Default: 3."
    )

    parser.add_argument(
        "--max-retry-wait",
        "--max_retry_wait",
        type=float,
        default=30.0,
        help="Maximum seconds to wait per retry on rate limit backoff. Default: 30.0 seconds."
    )

    parser.add_argument(
        "--retry-failed",
        "--retry_failed",
        action="store_true",
        help="Only retry examples that previously failed or were rate-limited in existing results CSV."
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider. If omitted, reads LLM_PROVIDER."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model. If omitted, reads LLM_MODEL."
    )

    return parser.parse_args()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def deduplicate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicates results DataFrame by tweet_id while preserving successful results
    (judge_success == True and judge_status == 'success') and retaining processing order.
    """
    if df.empty or "tweet_id" not in df.columns:
        return df

    str_ids = df["tweet_id"].astype(str).str.strip()
    if not str_ids.duplicated().any():
        return df

    records = df.to_dict(orient="records")
    best_by_id: Dict[str, Dict[str, Any]] = {}
    id_order: List[str] = []

    for rec in records:
        tid = str(rec.get("tweet_id", "")).strip()
        if tid not in best_by_id:
            best_by_id[tid] = rec
            id_order.append(tid)
        else:
            curr_success = bool(best_by_id[tid].get("judge_success", False)) and (
                str(best_by_id[tid].get("judge_status", "")).lower() == "success"
            )
            new_success = bool(rec.get("judge_success", False)) and (
                str(rec.get("judge_status", "")).lower() == "success"
            )
            # If the new record is successful and existing was not, prioritize the success
            if new_success and not curr_success:
                best_by_id[tid] = rec
            elif not curr_success and not new_success:
                # Keep latest failed record
                best_by_id[tid] = rec

    deduped_records = [best_by_id[tid] for tid in id_order]
    return pd.DataFrame(deduped_records)


def save_current_progress(
    results: List[Dict[str, Any]],
    existing_successful_results: Dict[str, Dict[str, Any]],
    output_path: str,
) -> pd.DataFrame:
    """
    Safely writes current progress to disk, merging any previously successful
    results for unvisited IDs and ensuring no duplicates.
    """
    combined_records = list(results)
    seen_ids = {str(r.get("tweet_id", "")).strip() for r in results}
    for tid, exist_rec in existing_successful_results.items():
        if tid not in seen_ids:
            combined_records.append(exist_rec)

    df_to_save = pd.DataFrame(combined_records)
    df_deduped = deduplicate_results(df_to_save)
    
    # Save atomically
    temp_path = str(output_path) + ".tmp"
    df_deduped.to_csv(temp_path, index=False)
    if os.path.exists(temp_path):
        import shutil
        shutil.move(temp_path, output_path)
    else:
        df_deduped.to_csv(output_path, index=False)

    return df_deduped


def generate_summary(
    results_df: pd.DataFrame,
    judge: LLMJudge,
    output_summary_path: str,
    total_requested: int = 50,
) -> None:
    """
    Generates and saves summary metrics from judge results.
    Distinguishes requested, processed, successful, and failed counts.
    """
    if len(results_df) == 0:
        print("No results generated.", flush=True)
        return

    # A row is successful ONLY when judge_success is True and judge_status == 'success'
    is_success_mask = (results_df["judge_success"] == True)
    if "judge_status" in results_df.columns:
        is_success_mask = is_success_mask & (results_df["judge_status"].astype(str).str.lower() == "success")

    successful_df = results_df[is_success_mask]

    examples_processed = len(results_df)
    successful_judgments = len(successful_df)
    failed_judgments = examples_processed - successful_judgments
    success_rate = round(successful_judgments / examples_processed, 4) if examples_processed > 0 else 0.0

    summary_metrics = [
        ("examples_requested", total_requested),
        ("examples_processed", examples_processed),
        ("successful_judgments", successful_judgments),
        ("failed_judgments", failed_judgments),
        ("success_rate", success_rate),
        # Legacy aliases for backwards compatibility
        ("examples_judged", examples_processed),
        ("judge_failures", failed_judgments),
        ("judge_success_rate", success_rate),
        ("provider", judge.provider_name),
        ("model", judge.model_name),
        (
            "average_correctness",
            round(successful_df["correctness_score"].mean(), 4) if len(successful_df) > 0 else 0.0,
        ),
        (
            "average_groundedness",
            round(successful_df["groundedness_score"].mean(), 4) if len(successful_df) > 0 else 0.0,
        ),
        (
            "average_relevance",
            round(successful_df["relevance_score"].mean(), 4) if len(successful_df) > 0 else 0.0,
        ),
        (
            "average_completeness",
            round(successful_df["completeness_score"].mean(), 4) if len(successful_df) > 0 else 0.0,
        ),
        (
            "average_tone",
            round(successful_df["tone_score"].mean(), 4) if len(successful_df) > 0 else 0.0,
        ),
        (
            "average_unsupported_claims",
            round(successful_df["unsupported_claims_score"].mean(), 4) if len(successful_df) > 0 else 0.0,
        ),
        (
            "average_overall_score",
            round(successful_df["overall_score"].mean(), 4) if len(successful_df) > 0 else 0.0,
        ),
        (
            "average_escalation_appropriateness",
            round(successful_df["escalation_appropriateness_score"].mean(), 4) if len(successful_df) > 0 else 0.0,
        ),
        (
            "unsupported_claims_detected_count",
            int(successful_df["unsupported_claims_found"].sum()) if len(successful_df) > 0 else 0,
        ),
    ]

    if len(successful_df) > 0:
        rounded_overall = successful_df["overall_score"].round().astype(int)
        for score_val in [1, 2, 3, 4, 5]:
            count = (rounded_overall == score_val).sum()
            summary_metrics.append(
                (f"overall_score_dist_{score_val}_star_count", int(count))
            )

    summary_df = pd.DataFrame(summary_metrics, columns=["metric", "value"])
    summary_df.to_csv(output_summary_path, index=False)
    print(f"\nSummary metrics saved to: {output_summary_path}", flush=True)

    print("\n==================================================")
    print("LLM JUDGE SUMMARY")
    print("==================================================")
    for _, summary_row in summary_df.iterrows():
        print(f"{summary_row['metric']:<40}: {summary_row['value']}")
    print("==================================================")

    # Completion check (Requirement 10)
    if successful_judgments == total_requested and failed_judgments == 0:
        print("\n" + "=" * 50)
        print("ALL REQUESTED EXAMPLES SUCCESSFULLY JUDGED")
        print(f"Total Successful: {successful_judgments}/{total_requested}")
        print("=" * 50 + "\n")
    else:
        print("\n" + "=" * 50)
        print("JUDGE INCOMPLETE")
        print(f"Successful: {successful_judgments}/{total_requested}")
        print(f"Failed: {failed_judgments}")
        print("Run --retry-failed to retry failed examples.")
        print("=" * 50 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_path = args.input
    output_results_path = args.output_results
    output_summary_path = args.output_summary

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", flush=True)
        sys.exit(1)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(f"Loading reply generation results from: {input_path}", flush=True)
    df_all = pd.read_csv(input_path)
    total_rows = len(df_all)
    print(f"Total examples available: {total_rows}", flush=True)

    # Determine requested subset
    total_target = total_rows
    if args.limit is not None and args.limit > 0:
        total_target = min(args.limit, total_rows)
        df_target = df_all.head(total_target).copy()
        print(f"Targeting evaluation limit: {total_target} examples.", flush=True)
    elif args.limit == 0:
        print("Error: --limit must be greater than 0.", flush=True)
        sys.exit(1)
    else:
        df_target = df_all.copy()

    # --------------------------------------------------------
    # Initialize judge
    # --------------------------------------------------------

    judge = LLMJudge(
        provider=args.provider,
        model=args.model,
        max_retries=args.max_retries,
        max_retry_wait=args.max_retry_wait
    )

    print(
        f"LLM Judge initialized with provider='{judge.provider_name}', "
        f"model='{judge.model_name}', max_retries={judge.max_retries}, "
        f"max_retry_wait={judge.max_retry_wait}s",
        flush=True
    )

    if judge.provider_name == "mock" and args.provider is None:
        print("[INFO] Operating in offline MOCK mode.", flush=True)

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    output_results_dir = os.path.dirname(output_results_path)
    if output_results_dir:
        os.makedirs(output_results_dir, exist_ok=True)

    output_summary_dir = os.path.dirname(output_summary_path)
    if output_summary_dir:
        os.makedirs(output_summary_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load existing results for resume capability
    # --------------------------------------------------------

    existing_successful_results: Dict[str, Dict[str, Any]] = {}
    existing_failed_results: Dict[str, Dict[str, Any]] = {}

    if os.path.exists(output_results_path):
        try:
            df_existing = pd.read_csv(output_results_path)
            if not df_existing.empty and "tweet_id" in df_existing.columns:
                # Backfill judge_status if missing from older runs
                if "judge_status" not in df_existing.columns:
                    def infer_status(row):
                        if bool(row.get("judge_success", False)):
                            return "success"
                        reason = str(row.get("correctness_reason", "")).lower()
                        if "429" in reason or "rate limit" in reason:
                            return "provider_rate_limit"
                        elif "timeout" in reason or "timed out" in reason:
                            return "timeout"
                        return "provider_error"
                    df_existing["judge_status"] = df_existing.apply(infer_status, axis=1)

                df_existing_deduped = deduplicate_results(df_existing)
                for _, exist_row in df_existing_deduped.iterrows():
                    tid = str(exist_row.get("tweet_id", "")).strip()
                    rec = exist_row.to_dict()
                    is_success = bool(rec.get("judge_success", False)) and (
                        str(rec.get("judge_status", "")).lower() == "success"
                    )
                    if is_success:
                        existing_successful_results[tid] = rec
                    else:
                        existing_failed_results[tid] = rec

                print(
                    f"Found existing results at '{output_results_path}': "
                    f"{len(df_existing_deduped)} total rows "
                    f"({len(existing_successful_results)} successful, {len(existing_failed_results)} failed/rate-limited).",
                    flush=True
                )
        except Exception as e:
            print(
                f"[WARNING] Could not read existing results from {output_results_path}: {e}. "
                "Starting fresh evaluation.",
                flush=True
            )

    # --------------------------------------------------------
    # Handle --retry-failed mode
    # --------------------------------------------------------

    if args.retry_failed:
        if not os.path.exists(output_results_path):
            print(f"Error: Cannot use --retry-failed because '{output_results_path}' does not exist.", flush=True)
            sys.exit(1)

        # Target items needing retry are those in df_target not in existing_successful_results
        target_ids = set(df_target["tweet_id"].astype(str).str.strip())
        needed_retry_mask = df_target["tweet_id"].astype(str).str.strip().apply(
            lambda tid: tid not in existing_successful_results
        )
        df_to_eval = df_target[needed_retry_mask].copy()

        if len(df_to_eval) == 0:
            print(
                f"All {len(target_ids)} targeted examples in '{output_results_path}' "
                "are already successfully judged! Nothing to retry.",
                flush=True
            )
            # Re-generate summary to ensure completion status is printed
            full_saved_df = save_current_progress([], existing_successful_results, output_results_path)
            generate_summary(full_saved_df, judge, output_summary_path, total_requested=total_target)
            return

        print(
            f"\n[RETRY-FAILED MODE] Found {len(df_to_eval)} previously failed / unjudged examples to retry "
            f"out of {total_target} requested.",
            flush=True
        )

    else:
        df_to_eval = df_target

    # --------------------------------------------------------
    # Results accumulator
    # --------------------------------------------------------

    results: List[Dict[str, Any]] = []

    print("\nBeginning evaluation...", flush=True)
    interrupted = False

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    try:
        total_eval = len(df_to_eval)
        for position, (_, row) in enumerate(df_to_eval.iterrows(), start=1):

            tweet_id = row.get("tweet_id", f"sample_{position}")
            tweet_id_str = str(tweet_id).strip()

            # ----------------------------------------------------
            # RESUME CHECK: Skip if already successfully judged
            # ----------------------------------------------------
            if tweet_id_str in existing_successful_results:
                print(f"SKIP already judged: {position}/{total_eval}", flush=True)
                results.append(existing_successful_results[tweet_id_str])
                continue

            customer_msg = str(row.get("customer_message", ""))
            predicted_intent = str(row.get("predicted_intent", ""))
            generated_reply = str(row.get("reply", row.get("generated_reply", "")))
            evidence_text = str(row.get("selected_evidence_text", row.get("evidence", "")))
            if evidence_text == "nan":
                evidence_text = ""
            escalation_decision = str(row.get("escalation_decision", "AUTO_HANDLE"))

            # ----------------------------------------------------
            # Call LLM judge
            # ----------------------------------------------------

            try:
                judge_result = judge.judge(
                    customer_message=customer_msg,
                    predicted_intent=predicted_intent,
                    historical_evidence=evidence_text,
                    generated_reply=generated_reply,
                    escalation_decision=escalation_decision
                )
            except Exception as exc:
                print(f"\nERROR judging example {position}: {exc}", flush=True)
                err_str = str(exc).lower()
                if "429" in err_str or "rate limit" in err_str:
                    j_status = "provider_rate_limit"
                elif "timeout" in err_str or "timed out" in err_str:
                    j_status = "timeout"
                else:
                    j_status = "provider_error"

                res_dict = {
                    "tweet_id": tweet_id,
                    "customer_message": customer_msg,
                    "predicted_intent": predicted_intent,
                    "selected_evidence_text": evidence_text,
                    "generated_reply": generated_reply,
                    "escalation_decision": escalation_decision,
                    "provider": judge.provider_name,
                    "model": judge.model_name,
                    "judge_success": False,
                    "judge_status": j_status,

                    "correctness_score": None,
                    "correctness_reason": str(exc),
                    "groundedness_score": None,
                    "groundedness_reason": str(exc),
                    "relevance_score": None,
                    "relevance_reason": str(exc),
                    "completeness_score": None,
                    "completeness_reason": str(exc),
                    "tone_score": None,
                    "tone_reason": str(exc),
                    "unsupported_claims_score": None,
                    "unsupported_claims_reason": str(exc),
                    "unsupported_claims_found": False,
                    "unsupported_claims_list": "",
                    "escalation_appropriateness_score": None,
                    "escalation_appropriateness_reason": str(exc),
                    "overall_score": None,
                    "overall_reason": str(exc)
                }

                results.append(res_dict)
                save_current_progress(results, existing_successful_results, output_results_path)
                print(f"Saved progress after failed example {position} ({j_status}).", flush=True)

                if judge.provider_name != "mock" and position < total_eval:
                    time.sleep(args.delay)
                continue

            # ----------------------------------------------------
            # Determine success and judge_status
            # ----------------------------------------------------

            is_success = bool(judge_result.get("judge_success", False))
            if is_success:
                j_status = "success"
            else:
                j_status = judge_result.get("judge_status", "")
                if not j_status:
                    err_str = str(judge_result.get("judge_error", "")).lower()
                    if "429" in err_str or "rate limit" in err_str:
                        j_status = "provider_rate_limit"
                    elif "timeout" in err_str or "timed out" in err_str:
                        j_status = "timeout"
                    else:
                        j_status = "provider_error"

            res_dict = {
                "tweet_id": tweet_id,
                "customer_message": customer_msg,
                "predicted_intent": predicted_intent,
                "selected_evidence_text": evidence_text,
                "generated_reply": generated_reply,
                "escalation_decision": escalation_decision,
                "provider": judge.provider_name,
                "model": judge.model_name,
                "judge_success": is_success,
                "judge_status": j_status,

                # Scores
                "correctness_score": judge_result["correctness"]["score"],
                "correctness_reason": judge_result["correctness"]["reason"],
                "groundedness_score": judge_result["groundedness"]["score"],
                "groundedness_reason": judge_result["groundedness"]["reason"],
                "relevance_score": judge_result["relevance"]["score"],
                "relevance_reason": judge_result["relevance"]["reason"],
                "completeness_score": judge_result["completeness"]["score"],
                "completeness_reason": judge_result["completeness"]["reason"],
                "tone_score": judge_result["tone"]["score"],
                "tone_reason": judge_result["tone"]["reason"],
                "unsupported_claims_score": judge_result["unsupported_claims"]["score"],
                "unsupported_claims_reason": judge_result["unsupported_claims"]["reason"],
                "unsupported_claims_found": judge_result["unsupported_claims"].get("found", False),
                "unsupported_claims_list": "; ".join(judge_result["unsupported_claims"].get("claims", [])),
                "escalation_appropriateness_score": judge_result["escalation_appropriateness"]["score"],
                "escalation_appropriateness_reason": judge_result["escalation_appropriateness"]["reason"],
                "overall_score": judge_result["overall_score"],
                "overall_reason": judge_result["overall_reason"]
            }

            results.append(res_dict)
            if is_success:
                existing_successful_results[tweet_id_str] = res_dict

            # ----------------------------------------------------
            # Save progress immediately after EVERY example
            # ----------------------------------------------------
            save_current_progress(results, existing_successful_results, output_results_path)

            status_desc = "SUCCESS" if is_success else f"FAILED ({j_status})"
            print(f"Judged {position}/{total_eval} examples... [{status_desc}]", flush=True)

            # ----------------------------------------------------
            # Rate-limit protection delay
            # ----------------------------------------------------
            if judge.provider_name != "mock" and position < total_eval:
                print(f"Waiting {args.delay:.1f}s before next request...", flush=True)
                time.sleep(args.delay)

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Previously completed results have been preserved.", flush=True)

    # ========================================================
    # FINAL RESULTS DATAFRAME & PERSISTENCE
    # ========================================================

    final_df = save_current_progress(results, existing_successful_results, output_results_path)
    print(f"\nDetailed judge results saved to: {output_results_path} ({len(final_df)} records)", flush=True)

    # Generate summary with explicit completion check
    generate_summary(final_df, judge, output_summary_path, total_requested=total_target)

    if interrupted:
        sys.exit(0)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()