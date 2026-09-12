"""
src/evaluation/run_current_llm_judge_50.py

Dedicated runner for obtaining a fresh LLM-as-a-Judge evaluation of the
CURRENT frozen system responses for all 50 cases in golden_set/human_evaluation_50.csv.

Invariants & Scientific Integrity:
1. Never modifies human_evaluation_50.csv, golden labels, or system replies.
2. Never fabricates LLM judge scores.
3. Evaluates the CURRENT frozen system reply for each case.
4. Preserves atomic saving (.tmp -> rename) and resume capability.
5. Saves results separately to:
   - evaluation/results/llm_judge_current_50.csv
   - evaluation/results/llm_judge_current_summary.csv
6. If the provider returns 403 / 429 or is unavailable, stops cleanly without hammering.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=str(env_file), override=False)
    else:
        load_dotenv(override=False)
except ImportError:
    pass

from src.evaluation.llm_judge import LLMJudge


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM Judge on the 50 human evaluation cases for current system responses."
    )
    parser.add_argument(
        "--human_set",
        type=str,
        default="golden_set/human_evaluation_50.csv",
        help="Path to 50-case human evaluation CSV."
    )
    parser.add_argument(
        "--system_results",
        type=str,
        default="evaluation/results/reply_generator_results.csv",
        help="Path to current reply generator full results CSV (for tweet_id and evidence mapping)."
    )
    parser.add_argument(
        "--output_results",
        type=str,
        default="evaluation/results/llm_judge_current_50.csv",
        help="Output path for current 50-case judge results."
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="evaluation/results/llm_judge_current_summary.csv",
        help="Output path for current 50-case judge summary."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.5,
        help="Pacing delay between API calls in seconds (default: 3.5s)."
    )
    parser.add_argument(
        "--smoke_test",
        "--smoke-test",
        type=int,
        default=0,
        help="Run only N smoke-test cases (e.g. 3) to verify schema, scores, and connection."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries on transient errors."
    )
    parser.add_argument(
        "--max_retry_wait",
        type=float,
        default=30.0,
        help="Max retry backoff wait in seconds."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider override (default from LLM_PROVIDER or .env)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model override (default from LLM_MODEL or .env)."
    )
    return parser.parse_args()


def load_and_verify_dataset(human_set_path: str, system_results_path: str) -> pd.DataFrame:
    human_path = PROJECT_ROOT / human_set_path
    if not human_path.exists():
        raise FileNotFoundError(f"Human evaluation set not found at {human_path}")

    df_human = pd.read_csv(human_path)
    if len(df_human) != 50:
        raise ValueError(f"Expected 50 cases in {human_set_path}, found {len(df_human)}")

    # Verify required columns exist
    req_cols = ["case_id", "customer_message", "system_intent", "system_reply", "system_decision"]
    for col in req_cols:
        if col not in df_human.columns:
            raise ValueError(f"Required column '{col}' missing from {human_set_path}")

    # Map tweet_id and evidence details from system_results if available
    sys_path = PROJECT_ROOT / system_results_path
    if sys_path.exists():
        df_sys = pd.read_csv(sys_path)
        # Match on customer_message
        merged = pd.merge(
            df_human,
            df_sys[["customer_message", "tweet_id", "reply", "intent_match", "problem_match", "escalation_reason", "evidence_used"]],
            on="customer_message",
            how="left",
            suffixes=("", "_sys")
        )
        # Verify replies match exactly
        reply_mismatches = 0
        for _, row in merged.iterrows():
            curr_sys_reply = str(row.get("reply", "")).strip()
            human_sys_reply = str(row.get("system_reply", "")).strip()
            if curr_sys_reply and curr_sys_reply != human_sys_reply:
                reply_mismatches += 1
        if reply_mismatches > 0:
            print(f"[WARNING] {reply_mismatches} system replies in human set differ from system_results.")
        else:
            print("[VERIFIED] All 50 system replies match the current frozen system replies with 100% exact equality.")
        return merged

    return df_human


def save_atomic_progress(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    tmp_path = output_path.with_suffix(".tmp")
    df.to_csv(tmp_path, index=False)
    if tmp_path.exists():
        import shutil
        shutil.move(tmp_path, output_path)
    else:
        df.to_csv(output_path, index=False)


def generate_judge_summary(df_results: pd.DataFrame, judge: LLMJudge, output_path: Path, requested_count: int) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_mask = (df_results["judge_success"] == True) & (df_results["judge_status"].astype(str).str.lower() == "success")
    df_succ = df_results[success_mask]

    successful = len(df_succ)
    failed = len(df_results) - successful
    rate = round(successful / requested_count, 4) if requested_count > 0 else 0.0

    metrics = [
        ("requested_cases", requested_count),
        ("processed_cases", len(df_results)),
        ("successful_judgments", successful),
        ("failed_judgments", failed),
        ("success_rate", rate),
        ("provider", judge.provider_name),
        ("model", judge.model_name),
        ("average_correctness", round(df_succ["correctness_score"].mean(), 4) if successful > 0 else 0.0),
        ("average_groundedness", round(df_succ["groundedness_score"].mean(), 4) if successful > 0 else 0.0),
        ("average_relevance", round(df_succ["relevance_score"].mean(), 4) if successful > 0 else 0.0),
        ("average_completeness", round(df_succ["completeness_score"].mean(), 4) if successful > 0 else 0.0),
        ("average_tone", round(df_succ["tone_score"].mean(), 4) if successful > 0 else 0.0),
        ("average_unsupported_claims", round(df_succ["unsupported_claims_score"].mean(), 4) if successful > 0 else 0.0),
        ("average_escalation_appropriateness", round(df_succ["escalation_appropriateness_score"].mean(), 4) if successful > 0 else 0.0),
        ("average_overall_score", round(df_succ["overall_score"].mean(), 4) if successful > 0 else 0.0),
    ]

    summary_df = pd.DataFrame(metrics, columns=["metric", "value"])
    summary_df.to_csv(output_path, index=False)
    return summary_df


def run_evaluation():
    args = parse_args()
    print("=" * 70)
    print("CURRENT FROZEN SYSTEM LLM-AS-A-JUDGE EVALUATION (50 CASES)")
    print("=" * 70)

    dataset = load_and_verify_dataset(args.human_set, args.system_results)

    total_target = args.smoke_test if args.smoke_test > 0 else len(dataset)
    df_eval = dataset.head(total_target).copy()

    if args.smoke_test > 0:
        print(f"\n[SMOKE TEST MODE] Running evaluation on {total_target} cases only.\n")
    else:
        print(f"\n[FULL EVALUATION] Running evaluation on all {total_target} cases.\n")

    # Initialize LLM Judge
    judge = LLMJudge(
        provider=args.provider,
        model=args.model,
        max_retries=args.max_retries,
        max_retry_wait=args.max_retry_wait
    )
    print(f"Initialized LLMJudge: provider='{judge.provider_name}', model='{judge.model_name}'")

    out_results_path = PROJECT_ROOT / args.output_results
    out_summary_path = PROJECT_ROOT / args.output_summary

    # Load existing results for resume capability
    existing_records: Dict[str, Dict[str, Any]] = {}
    if out_results_path.exists():
        try:
            df_prev = pd.read_csv(out_results_path)
            for _, prev_row in df_prev.iterrows():
                cid = str(prev_row.get("case_id", "")).strip()
                is_succ = bool(prev_row.get("judge_success", False)) and str(prev_row.get("judge_status", "")).lower() == "success"
                if is_succ and cid:
                    existing_records[cid] = prev_row.to_dict()
            print(f"Found {len(existing_records)} existing successful judgments to resume from.")
        except Exception as e:
            print(f"[WARNING] Could not read existing results: {e}. Starting fresh.")

    results: List[Dict[str, Any]] = []

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Process cases
    for idx, (_, row) in enumerate(df_eval.iterrows(), 1):
        case_id = str(row.get("case_id", f"case_{idx:02d}")).strip()
        customer_msg = str(row.get("customer_message", "")).strip()
        system_reply = str(row.get("system_reply", "")).strip()
        system_intent = str(row.get("system_intent", "")).strip()
        system_decision = str(row.get("system_decision", "AUTO_HANDLE")).strip()
        escalation_reason = str(row.get("system_escalation_reason", "")).strip()
        if escalation_reason == "nan":
            escalation_reason = ""

        historical_cust = str(row.get("historical_customer_message", "")).strip()
        historical_resp = str(row.get("historical_response", "")).strip()
        if historical_resp and historical_resp != "nan":
            evidence_text = f"Customer: {historical_cust}\nResponse: {historical_resp}"
        else:
            evidence_text = ""

        tweet_id = row.get("tweet_id", "")
        evidence_used = row.get("evidence_used", True if evidence_text else False)

        # Resume check
        if case_id in existing_records:
            print(f"[{idx}/{total_target}] SKIP {case_id} (already evaluated successfully)")
            results.append(existing_records[case_id])
            continue

        safe_preview = customer_msg[:50].encode("ascii", "replace").decode("ascii")
        print(f"[{idx}/{total_target}] Evaluating {case_id}: \"{safe_preview}...\"")

        try:
            res = judge.judge(
                customer_message=customer_msg,
                predicted_intent=system_intent,
                historical_evidence=evidence_text,
                generated_reply=system_reply,
                escalation_decision=system_decision,
                expected_intent=row.get("human_intent", system_intent),
                evidence_used=evidence_used,
                escalation_reason=escalation_reason,
            )
        except Exception as exc:
            print(f"  FAILED {case_id}: {exc}")
            res = {
                "judge_success": False,
                "judge_status": "error",
                "judge_error": str(exc),
            }

        # Build output record
        is_succ = bool(res.get("judge_success", False)) and str(res.get("judge_status", "")).lower() == "success"

        rec = {
            "case_id": case_id,
            "tweet_id": tweet_id,
            "customer_message": customer_msg,
            "system_intent": system_intent,
            "system_decision": system_decision,
            "system_escalation_reason": escalation_reason,
            "system_reply": system_reply,
            "evidence_text": evidence_text,
            "judge_provider": judge.provider_name,
            "judge_model": judge.model_name,
            "judge_success": is_succ,
            "judge_status": res.get("judge_status", "error"),
            "judge_error": res.get("judge_error", ""),
            "correctness_score": res["correctness"]["score"] if is_succ else None,
            "correctness_reason": res["correctness"]["reason"] if is_succ else res.get("judge_error", ""),
            "groundedness_score": res["groundedness"]["score"] if is_succ else None,
            "groundedness_reason": res["groundedness"]["reason"] if is_succ else res.get("judge_error", ""),
            "relevance_score": res["relevance"]["score"] if is_succ else None,
            "relevance_reason": res["relevance"]["reason"] if is_succ else res.get("judge_error", ""),
            "completeness_score": res["completeness"]["score"] if is_succ else None,
            "completeness_reason": res["completeness"]["reason"] if is_succ else res.get("judge_error", ""),
            "tone_score": res["tone"]["score"] if is_succ else None,
            "tone_reason": res["tone"]["reason"] if is_succ else res.get("judge_error", ""),
            "unsupported_claims_score": res["unsupported_claims"]["score"] if is_succ else None,
            "unsupported_claims_reason": res["unsupported_claims"]["reason"] if is_succ else res.get("judge_error", ""),
            "unsupported_claims_found": res["unsupported_claims"].get("found", False) if is_succ else False,
            "escalation_appropriateness_score": res["escalation_appropriateness"]["score"] if is_succ else None,
            "escalation_appropriateness_reason": res["escalation_appropriateness"]["reason"] if is_succ else res.get("judge_error", ""),
            "overall_score": res.get("overall_score") if is_succ else None,
            "overall_reason": res.get("overall_reason", "") if is_succ else res.get("judge_error", ""),
        }

        results.append(rec)

        # Atomic save after every row
        save_atomic_progress(results, out_results_path)

        if is_succ:
            print(f"  SUCCESS {case_id}: overall={rec['overall_score']}, correctness={rec['correctness_score']}, groundedness={rec['groundedness_score']}")
        else:
            print(f"  FAILURE {case_id}: {rec['judge_status']} - {rec['judge_error']}")
            # Check for fatal Cloudflare block
            err_lower = str(rec['judge_error']).lower()
            if "403" in err_lower or "1010" in err_lower or "cloudflare" in err_lower:
                print("\n[CRITICAL] Cloudflare HTTP 403 / 1010 block detected. Aborting to avoid hammering provider.\n")
                break

        # Rate-limiting delay between calls
        if idx < total_target:
            time.sleep(args.delay)

    # Final summary
    df_all_results = pd.DataFrame(results)
    summary_df = generate_judge_summary(df_all_results, judge, out_summary_path, requested_count=total_target)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for _, srow in summary_df.iterrows():
        print(f"  {srow['metric']:<35}: {srow['value']}")
    print("=" * 50)

    return df_all_results, summary_df


if __name__ == "__main__":
    run_evaluation()
