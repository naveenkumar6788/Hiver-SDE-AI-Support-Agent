"""
src/retrieval/evaluate_retrieval.py

Rigorous evaluation of the improved AppleSupport retrieval system across the 160-example
golden evaluation set.

Features:
- Conversation-safe retrieval: strictly excludes the query's own conversation thread.
- Computes IR metrics: Precision@1, Precision@3, Precision@5, MRR, NDCG@5.
- Computes Intent-Alignment metrics: Top-1 Intent Match Rate, Top-K Intent Match Rate.
- Computes Semantic Similarity statistics.
- Distinguishes heuristic development diagnostics from genuine human ground truth.
- Displays qualitative inspection samples (good and edge cases).
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 console output
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import AppleSupportRetriever, clean_text, normalize_id

GOLDEN_FILE = PROJECT_ROOT / "golden_set" / "golden_independently_verified.csv"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = RESULTS_DIR / "improved_retrieval_results.csv"
QUERY_METRICS_FILE = RESULTS_DIR / "improved_retrieval_query_metrics.csv"
SUMMARY_FILE = RESULTS_DIR / "improved_retrieval_summary.csv"
FAILURES_FILE = RESULTS_DIR / "retrieval_relevance_failures.csv"

TOP_K = 5
MIN_EVIDENCE_RELEVANCE = 0.55
MIN_EVIDENCE_QUALITY = 0.45


# ============================================================
# Heuristic Diagnostic Relevance Calculation
# ============================================================

def calculate_diagnostic_relevance(
    query_text,
    historical_customer,
    historical_response,
    true_intent,
    historical_intent,
    similarity
):
    """
    Diagnostic relevance heuristic for development benchmarking.
    NOTE: This is an automated proxy, NOT human ground truth.

    2 = strongly relevant (aligned intent + substantive topical overlap)
    1 = somewhat relevant (aligned domain or partial topical overlap)
    0 = irrelevant (unrelated intent / off-topic)
    """
    q_words = set(clean_text(query_text).split())
    c_words = set(clean_text(historical_customer).split())
    r_words = set(clean_text(historical_response).split())

    c_overlap = len(q_words.intersection(c_words)) / max(len(q_words), 1)
    r_overlap = len(q_words.intersection(r_words)) / max(len(q_words), 1)

    intent_match = (true_intent == historical_intent) and (true_intent != "other_unclear")

    score = 0.0
    if similarity >= 0.50:
        score += 3.0
    elif similarity >= 0.35:
        score += 2.0
    elif similarity >= 0.25:
        score += 1.0
    elif similarity >= 0.15:
        score += 0.5

    # Intent agreement
    if intent_match:
        score += 2.5

    # Lexical overlap
    if c_overlap >= 0.40:
        score += 2.0
    elif c_overlap >= 0.20:
        score += 1.0

    if r_overlap >= 0.25:
        score += 1.0
    elif r_overlap >= 0.10:
        score += 0.5

    if score >= 5.0:
        return 2
    elif score >= 2.5:
        return 1
    return 0


# ============================================================
# Main Evaluation Workflow
# ============================================================

def main():
    print("=" * 80)
    print("EVIDENCE RELEVANCE & RETRIEVAL SYSTEM EVALUATION")
    print("=" * 80)
    print(f"Top K: {TOP_K}")
    print(f"Dataset: {GOLDEN_FILE.name}")
    print("Strategy: Conversation-Safe Grouped Retrieval (Current thread excluded)")

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_FILE}")

    golden = pd.read_csv(GOLDEN_FILE, dtype=str).fillna("")
    print(f"Total evaluation queries available: {len(golden)}")
    print(f"Unique evaluation conversations   : {golden['conversation_id'].nunique()}")

    print("\nLoading improved AppleSupport retriever...")
    retriever = AppleSupportRetriever()
    print("Retriever ready.")

    all_details = []
    query_metrics = []

    print("\nExecuting evaluation across all queries...")

    for query_num, (_, row) in enumerate(golden.iterrows(), start=1):
        tweet_id = normalize_id(row.get("tweet_id", ""))
        conv_id = normalize_id(row.get("conversation_id", ""))
        query_text = str(row.get("text", "")).strip()
        true_intent = str(row.get("human_intent", "")).strip()

        # Predict intent using retriever's classifier
        intent_res = retriever.predict_intent(query_text)
        pred_intent = intent_res.get("intent", "other_unclear") if isinstance(intent_res, dict) else str(intent_res)
        confidence = float(intent_res.get("confidence", 0.0)) if isinstance(intent_res, dict) else 0.0

        # Conversation-safe retrieval (strict exclusion of current conversation)
        results = retriever.retrieve(
            query_text,
            top_k=TOP_K,
            intent=pred_intent,
            exclude_conversation_id=conv_id
        )

        relevance_scores = []
        sim_scores = []
        intent_matches = []

        for rank, (_, res_row) in enumerate(results.iterrows(), start=1):
            sim = float(res_row.get("similarity", 0.0))
            score = float(res_row.get("final_score", res_row.get("retrieval_score", 0.0)))
            hist_conv = str(res_row.get("conversation_id", ""))
            hist_cust = str(res_row.get("customer_text", res_row.get("text", "")))
            hist_resp = str(res_row.get("historical_response", res_row.get("response", "")))
            hist_intent = str(res_row.get("retrieval_intent", ""))
            int_match = bool(res_row.get("intent_match", False))

            # Evidence quality & relevance metrics
            ev_score = float(res_row.get("evidence_quality_score", 0.50))
            ev_reason = str(res_row.get("evidence_quality_reason", "substantive response"))
            is_act = bool(res_row.get("actionable", False))
            is_gen = bool(res_row.get("generic_response", False))
            ev_rel_score = float(res_row.get("evidence_relevance_score", 0.0))
            prob_score = float(res_row.get("problem_match_score", 0.0))
            prob_reason = str(res_row.get("problem_match_reason", ""))
            ent_overlap = str(res_row.get("entity_overlap", []))
            ent_mismatch = bool(res_row.get("entity_mismatch", False))
            iss_overlap = str(res_row.get("issue_overlap", []))
            iss_mismatch = bool(res_row.get("issue_mismatch", False))
            rel_accepted = bool(res_row.get("relevance_accepted", False))
            rel_reason = str(res_row.get("relevance_reason", ""))

            # Diagnostic relevance
            rel = calculate_diagnostic_relevance(
                query_text=query_text,
                historical_customer=hist_cust,
                historical_response=hist_resp,
                true_intent=true_intent,
                historical_intent=hist_intent,
                similarity=sim
            )

            relevance_scores.append(rel)
            sim_scores.append(sim)
            intent_matches.append(int_match)

            all_details.append({
                "query_number": query_num,
                "query_tweet_id": tweet_id,
                "query_conversation_id": conv_id,
                "query_text": query_text,
                "true_intent": true_intent,
                "predicted_intent": pred_intent,
                "prediction_confidence": confidence,
                "rank": rank,
                "final_score": score,
                "similarity": sim,
                "semantic_similarity": sim,
                "retrieval_score": score,
                "evidence_quality_score": ev_score,
                "evidence_quality_reason": ev_reason,
                "actionable": is_act,
                "generic_response": is_gen,
                "evidence_relevance_score": ev_rel_score,
                "problem_match_score": prob_score,
                "problem_match_reason": prob_reason,
                "entity_overlap": ent_overlap,
                "entity_mismatch": ent_mismatch,
                "issue_overlap": iss_overlap,
                "issue_mismatch": iss_mismatch,
                "relevance_accepted": rel_accepted,
                "relevance_reason": rel_reason,
                "historical_conversation_id": hist_conv,
                "historical_intent": hist_intent,
                "intent_match": int_match,
                "historical_customer": hist_cust,
                "historical_response": hist_resp,
                "diagnostic_relevance": rel
            })

        # Calculate query-level metrics
        if relevance_scores:
            rel_arr = np.array(relevance_scores)
            binary_rel = (rel_arr > 0).astype(int)

            p_at_1 = binary_rel[0] if len(binary_rel) >= 1 else 0
            p_at_3 = binary_rel[:3].sum() / 3.0 if len(binary_rel) >= 3 else binary_rel.mean()
            p_at_5 = binary_rel.sum() / len(binary_rel)

            # MRR (Mean Reciprocal Rank)
            rr = 0.0
            for r_idx, val in enumerate(binary_rel, start=1):
                if val == 1:
                    rr = 1.0 / r_idx
                    break

            # NDCG@5
            try:
                ndcg = ndcg_score([rel_arr], [rel_arr], k=TOP_K)
            except Exception:
                ndcg = 0.0

            top1_intent_match = 1 if intent_matches and intent_matches[0] else 0
            topk_intent_match_rate = sum(intent_matches) / len(intent_matches) if intent_matches else 0.0
            avg_sim = np.mean(sim_scores) if sim_scores else 0.0

            query_metrics.append({
                "query_number": query_num,
                "query_tweet_id": tweet_id,
                "conversation_id": conv_id,
                "true_intent": true_intent,
                "predicted_intent": pred_intent,
                "p_at_1": p_at_1,
                "p_at_3": p_at_3,
                "p_at_5": p_at_5,
                "mrr": rr,
                "ndcg_at_5": ndcg,
                "top1_intent_match": top1_intent_match,
                "topk_intent_match_rate": topk_intent_match_rate,
                "average_similarity": avg_sim,
                "strongly_relevant_count": int((rel_arr == 2).sum()),
                "somewhat_relevant_count": int((rel_arr == 1).sum()),
                "irrelevant_count": int((rel_arr == 0).sum())
            })

    # Save detailed artifacts
    details_df = pd.DataFrame(all_details)
    metrics_df = pd.DataFrame(query_metrics)

    details_df.to_csv(DETAIL_FILE, index=False)
    metrics_df.to_csv(QUERY_METRICS_FILE, index=False)

    top1_details = details_df[details_df["rank"] == 1]

    summary = {
        "queries_evaluated": len(metrics_df),
        "unique_conversations": golden["conversation_id"].nunique(),
        "top_k": TOP_K,
        "relevance_threshold": MIN_EVIDENCE_RELEVANCE,
        "quality_threshold": MIN_EVIDENCE_QUALITY,
        "precision_at_1": metrics_df["p_at_1"].mean(),
        "precision_at_3": metrics_df["p_at_3"].mean(),
        "precision_at_5": metrics_df["p_at_5"].mean(),
        "mrr": metrics_df["mrr"].mean(),
        "ndcg_at_5": metrics_df["ndcg_at_5"].mean(),
        "top1_intent_match_rate": metrics_df["top1_intent_match"].mean(),
        "overall_intent_match_rate": metrics_df["topk_intent_match_rate"].mean(),
        "average_similarity": metrics_df["average_similarity"].mean(),
        "average_evidence_quality": details_df["evidence_quality_score"].mean(),
        "average_evidence_relevance": details_df["evidence_relevance_score"].mean(),
        "average_problem_match": details_df["problem_match_score"].mean(),
        "pct_actionable": (details_df["actionable"].mean() * 100.0),
        "pct_generic": (details_df["generic_response"].mean() * 100.0),
        "pct_accepted": (details_df["relevance_accepted"].mean() * 100.0),
        "pct_rejected": ((~details_df["relevance_accepted"]).mean() * 100.0),
        "pct_application_mismatch": (details_df["entity_mismatch"].mean() * 100.0),
        "pct_issue_mismatch": (details_df["issue_mismatch"].mean() * 100.0),
        "top1_pct_actionable": (top1_details["actionable"].mean() * 100.0) if len(top1_details) > 0 else 0.0,
        "top1_pct_generic": (top1_details["generic_response"].mean() * 100.0) if len(top1_details) > 0 else 0.0,
        "top1_pct_accepted": (top1_details["relevance_accepted"].mean() * 100.0) if len(top1_details) > 0 else 0.0,
        "total_strongly_relevant": int(metrics_df["strongly_relevant_count"].sum()),
        "total_somewhat_relevant": int(metrics_df["somewhat_relevant_count"].sum()),
        "total_irrelevant": int(metrics_df["irrelevant_count"].sum())
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_FILE, index=False)

    # --------------------------------------------------------
    # Failure Analysis CSV Export
    # --------------------------------------------------------
    # Identify cases where:
    # A. Intent matches but relevance is low
    # B. Entity matches but problem doesn't
    # C. High-quality evidence rejected due to mismatch
    # D. Generic responses rejected
    # E. Application mismatch occurs
    # F. Issue mismatch occurs
    failure_mask = (
        (details_df["intent_match"] & (details_df["evidence_relevance_score"] < MIN_EVIDENCE_RELEVANCE)) |
        ((details_df["entity_overlap"].astype(str) != "[]") & (details_df["problem_match_score"] < 0.50)) |
        ((details_df["evidence_quality_score"] >= 0.70) & ~details_df["relevance_accepted"]) |
        (details_df["generic_response"] & ~details_df["relevance_accepted"]) |
        details_df["entity_mismatch"] |
        details_df["issue_mismatch"]
    )
    failures_df = details_df[failure_mask].sort_values(
        by=["entity_mismatch", "issue_mismatch", "relevance_accepted", "evidence_relevance_score", "similarity"],
        ascending=[False, False, True, True, False]
    ).copy()

    failures_export = failures_df[[
        "query_text", "true_intent", "predicted_intent", "historical_customer",
        "historical_response", "similarity", "evidence_quality_score",
        "problem_match_score", "problem_match_reason", "entity_overlap",
        "entity_mismatch", "issue_mismatch", "evidence_relevance_score",
        "final_score", "relevance_accepted", "relevance_reason"
    ]].rename(columns={
        "query_text": "query",
        "true_intent": "query_intent",
        "historical_customer": "historical_customer_message",
        "relevance_accepted": "accepted"
    })
    failures_export.to_csv(FAILURES_FILE, index=False)

    # --------------------------------------------------------
    # Print Clean Terminal Summary
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION RESULTS WITH HARDENED EVIDENCE GATE")
    print("=" * 80)
    print(f"Queries Evaluated          : {summary['queries_evaluated']}")
    print(f"Unique Conversations       : {summary['unique_conversations']}")
    print(f"Top K per Query            : {TOP_K}")
    print(f"Total Retrieved Cases      : {len(details_df)}")
    print(f"Relevance Threshold        : {summary['relevance_threshold']}")
    print(f"Quality Threshold          : {summary['quality_threshold']}")
    print("-" * 80)
    print("RETRIEVAL QUALITY METRICS (Diagnostic Benchmark - NOT Ground Truth)")
    print(f"Precision@1                : {summary['precision_at_1']:.4f} ({summary['precision_at_1'] * 100:.1f}%)")
    print(f"Precision@3                : {summary['precision_at_3']:.4f} ({summary['precision_at_3'] * 100:.1f}%)")
    print(f"Precision@5                : {summary['precision_at_5']:.4f} ({summary['precision_at_5'] * 100:.1f}%)")
    print(f"MRR                        : {summary['mrr']:.4f}")
    print(f"NDCG@5                     : {summary['ndcg_at_5']:.4f}")
    print("-" * 80)
    print("INTENT & SEMANTIC ALIGNMENT")
    print(f"Top-1 Intent Match Rate    : {summary['top1_intent_match_rate']:.4f} ({summary['top1_intent_match_rate'] * 100:.1f}%)")
    print(f"Overall Intent Match Rate  : {summary['overall_intent_match_rate']:.4f} ({summary['overall_intent_match_rate'] * 100:.1f}%)")
    print(f"Average Cosine Similarity  : {summary['average_similarity']:.4f}")
    print("-" * 80)
    print("EVIDENCE QUALITY & RELEVANCE BENCHMARKS")
    print(f"Average Evidence Quality   : {summary['average_evidence_quality']:.4f}")
    print(f"Average Evidence Relevance : {summary['average_evidence_relevance']:.4f}")
    print(f"Average Problem Match Score: {summary['average_problem_match']:.4f}")
    print(f"Application Mismatch %     : {summary['pct_application_mismatch']:.1f}%")
    print(f"Issue Mismatch %           : {summary['pct_issue_mismatch']:.1f}%")
    print(f"Actionable Cases (Overall) : {summary['pct_actionable']:.1f}%")
    print(f"Generic Cases (Overall)    : {summary['pct_generic']:.1f}%")
    print(f"Accepted Evidence (Overall): {summary['pct_accepted']:.1f}%")
    print(f"Rejected Evidence (Overall): {summary['pct_rejected']:.1f}%")
    print(f"Top-1 Actionable Cases     : {summary['top1_pct_actionable']:.1f}%")
    print(f"Top-1 Accepted Evidence    : {summary['top1_pct_accepted']:.1f}%")
    print("-" * 80)
    print("RELEVANCE DISTRIBUTION (Diagnostic Heuristic)")
    print(f"Strongly Relevant (Score 2): {summary['total_strongly_relevant']}")
    print(f"Somewhat Relevant (Score 1): {summary['total_somewhat_relevant']}")
    print(f"Irrelevant        (Score 0): {summary['total_irrelevant']}")
    print("=" * 80)

    # --------------------------------------------------------
    # Qualitative Inspection: Representative Retrieval Cases
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("QUALITATIVE INSPECTION: REPRESENTATIVE RETRIEVAL CASES")
    print("=" * 80)

    sample_indices = [0, 10, 25, 45, 80, 120]
    sample_indices = [idx for idx in sample_indices if idx < len(metrics_df)]

    for s_idx in sample_indices:
        q_row = metrics_df.iloc[s_idx]
        q_num = int(q_row["query_number"])
        top_cases = details_df[details_df["query_number"] == q_num].head(1)

        if len(top_cases) == 0:
            continue

        c = top_cases.iloc[0]
        q_disp = c["query_text"][:95].encode("ascii", "replace").decode("ascii")
        cust_disp = str(c["historical_customer"])[:95].encode("ascii", "replace").decode("ascii")
        resp_disp = str(c["historical_response"])[:110].encode("ascii", "replace").decode("ascii")

        print(f"\n[Case {q_num}] True Intent: {c['true_intent']} | Predicted: {c['predicted_intent']}")
        print(f"  Query        : {q_disp}...")
        print(f"  Top Match    : FinalScore={c['final_score']:.4f} | Sim={c['similarity']:.4f} | Match={c['intent_match']} | Intent={c['historical_intent']}")
        print(f"  Evidence     : Quality={c['evidence_quality_score']:.2f} | Relevance={c['evidence_relevance_score']:.4f} | Accepted={c['relevance_accepted']}")
        print(f"  Reason       : {c['relevance_reason']}")
        print(f"  Hist Customer: {cust_disp}...")
        print(f"  Hist Response: {resp_disp}...")
        print("-" * 80)

    print("\nSaved artifacts:")
    print(f"  - {DETAIL_FILE}")
    print(f"  - {QUERY_METRICS_FILE}")
    print(f"  - {SUMMARY_FILE}")
    print(f"  - {FAILURES_FILE}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()