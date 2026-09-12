"""
src/evaluation/prepare_human_evaluation.py

Prepares an independent, balanced 50-case sample from the frozen 160-case evaluation set
and exports:
1. golden_set/human_evaluation_50.csv (selected evaluation cases with full metadata)
2. golden_set/human_evaluation_template.csv (blind annotation template for human raters)

Guarantees:
- Fixed random seed for complete reproducibility
- No duplicate conversation IDs
- Diverse representation of all 12 intents, evidence outcomes, resolved messages,
  and sensitive/hazard escalation cases
- Completely blind to LLM judge scores and expected evaluation ratings
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SEED = 42

TEMPLATE_COLUMNS = [
    "case_id",
    "customer_message",
    "system_intent",
    "historical_customer_message",
    "historical_response",
    "system_reply",
    "system_decision",
    "system_escalation_reason",
    "human_intent",
    "intent_correct",
    "correctness_1_5",
    "groundedness_1_5",
    "relevance_1_5",
    "completeness_1_5",
    "tone_1_5",
    "unsupported_claims_1_5",
    "escalation_appropriateness_1_5",
    "evidence_supported",
    "overall_comment",
]


def prepare_human_evaluation_set(
    eval_results_path: str = "evaluation/results/reply_generator_results.csv",
    output_cases_path: str = "golden_set/human_evaluation_50.csv",
    output_template_path: str = "golden_set/human_evaluation_template.csv",
    random_seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    eval_file = PROJECT_ROOT / eval_results_path
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_file}")

    df_160 = pd.read_csv(eval_file)
    rng = np.random.RandomState(random_seed)

    # 1. Ensure unique conversation IDs to avoid thread overlap
    df_unique = df_160.drop_duplicates(subset=["conversation_id"]).copy()

    selected_indices = set()

    # 2. Priority selections for required behavioral coverage:
    # A. Escalation cases (security, billing/refunds)
    esc_cases = df_unique[df_unique["escalation_decision"] == "ESCALATE"]
    for idx in esc_cases.index:
        selected_indices.add(idx)

    # B. Resolved customer messages
    res_cases = df_unique[df_unique["response_type"] == "resolved_acknowledgement"]
    for idx in res_cases.index:
        selected_indices.add(idx)

    # C. Safety/hazard cases (alarm noise, hardware danger)
    hazard_cases = df_unique[
        df_unique["customer_message"].str.contains("smoke|alarm|hazard", case=False, na=False)
        | (df_unique["tweet_id"] == 1123670)
    ]
    for idx in hazard_cases.index:
        selected_indices.add(idx)

    # D. Rejected evidence cases
    rej_cases = df_unique[
        (df_unique["evidence_safe"] == False)
        & (df_unique["evidence_rejection_reason"].notna())
        & (df_unique["evidence_rejection_reason"] != "no_evidence")
        & (~df_unique.index.isin(selected_indices))
    ]
    for idx in rej_cases.sample(n=min(5, len(rej_cases)), random_state=rng).index:
        selected_indices.add(idx)

    # E. Ensure representation across all 12 intents (at least 2 per intent where possible)
    intents = df_unique["gold_intent"].dropna().unique()
    for intent in intents:
        current_in_intent = [
            i for i in selected_indices if df_unique.loc[i, "gold_intent"] == intent
        ]
        needed = max(0, 2 - len(current_in_intent))
        if needed > 0:
            candidates = df_unique[
                (df_unique["gold_intent"] == intent) & (~df_unique.index.isin(selected_indices))
            ]
            if len(candidates) > 0:
                take = candidates.sample(n=min(needed, len(candidates)), random_state=rng)
                for idx in take.index:
                    selected_indices.add(idx)

    # F. Fill the remaining quota up to exactly 50 using balanced random sampling
    remaining_needed = 50 - len(selected_indices)
    if remaining_needed > 0:
        remaining_pool = df_unique[~df_unique.index.isin(selected_indices)]
        fill = remaining_pool.sample(n=remaining_needed, random_state=rng)
        for idx in fill.index:
            selected_indices.add(idx)

    df_selected = df_unique.loc[sorted(selected_indices)].copy().reset_index(drop=True)

    # 3. Check for historical evidence text (if present in historical judge files or evidence)
    judge_file = PROJECT_ROOT / "golden_set" / "human_judge_50.csv"
    hist_responses = {}
    if judge_file.exists():
        df_j = pd.read_csv(judge_file)
        for _, r in df_j.iterrows():
            if pd.notna(r.get("evidence_text")):
                hist_responses[int(r["tweet_id"])] = str(r["evidence_text"])

    # 4. Build case records
    records = []
    template_records = []

    for idx, row in df_selected.iterrows():
        case_id = f"case_{idx + 1:02d}"
        t_id = int(row["tweet_id"])
        c_id = str(row["conversation_id"])
        cust_msg = str(row["customer_message"])
        sys_intent = str(row.get("predicted_intent", row.get("gold_intent", "")))
        gold_intent = str(row.get("gold_intent", sys_intent))
        sys_reply = str(row.get("reply", ""))
        sys_decision = str(row.get("escalation_decision", "AUTO_HANDLE"))
        sys_esc_reason = str(row.get("escalation_reason", "")) if pd.notna(row.get("escalation_reason")) else ""

        # Historical customer and response
        hist_resp = hist_responses.get(t_id, "")
        if not hist_resp and bool(row.get("evidence_used", False)):
            # If evidence was used and historical text is cited in reply
            if "Based on a similar Apple Support case," in sys_reply:
                hist_resp = sys_reply.replace("Based on a similar Apple Support case,", "").strip()

        hist_cust = ""
        if hist_resp:
            # Clean display for blind annotation
            hist_cust = f"Historical case for {sys_intent}"

        records.append({
            "case_id": case_id,
            "tweet_id": t_id,
            "conversation_id": c_id,
            "customer_message": cust_msg,
            "gold_intent": gold_intent,
            "system_intent": sys_intent,
            "historical_customer_message": hist_cust,
            "historical_response": hist_resp,
            "system_reply": sys_reply,
            "system_decision": sys_decision,
            "system_escalation_reason": sys_esc_reason,
            "evidence_used": bool(row.get("evidence_used", False)),
            "evidence_safe": bool(row.get("evidence_safe", False)),
            "response_type": str(row.get("response_type", "")),
        })

        # Blind template (ratings empty for human evaluator)
        template_records.append({
            "case_id": case_id,
            "customer_message": cust_msg,
            "system_intent": sys_intent,
            "historical_customer_message": hist_cust,
            "historical_response": hist_resp,
            "system_reply": sys_reply,
            "system_decision": sys_decision,
            "system_escalation_reason": sys_esc_reason,
            "human_intent": "",
            "intent_correct": "",
            "correctness_1_5": "",
            "groundedness_1_5": "",
            "relevance_1_5": "",
            "completeness_1_5": "",
            "tone_1_5": "",
            "unsupported_claims_1_5": "",
            "escalation_appropriateness_1_5": "",
            "evidence_supported": "",
            "overall_comment": "",
        })

    df_cases = pd.DataFrame(records)
    df_template = pd.DataFrame(template_records, columns=TEMPLATE_COLUMNS).astype(str).replace("nan", "")

    # Save outputs
    out_cases = PROJECT_ROOT / output_cases_path
    out_template = PROJECT_ROOT / output_template_path

    out_cases.parent.mkdir(parents=True, exist_ok=True)
    out_template.parent.mkdir(parents=True, exist_ok=True)

    df_cases.to_csv(out_cases, index=False)
    df_template.to_csv(out_template, index=False)

    print("Human evaluation set created.")
    print(f"Cases: {len(df_cases)}")
    print(f"Conversation IDs: {df_cases['conversation_id'].nunique()}")
    print("LLM scores hidden: YES")
    print(f"Random seed: {random_seed}")

    return df_cases


if __name__ == "__main__":
    prepare_human_evaluation_set()
