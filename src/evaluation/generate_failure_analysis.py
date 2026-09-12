import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("d:/Hiver-SDE-AI-Support-Agent")

jr = pd.read_csv(PROJECT_ROOT / "evaluation/results/judge_results.csv")
rg = pd.read_csv(PROJECT_ROOT / "evaluation/results/reply_generator_results.csv")

merged = pd.merge(
    jr,
    rg[['tweet_id', 'intent_correct', 'evidence_safe', 'evidence_relevance', 'problem_match', 'escalation_decision', 'escalation_reason']],
    on='tweet_id',
    suffixes=('', '_rg')
)

# 10 prioritized worst cases
worst_ids = [
    (2547281, "grounded response issue", "Historical evidence reply was truncated and general iOS update guidance did not clearly provide the autocorrect/keyboard fix."),
    (162872, "retrieval error", "Customer on iOS reported Snapchat stopping Apple Music; retriever matched an Android-specific Apple Music installation thread."),
    (1477617, "insufficient evidence", "Customer submitted an ambiguous profanity query ('what the fuck is this? I️'); pipeline attempted an empty article reference instead of requesting clarification."),
    (1340829, "judge disagreement", "Customer discussed carrier restriction on iPhone X purchase; agent asked a valid diagnostic clarification question, but LLM judge penalized it for not providing immediate resolution."),
    (440797, "specific problem mismatch", "Customer encountered a web error during account recovery; evidence matched account wait-time guidance rather than web error troubleshooting."),
    (724937, "grounded response issue", "Customer reported device battery dying at 100%; safe clarification fallback erroneously asked cellular/call questions instead of battery diagnostics."),
    (1475391, "escalation error", "Customer asked in Spanish whether an iCloud reset email was a phishing virus; system provided iCloud toggle advice instead of escalating potential account security threat."),
    (1645496, "grounded response issue", "Customer suffered photo loss; evidence guidance was truncated ('have you checked to confir...') rather than providing full iCloud photo recovery instructions."),
    (2002700, "specific problem mismatch", "Customer reported dead battery unable to charge; retrieved evidence provided post-iOS update settings review rather than hardware charging diagnostics."),
    (1861583, "specific problem mismatch", "Customer had broken headphone plug physically lodged in device; retriever returned generic force-restart advice for a physical hardware extraction issue."),
]

worst_rows = []
for tid, f_class, f_expl in worst_ids:
    match = merged[merged['tweet_id'] == tid]
    if len(match) == 0:
        continue
    r = match.iloc[0]
    worst_rows.append({
        "tweet_id": r["tweet_id"],
        "customer_message": r["customer_message"],
        "gold_intent": r["gold_intent"],
        "predicted_intent": r["predicted_intent"],
        "intent_correct": r["intent_correct"],
        "reply": r["reply"],
        "evidence_used": r["evidence_used"],
        "evidence_safe": r["evidence_safe"],
        "evidence_relevance": r["evidence_relevance"],
        "problem_match": r["problem_match"],
        "correctness_score": r["correctness_score"],
        "groundedness_score": r["groundedness_score"],
        "relevance_score": r["relevance_score"],
        "completeness_score": r["completeness_score"],
        "tone_score": r["tone_score"],
        "unsupported_claims_score": r["unsupported_claims_score"],
        "overall_score": r["overall_score"],
        "overall_judge_decision": r["judge_decision"],
        "judge_reason": r["judge_reason"],
        "escalation_decision": r["escalation_decision"],
        "escalation_reason": r["escalation_reason"],
        "failure_classification": f_class,
        "failure_explanation": f_expl,
    })

failure_df = pd.DataFrame(worst_rows)
out_path = PROJECT_ROOT / "evaluation/results/judge_failure_analysis.csv"
failure_df.to_csv(out_path, index=False)
print(f"Saved 10 worst cases failure analysis to: {out_path}")
print(f"Total rows: {len(failure_df)}")
