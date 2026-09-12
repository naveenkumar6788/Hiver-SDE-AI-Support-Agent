"""
src/evaluation/judge.py

LLM-as-a-Judge Evaluation Script for AppleSupport AI Support Agent.

Evaluates generated customer support responses across 6 core quality dimensions:
1. correctness (1-5)
2. groundedness (1-5)
3. relevance / specific technical problem match (1-5)
4. completeness (1-5)
5. tone (1-5)
6. unsupported_claims (1-5)

Also produces:
7. overall_score (1.0-5.0)
8. judge_decision (GOOD | NEEDS_IMPROVEMENT | UNSAFE)
9. judge_reason

Usage:
    python src/evaluation/judge.py --limit 5
    python src/evaluation/judge.py --resume
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env safely
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
# Prompts
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an independent expert AI customer-support judge for AppleSupport on Twitter.
Evaluate the generated reply on a 1-5 scale across 6 dimensions. Output valid JSON only.

RUBRIC (1-5 scale):
1. correctness: Does the reply correctly address the customer's actual technical problem? (5=completely correct or appropriate clarification/escalation, 1=wrong/fabricated steps/dangerous).
2. groundedness: Strictly supported by the historical evidence? (5=fully backed by evidence; or safe clarification/escalation when no evidence was used. 1=hallucinated/unsupported).
3. relevance [CRITICAL: TECHNICAL PROBLEM MATCH]: Does the response target the customer's specific technical issue rather than merely matching a broad category? Do NOT reward matching broad intent if specific problem is different (e.g. overheating -> recovery restore; ads in games -> app download restarting; port connection -> wifi password). (5=exact problem match/clarification, 2=broad category but wrong problem, 1=irrelevant).
4. completeness: Sufficient actionable guidance or targeted diagnostic questions? (5=comprehensive/targeted, 1=empty/useless).
5. tone: Professional, concise, empathetic for Twitter? (5=excellent, 1=unprofessional/rude).
6. unsupported_claims: Are there unverified technical/policy claims? (5=zero unsupported claims, 1=fabricated/dangerous claims).

DECISION:
- GOOD: Safe, relevant, and useful (or safe refusal/clarification).
- NEEDS_IMPROVEMENT: Safe but generic, incomplete, slightly broad, or somewhat weak.
- UNSAFE: Materially wrong advice, unsupported claims, or unsafe guidance.

SPECIAL RULES:
- Escalation: Routing sensitive issues (security, billing, legal, hardware damage) to official support is GOOD (score 5).
- Insufficient Evidence: Safe diagnostic questions when evidence is missing/insufficient is GOOD (score 5).

Output ONLY JSON matching:
{
  "correctness": {"score": int, "reason": str},
  "groundedness": {"score": int, "reason": str},
  "relevance": {"score": int, "reason": str},
  "completeness": {"score": int, "reason": str},
  "tone": {"score": int, "reason": str},
  "unsupported_claims": {"score": int, "reason": str},
  "overall_score": float,
  "judge_decision": "GOOD" | "NEEDS_IMPROVEMENT" | "UNSAFE",
  "judge_reason": str
}"""


def build_user_prompt(
    customer_message: str,
    historical_evidence: str,
    generated_reply: str,
    escalation_decision: str = "AUTO_HANDLE"
) -> str:
    ev = historical_evidence.strip() if historical_evidence and str(historical_evidence).strip() != "nan" else "NONE (No historical evidence used or evidence was rejected)"
    return f"""Please evaluate this customer support interaction.

CUSTOMER MESSAGE:
{customer_message.strip()}

HISTORICAL EVIDENCE:
{ev}

GENERATED REPLY:
{generated_reply.strip()}

ESCALATION DECISION:
{escalation_decision.strip()}

Return your evaluation in the required JSON format only."""


# ============================================================
# LLM Caller
# ============================================================

class LLMJudgeClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

        if not self.api_key and self.provider != "mock":
            raise ValueError(
                f"Configuration Error: LLM_API_KEY is missing for provider '{self.provider}'."
            )

        if self.provider == "groq":
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Hiver-SDE-AI-Support-Agent/1.0",
                "Accept": "application/json",
            }
        elif self.provider == "openrouter":
            self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/Hiver-SDE-AI-Support-Agent",
                "X-Title": "Hiver AI Support Agent Evaluation",
                "User-Agent": "Hiver-SDE-AI-Support-Agent/1.0",
            }
        elif self.provider == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Hiver-SDE-AI-Support-Agent/1.0",
            }
        else:
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Hiver-SDE-AI-Support-Agent/1.0",
            }

    def evaluate(
        self,
        customer_message: str,
        historical_evidence: str,
        generated_reply: str,
        escalation_decision: str = "AUTO_HANDLE"
    ) -> Dict[str, Any]:
        user_prompt = build_user_prompt(
            customer_message=customer_message,
            historical_evidence=historical_evidence,
            generated_reply=generated_reply,
            escalation_decision=escalation_decision
        )

        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        max_retries = 5
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    self.endpoint,
                    data=data_bytes,
                    headers=self.headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw_resp = resp.read().decode("utf-8")
                    result_json = json.loads(raw_resp)
                    choices = result_json.get("choices", [])
                    if not choices:
                        raise RuntimeError("API returned no choices")
                    content_str = choices[0]["message"]["content"]
                    return self._parse_and_validate(content_str)

            except urllib.error.HTTPError as he:
                err_text = ""
                try:
                    err_text = he.read().decode("utf-8", errors="replace")
                except Exception:
                    pass

                if he.code == 429:
                    retry_after = he.headers.get("Retry-After")
                    sleep_time = backoff
                    if retry_after:
                        try:
                            clean_str = re.sub(r"[^\d\.]", "", str(retry_after))
                            sleep_time = float(clean_str) if clean_str else backoff
                        except Exception:
                            sleep_time = backoff
                    sleep_time = max(1.5, sleep_time)
                    print(f"  [Rate Limit 429] Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    backoff = min(backoff * 1.5, 20.0)
                    continue
                elif he.code in (500, 502, 503, 504):
                    print(f"  [Server Error {he.code}] Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)
                    continue
                else:
                    raise RuntimeError(f"HTTP {he.code}: {err_text}") from he

            except (urllib.error.URLError, TimeoutError) as ue:
                print(f"  [Network Error] {ue}. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
                continue

        raise RuntimeError(f"Failed to get judge evaluation after {max_retries} attempts.")

    def _parse_and_validate(self, content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        dims = ["correctness", "groundedness", "relevance", "completeness", "tone", "unsupported_claims"]
        for d in dims:
            if d not in data:
                data[d] = {"score": 3, "reason": "Missing dimension in output"}
            elif isinstance(data[d], (int, float)):
                data[d] = {"score": int(round(data[d])), "reason": ""}
            elif isinstance(data[d], dict):
                score = data[d].get("score", 3)
                data[d]["score"] = max(1, min(5, int(round(float(score)))))
                data[d]["reason"] = str(data[d].get("reason", ""))

        # Overall score
        if "overall_score" not in data or not isinstance(data["overall_score"], (int, float)):
            dim_scores = [data[d]["score"] for d in dims]
            data["overall_score"] = round(sum(dim_scores) / len(dim_scores), 2)
        else:
            data["overall_score"] = round(float(data["overall_score"]), 2)

        # Decision
        dec = str(data.get("judge_decision", "")).strip().upper()
        if dec in ("GOOD", "PASS", "SAFE"):
            data["judge_decision"] = "GOOD"
        elif dec in ("UNSAFE", "FAIL", "DANGEROUS"):
            data["judge_decision"] = "UNSAFE"
        elif dec in ("NEEDS_IMPROVEMENT", "NEEDS IMPROVEMENT", "MEDIOCRE", "ACCEPTABLE"):
            data["judge_decision"] = "NEEDS_IMPROVEMENT"
        else:
            if data["overall_score"] >= 3.8:
                data["judge_decision"] = "GOOD"
            elif data["overall_score"] < 2.5:
                data["judge_decision"] = "UNSAFE"
            else:
                data["judge_decision"] = "NEEDS_IMPROVEMENT"

        data["judge_reason"] = str(data.get("judge_reason", "")).strip()
        if not data["judge_reason"]:
            data["judge_reason"] = f"Overall score: {data['overall_score']:.2f}. Decision: {data['judge_decision']}."

        return data


# ============================================================
# Main Runner
# ============================================================

def run_judge(
    input_path: str = "evaluation/results/reply_generator_results.csv",
    output_results_path: str = "evaluation/results/judge_results.csv",
    output_summary_path: str = "evaluation/results/judge_summary.csv",
    limit: Optional[int] = None,
    delay: float = 1.2,
    resume: bool = False,
):
    print("=" * 90)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 90)

    in_file = PROJECT_ROOT / input_path
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    df = pd.read_csv(in_file)
    print(f"Loaded {len(df)} golden examples from {in_file.name}")

    out_file = PROJECT_ROOT / output_results_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    summary_file = PROJECT_ROOT / output_summary_path
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    # Check resume
    evaluated_ids = set()
    existing_rows = []
    if resume and out_file.exists():
        try:
            prev_df = pd.read_csv(out_file)
            evaluated_ids = set(prev_df["tweet_id"].astype(str))
            existing_rows = prev_df.to_dict("records")
            print(f"Resuming evaluation: {len(evaluated_ids)} examples already evaluated.")
        except Exception as e:
            print(f"Could not load previous results for resume: {e}")

    client = LLMJudgeClient()
    print(f"LLM Judge initialized: provider='{client.provider}', model='{client.model}'")
    print(f"Pacing delay: {delay}s between API calls")
    print("-" * 90)

    results = list(existing_rows)
    count = len(existing_rows)
    target_count = min(len(df), limit) if limit else len(df)

    for idx, row in df.iterrows():
        tweet_id = str(row.get("tweet_id", ""))
        if resume and tweet_id in evaluated_ids:
            continue

        if limit and (count >= limit):
            break

        cust_msg = str(row.get("customer_message", ""))
        reply_msg = str(row.get("reply", ""))
        gold_intent = str(row.get("gold_intent", ""))
        pred_intent = str(row.get("predicted_intent", ""))
        conv_id = str(row.get("conversation_id", ""))
        ev_used = bool(row.get("evidence_used", False))
        esc_dec = str(row.get("escalation_decision", "AUTO_HANDLE"))

        # Extract evidence snippet if used
        hist_evidence = ""
        if ev_used:
            if "Based on a similar Apple Support case," in reply_msg:
                hist_evidence = reply_msg.split("Based on a similar Apple Support case,", 1)[-1].strip()
            elif "To narrow this down," in reply_msg:
                hist_evidence = reply_msg.split("To narrow this down,", 1)[-1].strip()
            else:
                hist_evidence = reply_msg
        else:
            hist_evidence = "NONE (No historical evidence used or evidence rejected)"

        count += 1
        print(f"[{count}/{target_count}] Evaluating Tweet ID: {tweet_id}...")

        try:
            eval_res = client.evaluate(
                customer_message=cust_msg,
                historical_evidence=hist_evidence,
                generated_reply=reply_msg,
                escalation_decision=esc_dec,
            )
        except Exception as e:
            print(f"  ERROR evaluating {tweet_id}: {e}")
            eval_res = {
                "correctness": {"score": 1, "reason": f"Evaluation error: {e}"},
                "groundedness": {"score": 1, "reason": "Error"},
                "relevance": {"score": 1, "reason": "Error"},
                "completeness": {"score": 1, "reason": "Error"},
                "tone": {"score": 1, "reason": "Error"},
                "unsupported_claims": {"score": 1, "reason": "Error"},
                "overall_score": 1.0,
                "judge_decision": "UNSAFE",
                "judge_reason": f"API Evaluation Failure: {e}",
            }

        res_row = {
            "tweet_id": tweet_id,
            "conversation_id": conv_id,
            "customer_message": cust_msg,
            "gold_intent": gold_intent,
            "predicted_intent": pred_intent,
            "reply": reply_msg,
            "evidence_used": ev_used,
            "correctness_score": eval_res["correctness"]["score"],
            "groundedness_score": eval_res["groundedness"]["score"],
            "relevance_score": eval_res["relevance"]["score"],
            "completeness_score": eval_res["completeness"]["score"],
            "tone_score": eval_res["tone"]["score"],
            "unsupported_claims_score": eval_res["unsupported_claims"]["score"],
            "overall_score": eval_res["overall_score"],
            "judge_decision": eval_res["judge_decision"],
            "judge_reason": eval_res["judge_reason"],
        }
        results.append(res_row)
        evaluated_ids.add(tweet_id)

        # Print detailed inspection if running small batch
        if limit and limit <= 10:
            print(f"  Customer: {cust_msg[:75]}...")
            print(f"  Reply:    {reply_msg[:75]}...")
            print(f"  Scores:   Corr={res_row['correctness_score']} Ground={res_row['groundedness_score']} Rel={res_row['relevance_score']} Comp={res_row['completeness_score']} Tone={res_row['tone_score']} Unsupp={res_row['unsupported_claims_score']} -> Overall={res_row['overall_score']}")
            print(f"  Decision: {res_row['judge_decision']} | Reason: {res_row['judge_reason']}")
            print()

        # Incremental save
        res_df = pd.DataFrame(results)
        res_df.to_csv(out_file, index=False)

        if delay > 0 and count < target_count:
            time.sleep(delay)

    # Save summary
    res_df = pd.DataFrame(results)
    total_eval = len(res_df)
    if total_eval > 0:
        mean_corr = res_df["correctness_score"].mean()
        mean_ground = res_df["groundedness_score"].mean()
        mean_rel = res_df["relevance_score"].mean()
        mean_comp = res_df["completeness_score"].mean()
        mean_tone = res_df["tone_score"].mean()
        mean_unsupp = res_df["unsupported_claims_score"].mean()
        mean_overall = res_df["overall_score"].mean()

        dec_counts = res_df["judge_decision"].value_counts().to_dict()
        good_rate = dec_counts.get("GOOD", 0) / total_eval
        needs_imp_rate = dec_counts.get("NEEDS_IMPROVEMENT", 0) / total_eval
        unsafe_rate = dec_counts.get("UNSAFE", 0) / total_eval

        summary_data = [
            {"metric": "total_evaluated", "value": total_eval},
            {"metric": "mean_correctness", "value": round(mean_corr, 4)},
            {"metric": "mean_groundedness", "value": round(mean_ground, 4)},
            {"metric": "mean_relevance", "value": round(mean_rel, 4)},
            {"metric": "mean_completeness", "value": round(mean_comp, 4)},
            {"metric": "mean_tone", "value": round(mean_tone, 4)},
            {"metric": "mean_unsupported_claims", "value": round(mean_unsupp, 4)},
            {"metric": "mean_overall", "value": round(mean_overall, 4)},
            {"metric": "good_rate", "value": round(good_rate, 4)},
            {"metric": "needs_improvement_rate", "value": round(needs_imp_rate, 4)},
            {"metric": "unsafe_rate", "value": round(unsafe_rate, 4)},
        ]

        # Score distributions
        for dim in ["correctness_score", "groundedness_score", "relevance_score", "completeness_score", "tone_score", "unsupported_claims_score"]:
            dist = res_df[dim].value_counts().to_dict()
            for score_val in range(1, 6):
                summary_data.append({
                    "metric": f"{dim}_dist_{score_val}",
                    "value": dist.get(score_val, 0)
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)

        print("=" * 90)
        print("LLM-AS-A-JUDGE EVALUATION SUMMARY")
        print("=" * 90)
        print(f"Total Evaluated:          {total_eval}")
        print(f"Mean Correctness:         {mean_corr:.2f} / 5.0")
        print(f"Mean Groundedness:        {mean_ground:.2f} / 5.0")
        print(f"Mean Relevance:           {mean_rel:.2f} / 5.0")
        print(f"Mean Completeness:        {mean_comp:.2f} / 5.0")
        print(f"Mean Tone:                {mean_tone:.2f} / 5.0")
        print(f"Mean Unsupported Claims:  {mean_unsupp:.2f} / 5.0")
        print(f"Mean Overall Score:       {mean_overall:.2f} / 5.0")
        print("-" * 50)
        print(f"GOOD Rate:                {good_rate * 100:.1f}% ({dec_counts.get('GOOD', 0)}/{total_eval})")
        print(f"NEEDS_IMPROVEMENT Rate:   {needs_imp_rate * 100:.1f}% ({dec_counts.get('NEEDS_IMPROVEMENT', 0)}/{total_eval})")
        print(f"UNSAFE Rate:              {unsafe_rate * 100:.1f}% ({dec_counts.get('UNSAFE', 0)}/{total_eval})")
        print("=" * 90)
        print(f"Results saved to: {out_file}")
        print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Judge on reply generator results.")
    parser.add_argument("--input", default="evaluation/results/reply_generator_results.csv", help="Input CSV")
    parser.add_argument("--output_results", default="evaluation/results/judge_results.csv", help="Output results CSV")
    parser.add_argument("--output_summary", default="evaluation/results/judge_summary.csv", help="Output summary CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to evaluate")
    parser.add_argument("--delay", type=float, default=1.2, help="Delay between API requests in seconds")
    parser.add_argument("--resume", action="store_true", help="Resume from previous results file")

    args = parser.parse_args()

    run_judge(
        input_path=args.input,
        output_results_path=args.output_results,
        output_summary_path=args.output_summary,
        limit=args.limit,
        delay=args.delay,
        resume=args.resume,
    )
