import os
import sys
import json
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# -------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------

from src.retrieval.retriever import AppleSupportRetriever
from src.agent.reply_generator import (
    ReplyGenerator,
    is_evidence_safe,
)


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

GOLDEN_PATH = (
    PROJECT_ROOT
    / "golden_set"
    / "golden_human_160.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FULL_RESULTS_PATH = (
    RESULTS_DIR
    / "reply_generator_results.csv"
)

FAILURE_PATH = (
    RESULTS_DIR
    / "reply_generator_failure_audit.csv"
)

ESCALATION_PATH = (
    RESULTS_DIR
    / "reply_generator_escalation_cases.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "reply_generator_summary.csv"
)


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

TOP_K = 5


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return default

        value = float(value)

        if np.isnan(value):
            return default

        return value

    except Exception:
        return default


def normalize_intent(value):
    """
    Normalize intent labels so small formatting differences
    do not create artificial evaluation errors.
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    aliases = {
        "ios_software_update": "ios_update",
        "ios_update": "ios_update",

        "battery": "battery_charging",
        "battery_charging": "battery_charging",

        "wifi": "wifi_connectivity",
        "wifi_connectivity": "wifi_connectivity",

        "app": "app_problems",
        "app_problem": "app_problems",
        "app_problems": "app_problems",

        "app_store": "app_store_downloads",
        "app_store_download": "app_store_downloads",
        "app_store_downloads": "app_store_downloads",

        "apple_id": "apple_id_icloud",
        "icloud": "apple_id_icloud",
        "apple_id_icloud": "apple_id_icloud",

        "music": "apple_music_itunes",
        "itunes": "apple_music_itunes",
        "apple_music": "apple_music_itunes",
        "apple_music_itunes": "apple_music_itunes",

        "screen": "screen_display",
        "display": "screen_display",
        "screen_display": "screen_display",

        "calls": "calls_cellular",
        "cellular": "calls_cellular",
        "calls_cellular": "calls_cellular",

        "audio": "audio_speaker",
        "speaker": "audio_speaker",
        "audio_speaker": "audio_speaker",

        "hardware": "device_hardware",
        "device_hardware": "device_hardware",

        "other": "other_unclear",
        "unclear": "other_unclear",
        "other_unclear": "other_unclear",
    }

    return aliases.get(value, value)


def get_gold_intent(row):
    """
    Pick the correct final evaluation label.

    Priority:
        1. verified_intent
        2. human_intent
        3. golden_intent
        4. intent
    """

    columns = [
        "verified_intent",
        "human_intent",
        "golden_intent",
        "intent",
    ]

    for column in columns:

        if column not in row:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            return normalize_intent(value)

    return ""


def get_customer_message(row):
    """
    Extract customer message from the golden-set row.
    """

    for column in ["text", "customer_message", "message"]:

        if column not in row:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def get_conversation_id(row):
    """
    Extract conversation ID.
    """

    for column in [
        "conversation_id",
        "conversation",
        "thread_id",
    ]:

        if column not in row:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def get_tweet_id(row):
    """
    Extract tweet ID.
    """

    if "tweet_id" in row:

        value = row["tweet_id"]

        if not pd.isna(value):
            return str(value)

    return ""


def extract_evidence_from_result(result):
    """
    Extract the evidence object from ReplyGenerator output.
    """

    evidence = result.get("evidence")

    if isinstance(evidence, dict) and evidence:
        return evidence

    evidence_list = result.get("evidence_candidates")

    if isinstance(evidence_list, list) and evidence_list:

        first = evidence_list[0]

        if isinstance(first, dict):
            return first

    if result.get("evidence_used") or result.get("selected_evidence_text"):
        return {
            "customer_text": result.get("evidence_customer_text", ""),
            "response_text": result.get("selected_evidence_text", ""),
            "historical_response": result.get("selected_evidence_text", ""),
            "evidence_relevance_score": result.get("evidence_relevance", 0.0),
            "evidence_quality_score": result.get("evidence_quality", 0.0),
            "problem_match_score": result.get("problem_match_score", 0.0),
            "evidence_safe": result.get("evidence_safe", True),
        }

    return {}


def evaluate_evidence(
    query,
    evidence,
):
    """
    Evaluate whether the returned evidence is safe and useful.
    """

    if not evidence:
        return {
            "evidence_used": False,
            "evidence_safe": False,
            "evidence_relevance": 0.0,
            "evidence_quality": 0.0,
            "problem_match": 0.0,
            "customer_problem_similarity": 0.0,
            "response_problem_similarity": 0.0,
            "domain_match": 0.0,
            "action_match": 0.0,
            "conflict_detected": False,
            "generic_response": False,
            "final_evidence_score": 0.0,
            "evidence_rejection_reason": "no_evidence",
        }

    customer_text = str(
        evidence.get(
            "customer_text",
            evidence.get(
                "query",
                "",
            ),
        )
    )

    response_text = str(
        evidence.get(
            "response_text",
            evidence.get(
                "response",
                evidence.get(
                    "historical_response",
                    "",
                ),
            ),
        )
    )

    try:

        res = is_evidence_safe(
            query=query,
            evidence_candidate=evidence,
            evidence_customer=customer_text,
            evidence_response=response_text,
        )
        safe = res[0] if isinstance(res, (tuple, list)) else bool(res)

    except Exception:

        safe = bool(
            evidence.get(
                "evidence_safe",
                False,
            )
        )

    relevance = safe_float(
        evidence.get(
            "evidence_relevance_score",
            evidence.get(
                "evidence_relevance",
                evidence.get(
                    "similarity",
                    0.0,
                ),
            ),
        )
    )

    quality = safe_float(
        evidence.get(
            "evidence_quality_score",
            evidence.get(
                "evidence_quality",
                0.0,
            ),
        )
    )

    problem_match = safe_float(
        evidence.get(
            "problem_match_score",
            evidence.get(
                "problem_match",
                0.0,
            ),
        )
    )

    return {
        "evidence_used": True,
        "evidence_safe": bool(safe),
        "evidence_relevance": relevance,
        "evidence_quality": quality,
        "problem_match": problem_match,
        "customer_problem_similarity": safe_float(evidence.get("customer_problem_similarity", 0.0)),
        "response_problem_similarity": safe_float(evidence.get("response_problem_similarity", 0.0)),
        "domain_match": safe_float(evidence.get("domain_match", 0.0)),
        "action_match": safe_float(evidence.get("action_match", 0.0)),
        "conflict_detected": bool(evidence.get("conflict_detected", False)),
        "generic_response": bool(evidence.get("generic_response", False)),
        "final_evidence_score": safe_float(evidence.get("final_evidence_score", relevance)),
        "evidence_rejection_reason": str(evidence.get("evidence_rejection_reason", evidence.get("rejection_reason", ""))),
    }


def classify_escalation(result):
    """
    Extract escalation decision from the generator output.

    Handles several possible output field names so the evaluator
    remains compatible with the current ReplyGenerator.
    """

    decision = str(
        result.get(
            "escalation_decision",
            result.get(
                "decision",
                result.get(
                    "escalation",
                    "",
                ),
            ),
        )
    ).strip().upper()

    if decision in {
        "AUTO_HANDLE",
        "AUTO-HANDLE",
        "HANDLE",
        "AUTO",
    }:
        decision = "AUTO_HANDLE"

    elif decision in {
        "ESCALATE",
        "ESCALATED",
    }:
        decision = "ESCALATE"

    else:
        # Try boolean fields.
        escalate = result.get("escalate")

        if isinstance(escalate, bool):

            decision = (
                "ESCALATE"
                if escalate
                else "AUTO_HANDLE"
            )

        else:
            decision = "UNKNOWN"

    reason = str(
        result.get(
            "escalation_reason",
            result.get(
                "reason",
                "",
            ),
        )
    ).strip()

    return decision, reason


# -------------------------------------------------------------------
# LOAD GOLDEN SET
# -------------------------------------------------------------------

print("=" * 90)
print("APPLE SUPPORT REPLY GENERATOR EVALUATION")
print("=" * 90)

print()
print("Loading golden evaluation set...")

if not GOLDEN_PATH.exists():

    print()
    print("ERROR: Golden file not found:")
    print(GOLDEN_PATH)
    print()

    sys.exit(1)


golden_df = pd.read_csv(
    GOLDEN_PATH
)

print(
    f"Golden examples loaded: {len(golden_df)}"
)

print()
print("Available columns:")

for column in golden_df.columns:
    print(f"  - {column}")


# -------------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# -------------------------------------------------------------------

required_columns = [
    "text",
    "conversation_id",
]

missing_columns = [
    column
    for column in required_columns
    if column not in golden_df.columns
]

if missing_columns:

    print()
    print(
        "WARNING: Missing expected columns:",
        missing_columns,
    )


# -------------------------------------------------------------------
# INITIALIZE RETRIEVER
# -------------------------------------------------------------------

print()
print("=" * 90)
print("INITIALIZING RETRIEVER")
print("=" * 90)

retriever = AppleSupportRetriever()

generator = ReplyGenerator(
    retriever=retriever
)

print()
print("Reply generator initialized.")


# -------------------------------------------------------------------
# EVALUATION STORAGE
# -------------------------------------------------------------------

results = []
failures = []
escalation_cases = []

y_true = []
y_pred = []


# -------------------------------------------------------------------
# RUN EVALUATION
# -------------------------------------------------------------------

print()
print("=" * 90)
print("RUNNING EVALUATION")
print("=" * 90)

total = len(golden_df)

for index, row in golden_df.iterrows():

    example_number = index + 1

    tweet_id = get_tweet_id(row)

    conversation_id = get_conversation_id(
        row
    )

    customer_message = get_customer_message(
        row
    )

    gold_intent = get_gold_intent(
        row
    )

    # ---------------------------------------------------------------
    # EMPTY MESSAGE CHECK
    # ---------------------------------------------------------------

    if not customer_message:

        failures.append(
            {
                "index": example_number,
                "tweet_id": tweet_id,
                "conversation_id": conversation_id,
                "failure_type": "empty_message",
                "error": "Customer message is empty",
            }
        )

        continue

    # ---------------------------------------------------------------
    # GENERATE
    # ---------------------------------------------------------------

    try:

        generated = generator.generate(
            query=customer_message,
            intent=None,
            conversation_id=conversation_id,
            top_k=TOP_K,
        )

        if not isinstance(generated, dict):

            raise TypeError(
                "ReplyGenerator.generate() "
                "did not return a dictionary"
            )

    except Exception as exc:

        error_message = str(exc)

        print(
            f"[{example_number}] Generation failed: "
            f"{error_message}"
        )

        failures.append(
            {
                "index": example_number,
                "tweet_id": tweet_id,
                "conversation_id": conversation_id,
                "customer_message": customer_message,
                "gold_intent": gold_intent,
                "failure_type": "generation_error",
                "error": error_message,
            }
        )

        # Keep an evaluation row even when generation fails.
        results.append(
            {
                "index": example_number,
                "tweet_id": tweet_id,
                "conversation_id": conversation_id,
                "customer_message": customer_message,
                "gold_intent": gold_intent,
                "predicted_intent": "",
                "intent_correct": False,
                "reply": "",
                "evidence_used": False,
                "evidence_safe": False,
                "evidence_relevance": 0.0,
                "evidence_quality": 0.0,
                "problem_match": 0.0,
                "grounded": False,
                "unsupported_claims": "",
                "response_type": "generation_error",
                "escalation_decision": "UNKNOWN",
                "escalation_reason": error_message,
                "generation_error": error_message,
            }
        )

        continue

    # ---------------------------------------------------------------
    # INTENT
    # ---------------------------------------------------------------

    predicted_intent = normalize_intent(
        generated.get(
            "intent",
            generated.get(
                "predicted_intent",
                "",
            ),
        )
    )

    intent_correct = (
        predicted_intent == gold_intent
        if gold_intent
        else False
    )

    if gold_intent:

        y_true.append(
            gold_intent
        )

        y_pred.append(
            predicted_intent
        )

    # ---------------------------------------------------------------
    # REPLY
    # ---------------------------------------------------------------

    reply = str(
        generated.get(
            "reply",
            generated.get(
                "response",
                "",
            ),
        )
    )

    # ---------------------------------------------------------------
    # EVIDENCE
    # ---------------------------------------------------------------

    evidence = extract_evidence_from_result(
        generated
    )

    evidence_metrics = evaluate_evidence(
        query=customer_message,
        evidence=evidence,
    )

    # If generator explicitly says evidence was used,
    # preserve that information.
    if generated.get("evidence_used") is True:

        evidence_metrics["evidence_used"] = True

    # ---------------------------------------------------------------
    # GROUNDING
    # ---------------------------------------------------------------

    grounded = bool(
        generated.get(
            "grounded",
            generated.get(
                "is_grounded",
                False,
            ),
        )
    )

    unsupported_claims = generated.get(
        "unsupported_claims",
        [],
    )

    if isinstance(
        unsupported_claims,
        list,
    ):

        unsupported_claims_text = " | ".join(
            str(x)
            for x in unsupported_claims
        )

    else:

        unsupported_claims_text = str(
            unsupported_claims
        )

    # ---------------------------------------------------------------
    # RESPONSE TYPE
    # ---------------------------------------------------------------

    response_type = str(
        generated.get(
            "response_type",
            "",
        )
    )

    # ---------------------------------------------------------------
    # ESCALATION
    # ---------------------------------------------------------------

    escalation_decision, escalation_reason = (
        classify_escalation(
            generated
        )
    )

    if escalation_decision == "ESCALATE":

        escalation_cases.append(
            {
                "index": example_number,
                "tweet_id": tweet_id,
                "conversation_id": conversation_id,
                "customer_message": customer_message,
                "gold_intent": gold_intent,
                "predicted_intent": predicted_intent,
                "reason": escalation_reason,
                "response_type": response_type,
            }
        )

    # ---------------------------------------------------------------
    # SAFE TO ANSWER
    # ---------------------------------------------------------------

    safe_to_answer = (
        evidence_metrics["evidence_used"]
        and
        evidence_metrics["evidence_safe"]
        and
        grounded
        and
        not bool(unsupported_claims_text.strip())
    )

    # ---------------------------------------------------------------
    # RESULT ROW
    # ---------------------------------------------------------------

    results.append(
        {
            "index": example_number,
            "tweet_id": tweet_id,
            "conversation_id": conversation_id,
            "customer_message": customer_message,
            "gold_intent": gold_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": intent_correct,
            "reply": reply,
            "evidence_used": evidence_metrics[
                "evidence_used"
            ],
            "evidence_safe": evidence_metrics[
                "evidence_safe"
            ],
            "evidence_relevance": evidence_metrics[
                "evidence_relevance"
            ],
            "evidence_quality": evidence_metrics[
                "evidence_quality"
            ],
            "problem_match": evidence_metrics[
                "problem_match"
            ],
            "customer_problem_similarity": evidence_metrics.get("customer_problem_similarity", 0.0),
            "response_problem_similarity": evidence_metrics.get("response_problem_similarity", 0.0),
            "intent_match": evidence_metrics.get("intent_match", intent_correct),
            "domain_match": evidence_metrics.get("domain_match", 0.0),
            "action_match": evidence_metrics.get("action_match", 0.0),
            "conflict_detected": evidence_metrics.get("conflict_detected", False),
            "generic_response": evidence_metrics.get("generic_response", False),
            "final_evidence_score": evidence_metrics.get("final_evidence_score", 0.0),
            "evidence_rejection_reason": evidence_metrics.get("evidence_rejection_reason", ""),
            "grounded": grounded,
            "unsupported_claims": unsupported_claims_text,
            "response_type": response_type,
            "escalation_decision": escalation_decision,
            "escalation_reason": escalation_reason,
            "safe_to_answer": safe_to_answer,
            "generation_error": "",
        }
    )

    # ---------------------------------------------------------------
    # PROGRESS
    # ---------------------------------------------------------------

    if example_number % 10 == 0:

        print(
            f"Processed {example_number}/{total}"
        )


# -------------------------------------------------------------------
# SAVE FULL RESULTS
# -------------------------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    FULL_RESULTS_PATH,
    index=False,
)

print()
print(
    f"Saved full results: "
    f"{FULL_RESULTS_PATH}"
)


# -------------------------------------------------------------------
# SAVE FAILURE AUDIT
# -------------------------------------------------------------------

failures_df = pd.DataFrame(
    failures
)

failures_df.to_csv(
    FAILURE_PATH,
    index=False,
)

print(
    f"Saved failure audit: "
    f"{FAILURE_PATH}"
)


# -------------------------------------------------------------------
# SAVE ESCALATION CASES
# -------------------------------------------------------------------

escalation_df = pd.DataFrame(
    escalation_cases
)

escalation_df.to_csv(
    ESCALATION_PATH,
    index=False,
)

print(
    f"Saved escalation cases: "
    f"{ESCALATION_PATH}"
)


# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------

print()
print("=" * 90)
print("EVALUATION RESULTS")
print("=" * 90)


# -------------------------------------------------------------------
# INTENT METRICS
# -------------------------------------------------------------------

if y_true:

    intent_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    intent_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

else:

    intent_accuracy = 0.0
    intent_macro_f1 = 0.0
    intent_weighted_f1 = 0.0


print()
print(
    f"Intent Accuracy: "
    f"{intent_accuracy * 100:.2f}%"
)

print(
    f"Intent Macro F1: "
    f"{intent_macro_f1:.4f}"
)

print(
    f"Intent Weighted F1: "
    f"{intent_weighted_f1:.4f}"
)


# -------------------------------------------------------------------
# EVIDENCE METRICS
# -------------------------------------------------------------------

if len(results_df) > 0:

    evidence_used_count = int(
        results_df[
            "evidence_used"
        ].sum()
    )

    evidence_safe_count = int(
        results_df[
            "evidence_safe"
        ].sum()
    )

    grounded_count = int(
        results_df[
            "grounded"
        ].sum()
    )

    unsupported_count = int(
        (
            results_df[
                "unsupported_claims"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ).sum()
    )

    safe_to_answer_count = int(
        results_df[
            "safe_to_answer"
        ].sum()
    )

    avg_evidence_relevance = (
        results_df[
            "evidence_relevance"
        ].mean()
    )

    avg_evidence_quality = (
        results_df[
            "evidence_quality"
        ].mean()
    )

    avg_problem_match = (
        results_df[
            "problem_match"
        ].mean()
    )

    generation_error_count = int(
        (
            results_df[
                "generation_error"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ).sum()
    )

else:

    evidence_used_count = 0
    evidence_safe_count = 0
    grounded_count = 0
    unsupported_count = 0
    safe_to_answer_count = 0
    avg_evidence_relevance = 0.0
    avg_evidence_quality = 0.0
    avg_problem_match = 0.0
    generation_error_count = 0


total_results = len(
    results_df
)

if total_results > 0:

    evidence_used_rate = (
        evidence_used_count
        / total_results
    )

    evidence_safe_rate = (
        evidence_safe_count
        / total_results
    )

    grounded_rate = (
        grounded_count
        / total_results
    )

    unsupported_rate = (
        unsupported_count
        / total_results
    )

    safe_to_answer_rate = (
        safe_to_answer_count
        / total_results
    )

else:

    evidence_used_rate = 0.0
    evidence_safe_rate = 0.0
    grounded_rate = 0.0
    unsupported_rate = 0.0
    safe_to_answer_rate = 0.0


print()
print(
    f"Evidence Used: "
    f"{evidence_used_count}/{total_results} "
    f"({evidence_used_rate * 100:.2f}%)"
)

print(
    f"Evidence Safe: "
    f"{evidence_safe_count}/{total_results} "
    f"({evidence_safe_rate * 100:.2f}%)"
)

print(
    f"Average Evidence Relevance: "
    f"{avg_evidence_relevance:.4f}"
)

print(
    f"Average Evidence Quality: "
    f"{avg_evidence_quality:.4f}"
)

print(
    f"Average Problem Match: "
    f"{avg_problem_match:.4f}"
)

avg_cust_prob = results_df["customer_problem_similarity"].mean() if "customer_problem_similarity" in results_df else 0.0
avg_resp_prob = results_df["response_problem_similarity"].mean() if "response_problem_similarity" in results_df else 0.0
spec_match_count = (results_df["problem_match"] >= 0.70).sum() if "problem_match" in results_df else 0
conflict_count = results_df["conflict_detected"].sum() if "conflict_detected" in results_df else 0

print(
    f"Average Customer Problem Sim: "
    f"{avg_cust_prob:.4f}"
)
print(
    f"Average Response Problem Sim: "
    f"{avg_resp_prob:.4f}"
)
print(
    f"Specific Problem Match Rate: "
    f"{spec_match_count}/{total_results} ({spec_match_count / total_results * 100:.2f}%)"
)
print(
    f"Problem Conflict Detected: "
    f"{conflict_count}/{total_results} ({conflict_count / total_results * 100:.2f}%)"
)

print(
    f"Grounded: "
    f"{grounded_count}/{total_results} "
    f"({grounded_rate * 100:.2f}%)"
)

print(
    f"Unsupported Claims: "
    f"{unsupported_count}/{total_results} "
    f"({unsupported_rate * 100:.2f}%)"
)

print(
    f"Safe To Answer: "
    f"{safe_to_answer_count}/{total_results} "
    f"({safe_to_answer_rate * 100:.2f}%)"
)

print(
    f"Generation Errors: "
    f"{generation_error_count}/{total_results}"
)


# -------------------------------------------------------------------
# RESPONSE TYPES
# -------------------------------------------------------------------

print()
print("Response Types:")

if (
    len(results_df) > 0
    and "response_type" in results_df.columns
):

    response_counts = (
        results_df[
            "response_type"
        ]
        .fillna("unknown")
        .value_counts()
    )

    for response_type, count in (
        response_counts.items()
    ):

        print(
            f"  {response_type}: {count}"
        )


# -------------------------------------------------------------------
# ESCALATION
# -------------------------------------------------------------------

print()
print("Escalation:")

if (
    len(results_df) > 0
    and "escalation_decision"
    in results_df.columns
):

    escalation_counts = (
        results_df[
            "escalation_decision"
        ]
        .fillna("UNKNOWN")
        .value_counts()
    )

    for decision, count in (
        escalation_counts.items()
    ):

        print(
            f"  {decision}: {count}"
        )


# -------------------------------------------------------------------
# FAILURE SUMMARY
# -------------------------------------------------------------------

print()
print("Failure Summary:")

failure_summary = {}

if len(failures_df) > 0:

    failure_summary = (
        failures_df[
            "failure_type"
        ]
        .value_counts()
        .to_dict()
    )

    for failure_type, count in (
        failure_summary.items()
    ):

        print(
            f"  {failure_type}: {count}"
        )

else:

    print("  No generation failures.")


# -------------------------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------------------------

if y_true:

    print()
    print("=" * 90)
    print("CLASSIFICATION REPORT")
    print("=" * 90)

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    print(report)


# -------------------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------------------

if y_true:

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_path = (
        RESULTS_DIR
        / "reply_generator_confusion_matrix.csv"
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.to_csv(
        confusion_path
    )

    print()
    print(
        f"Saved confusion matrix: "
        f"{confusion_path}"
    )


# -------------------------------------------------------------------
# SUMMARY CSV
# -------------------------------------------------------------------

summary_rows = [
    {
        "metric": "total_examples",
        "value": total_results,
    },
    {
        "metric": "intent_accuracy",
        "value": intent_accuracy,
    },
    {
        "metric": "intent_macro_f1",
        "value": intent_macro_f1,
    },
    {
        "metric": "intent_weighted_f1",
        "value": intent_weighted_f1,
    },
    {
        "metric": "evidence_used_count",
        "value": evidence_used_count,
    },
    {
        "metric": "evidence_used_rate",
        "value": evidence_used_rate,
    },
    {
        "metric": "evidence_safe_count",
        "value": evidence_safe_count,
    },
    {
        "metric": "evidence_safe_rate",
        "value": evidence_safe_rate,
    },
    {
        "metric": "avg_evidence_relevance",
        "value": avg_evidence_relevance,
    },
    {
        "metric": "avg_evidence_quality",
        "value": avg_evidence_quality,
    },
    {
        "metric": "avg_problem_match",
        "value": avg_problem_match,
    },
    {
        "metric": "grounded_count",
        "value": grounded_count,
    },
    {
        "metric": "grounded_rate",
        "value": grounded_rate,
    },
    {
        "metric": "unsupported_claims_count",
        "value": unsupported_count,
    },
    {
        "metric": "unsupported_claims_rate",
        "value": unsupported_rate,
    },
    {
        "metric": "safe_to_answer_count",
        "value": safe_to_answer_count,
    },
    {
        "metric": "safe_to_answer_rate",
        "value": safe_to_answer_rate,
    },
    {
        "metric": "generation_error_count",
        "value": generation_error_count,
    },
    {
        "metric": "escalation_count",
        "value": len(escalation_cases),
    },
]


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)

print()
print(
    f"Saved summary: "
    f"{SUMMARY_PATH}"
)


# -------------------------------------------------------------------
# FINAL STATUS
# -------------------------------------------------------------------

print()
print("=" * 90)
print("EVALUATION COMPLETE")
print("=" * 90)

print()
print(
    f"Full results:       {FULL_RESULTS_PATH}"
)

print(
    f"Failure audit:      {FAILURE_PATH}"
)

print(
    f"Escalation cases:   {ESCALATION_PATH}"
)

print(
    f"Summary:            {SUMMARY_PATH}"
)

print()
print("Done.")