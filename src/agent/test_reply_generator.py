"""
src/agent/test_reply_generator.py

Automated test and evaluation script for the Hardened Grounded Reply Generator.
Tests:
1. Core 12 diagnostic cases (A through L)
2. Adversarial mismatch tests (Battery vs Wi-Fi, Screen vs Autocorrect, SIM vs Power-on, etc.)
3. Generates structured results, rejections log, and summary CSVs.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent.reply_generator import ReplyGenerator, is_evidence_safe

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPLY_RESULTS_FILE = RESULTS_DIR / "reply_generation_results.csv"
REPLY_SUMMARY_FILE = RESULTS_DIR / "reply_generation_summary.csv"
REPLY_REJECTIONS_FILE = RESULTS_DIR / "reply_evidence_rejections.csv"

# Core 12 Target Cases
TARGET_CASES = [
    ("A", "My iPhone battery is draining very quickly", "battery_charging"),
    ("B", "I cannot connect to WiFi", "wifi_connectivity"),
    ("C", "My iPhone won't make calls", "calls_cellular"),
    ("D", "I cannot download an app from the App Store", "app_store_downloads"),
    ("E", "My iCloud account is locked", "apple_id_icloud"),
    ("F", "My screen is completely black", "screen_display"),
    ("G", "My iPhone speaker is not working", "audio_speaker"),
    ("H", "My iPhone camera is not working", "device_hardware"),
    ("I", "My iPhone is stuck on the Apple logo", "device_hardware"),
    ("J", "The YouTube app keeps crashing on my iPhone", "app_problems"),
    ("K", "YouTube keeps crashing", "app_problems"),
    ("L", "My SIM card is not working", "calls_cellular")
]

# Adversarial Mismatch Cases (Query paired with tricky/mismatched candidate evidence)
ADVERSARIAL_CASES = [
    {
        "id": "ADV-1",
        "query": "My battery drains quickly",
        "intent": "battery_charging",
        "candidate": {
            "customer_text": "iPhone 7 battery draining very quickly",
            "historical_response": "@12345 In iOS 11 the Bluetooth and Wi-Fi controls in Control Center will disconnect. Turn them off in settings.",
            "problem_match_score": 0.95,
            "evidence_relevance_score": 0.80,
            "evidence_quality_score": 0.90,
            "entity_mismatch": False,
            "issue_mismatch": False,
            "actionable": True,
            "generic_response": False
        },
        "expected_rejection": "resolution_divergence"
    },
    {
        "id": "ADV-2",
        "query": "My screen is black",
        "intent": "screen_display",
        "candidate": {
            "customer_text": "My screen is black and display glitch",
            "historical_response": "@12345 We recently released iOS 11.1.1 which contains a fix for autocorrect and letter i glitch.",
            "problem_match_score": 0.70,
            "evidence_relevance_score": 0.65,
            "evidence_quality_score": 0.90,
            "entity_mismatch": False,
            "issue_mismatch": False,
            "actionable": True,
            "generic_response": False
        },
        "expected_rejection": "resolution_divergence"
    },
    {
        "id": "ADV-3",
        "query": "My SIM is not working",
        "intent": "calls_cellular",
        "candidate": {
            "customer_text": "My SIM card cannot connect",
            "historical_response": "@12345 If your iPhone won't power on try these steps to restore in recovery mode.",
            "problem_match_score": 0.70,
            "evidence_relevance_score": 0.65,
            "evidence_quality_score": 0.90,
            "entity_mismatch": False,
            "issue_mismatch": False,
            "actionable": True,
            "generic_response": False
        },
        "expected_rejection": "resolution_divergence"
    },
    {
        "id": "ADV-4",
        "query": "YouTube keeps crashing",
        "intent": "app_problems",
        "candidate": {
            "customer_text": "Music app keeps crashing",
            "historical_response": "@12345 Please delete and reinstall the Apple Music app to fix crashes.",
            "problem_match_score": 0.40,
            "evidence_relevance_score": 0.50,
            "evidence_quality_score": 0.85,
            "entity_mismatch": True,
            "issue_mismatch": False,
            "actionable": True,
            "generic_response": False
        },
        "expected_rejection": "application_mismatch"
    },
    {
        "id": "ADV-5",
        "query": "My iCloud account is locked",
        "intent": "apple_id_icloud",
        "candidate": {
            "customer_text": "Locked out of my iCloud account",
            "historical_response": "@12345 If your iPhone camera won't power on or screen is dark try these steps.",
            "problem_match_score": 0.30,
            "evidence_relevance_score": 0.50,
            "evidence_quality_score": 0.85,
            "entity_mismatch": False,
            "issue_mismatch": True,
            "actionable": True,
            "generic_response": False
        },
        "expected_rejection": "issue_mismatch"
    }
]

# 10 Required Evidence-Safety Regression Test Cases (Requirement 20)
REGRESSION_10_CASES = [
    {
        "id": "CASE 1",
        "name": "Battery query + Wi-Fi response",
        "query": "My iPhone battery is draining very quickly",
        "intent": "battery_charging",
        "customer": "My iPhone battery drains fast",
        "response": "Go to Settings > Wi-Fi and turn off Wi-Fi.",
        "expect_safe": False,
    },
    {
        "id": "CASE 2",
        "name": "Battery query + battery troubleshooting response",
        "query": "My iPhone battery is draining very quickly",
        "intent": "battery_charging",
        "customer": "battery draining fast",
        "response": "Check Battery Usage in Settings and identify apps using significant power.",
        "expect_safe": True,
    },
    {
        "id": "CASE 3",
        "name": "Wi-Fi query + Wi-Fi troubleshooting response",
        "query": "I cannot connect to WiFi",
        "intent": "wifi_connectivity",
        "customer": "wifi not connecting",
        "response": "Go to Settings > Wi-Fi, tap your network, tap Forget this Network, then reconnect.",
        "expect_safe": True,
    },
    {
        "id": "CASE 4",
        "name": "Wi-Fi query + generic DM us",
        "query": "I cannot connect to WiFi",
        "intent": "wifi_connectivity",
        "customer": "wifi problem",
        "response": "Please send us a DM so we can look into this for you.",
        "expect_safe": False,
    },
    {
        "id": "CASE 5",
        "name": "Apple ID query + camera troubleshooting response",
        "query": "My Apple ID password is not working",
        "intent": "apple_id_icloud",
        "customer": "forgot apple id password",
        "response": "Restart your iPhone and check the camera lens for dirt.",
        "expect_safe": False,
    },
    {
        "id": "CASE 6",
        "name": "Screen query + autocorrect/keyboard response",
        "query": "My screen is completely black",
        "intent": "screen_display",
        "customer": "screen black display",
        "response": "We released iOS 11.1.1 which contains a fix for autocorrect and keyboard typing.",
        "expect_safe": False,
    },
    {
        "id": "CASE 7",
        "name": "SIM query + SIM troubleshooting response",
        "query": "My SIM card says Invalid SIM",
        "intent": "calls_cellular",
        "customer": "no sim card installed error",
        "response": "Remove the SIM card with a paperclip, inspect it for damage, and reinsert it firmly.",
        "expect_safe": True,
    },
    {
        "id": "CASE 8",
        "name": "YouTube query + Apple Music response",
        "query": "The YouTube app keeps crashing",
        "intent": "app_problems",
        "customer": "app crashing",
        "response": "Delete and reinstall the Apple Music app from the App Store.",
        "expect_safe": False,
    },
    {
        "id": "CASE 9",
        "name": "iOS update query + iOS update troubleshooting response",
        "query": "My iPhone is unable to check for iOS update",
        "intent": "ios_software_update",
        "customer": "cannot update ios",
        "response": "Go to Settings > General > iPhone Storage, delete the update file, and download the iOS software update again.",
        "expect_safe": True,
    },
    {
        "id": "CASE 10",
        "name": "Generic customer message + generic AppleSupport response",
        "query": "Hey Apple need help",
        "intent": "other_unclear",
        "customer": "need help",
        "response": "Let us know what is going on and we will be happy to help!",
        "expect_safe": False,
    },
]


def run_tests():
    print("=" * 95)
    print("HARDENED GROUNDED REPLY GENERATOR TEST SUITE")
    print("=" * 95)

    generator = ReplyGenerator()
    results_records = []
    rejections_records = []

    print("\n--- SECTION 1: CORE TARGET CASES (A THROUGH L) ---")
    for case_id, query, expected_intent in TARGET_CASES:
        res = generator.generate(query)

        sel_ev = res.get("selected_evidence_text", "")
        ev_disp = sel_ev[:85].replace("\n", " ").encode("ascii", "replace").decode("ascii") if sel_ev else "None"
        reply_disp = res["reply"].replace("\n", " ").encode("ascii", "replace").decode("ascii")

        print(f"\n[Case {case_id}] QUERY          : {query}")
        print(f"         INTENT         : {res['predicted_intent']} (Expected: {expected_intent} | Conf: {res['intent_confidence']:.4f})")
        print(f"         EVIDENCE USED  : {res['evidence_used']} (ID: {res['evidence_id']})")
        print(f"         EVIDENCE SNIP  : {ev_disp}...")
        print(f"         EVID SCORES    : Relevance={res['evidence_relevance']:.4f} | Quality={res['evidence_quality']:.4f} | ProbMatch={res['problem_match_score']:.4f}")
        if not res["evidence_used"]:
            print(f"         REJECTION      : {res['evidence_rejection_reason']}")
            rejections_records.append({
                "test_type": "core_target",
                "query": query,
                "predicted_intent": res["predicted_intent"],
                "rejection_reason": res["evidence_rejection_reason"],
                "candidate_snippet": ev_disp
            })
        print(f"         RESPONSE TYPE  : {res['response_type']}")
        print(f"         GENERATED REPLY: {reply_disp}")
        print(f"         STATUS         : Grounded={res['grounded']} | EvSupported={res['evidence_supported']} | SafeToAnswer={res['safe_to_answer']}")
        print(f"         UNSUPPORTED    : {res['unsupported_claims']}")
        print("-" * 95)

        results_records.append(res)

    print("\n--- SECTION 2: ADVERSARIAL MISMATCH TESTS ---")
    for adv in ADVERSARIAL_CASES:
        cand_series = pd.Series(adv["candidate"])
        is_safe, rej_reason = is_evidence_safe(adv["query"], cand_series, adv["intent"])
        passed = (not is_safe) and (adv["expected_rejection"] in rej_reason)

        print(f"\n[{adv['id']}] Query: '{adv['query']}'")
        print(f"         Candidate Ev: {adv['candidate']['historical_response'][:75]}...")
        print(f"         Safe Gate Result: is_safe={is_safe} | Reason={rej_reason}")
        print(f"         Adversarial Filter Passed: {passed}")

        rejections_records.append({
            "test_type": "adversarial",
            "query": adv["query"],
            "predicted_intent": adv["intent"],
            "rejection_reason": rej_reason,
            "candidate_snippet": adv["candidate"]["historical_response"][:85]
        })

    print("\n--- SECTION 3: 10 REGRESSION EVIDENCE-SAFETY CASES (REQUIREMENT 20) ---")
    reg_passed_count = 0
    for reg in REGRESSION_10_CASES:
        is_safe, rej_reason = is_evidence_safe(
            query=reg["query"],
            evidence_customer=reg["customer"],
            evidence_response=reg["response"],
            query_intent=reg["intent"],
        )
        passed = (is_safe == reg["expect_safe"])
        if passed:
            reg_passed_count += 1
        status = "PASSED" if passed else "FAILED"
        print(f"\n[{reg['id']}] {reg['name']}")
        print(f"         Query: '{reg['query']}'")
        print(f"         Candidate Ev: {reg['response'][:75]}...")
        print(f"         Safe Gate Result: is_safe={is_safe} | Reason={rej_reason}")
        print(f"         Regression Test Status: {status}")

        rejections_records.append({
            "test_type": "regression_10",
            "query": reg["query"],
            "predicted_intent": reg["intent"],
            "rejection_reason": rej_reason,
            "candidate_snippet": reg["response"][:85]
        })
    print(f"\nRegression 10 Cases Score: {reg_passed_count}/{len(REGRESSION_10_CASES)} Passed")

    # Save CSV Artifacts
    results_df = pd.DataFrame(results_records)
    results_df.to_csv(REPLY_RESULTS_FILE, index=False)

    rejections_df = pd.DataFrame(rejections_records)
    rejections_df.to_csv(REPLY_REJECTIONS_FILE, index=False)

    # Calculate Summary Statistics
    total_queries = len(results_df)
    ev_used_count = int(results_df["evidence_used"].sum())
    ev_rejected_count = total_queries - ev_used_count
    safe_to_answer_count = int(results_df["safe_to_answer"].sum())
    grounded_count = int(results_df["grounded"].sum())
    ev_supported_count = int(results_df["evidence_supported"].sum())
    insufficient_count = int((results_df["response_type"] == "insufficient_evidence").sum())

    ev_used_rows = results_df[results_df["evidence_used"]]
    avg_rel = float(ev_used_rows["evidence_relevance"].mean()) if len(ev_used_rows) > 0 else 0.0
    avg_qual = float(ev_used_rows["evidence_quality"].mean()) if len(ev_used_rows) > 0 else 0.0
    avg_prob = float(ev_used_rows["problem_match_score"].mean()) if len(ev_used_rows) > 0 else 0.0

    type_dist = results_df["response_type"].value_counts().to_dict()

    summary_data = [{
        "total_queries": total_queries,
        "evidence_used": ev_used_count,
        "evidence_rejected": ev_rejected_count,
        "safe_to_answer": safe_to_answer_count,
        "grounded": grounded_count,
        "evidence_supported": ev_supported_count,
        "insufficient_evidence": insufficient_count,
        "average_evidence_relevance": round(avg_rel, 4),
        "average_evidence_quality": round(avg_qual, 4),
        "average_problem_match": round(avg_prob, 4),
        "response_type_distribution": str(type_dist)
    }]

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(REPLY_SUMMARY_FILE, index=False)

    print("\n" + "=" * 95)
    print("HARDENED REPLY GENERATOR SUMMARY")
    print("=" * 95)
    print(f"Total Queries Evaluated     : {total_queries}")
    print(f"Evidence Used               : {ev_used_count} ({ev_used_count / total_queries * 100:.1f}%)")
    print(f"Evidence Rejected           : {ev_rejected_count} ({ev_rejected_count / total_queries * 100:.1f}%)")
    print(f"Safe to Answer              : {safe_to_answer_count} ({safe_to_answer_count / total_queries * 100:.1f}%)")
    print(f"Grounded Replies            : {grounded_count} ({grounded_count / total_queries * 100:.1f}%)")
    print(f"Evidence Supported Replies  : {ev_supported_count} ({ev_supported_count / total_queries * 100:.1f}%)")
    print(f"Insufficient Evidence       : {insufficient_count}")
    print(f"Average Evidence Relevance  : {avg_rel:.4f}")
    print(f"Average Evidence Quality    : {avg_qual:.4f}")
    print(f"Average Problem Match Score : {avg_prob:.4f}")
    print(f"Response Type Distribution  : {type_dist}")
    print(f"\nSaved artifacts:")
    print(f"  - {REPLY_RESULTS_FILE}")
    print(f"  - {REPLY_SUMMARY_FILE}")
    print(f"  - {REPLY_REJECTIONS_FILE}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_tests()
