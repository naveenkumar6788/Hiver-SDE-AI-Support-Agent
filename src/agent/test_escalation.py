"""
src/agent/test_escalation.py

Comprehensive Test Suite for the Deterministic Escalation Decision Module.
Tests:
- High-risk security, payment, account access, privacy, legal, and safety cases
- Routine supported technical queries (AUTO_HANDLE)
- Context distinction (locked on Apple logo vs iCloud account locked)
- Low intent confidence, unsafe evidence, and insufficient evidence fallbacks
- Safety invariant assertions (verifying no high-risk case is ever auto-handled)
- Outputs results and summary CSVs.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
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

from src.agent.escalation import decide_escalation

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_RESULTS_FILE = RESULTS_DIR / "escalation_test_results.csv"
TEST_SUMMARY_FILE = RESULTS_DIR / "escalation_summary.csv"


TEST_CASES = [
    # ----------------------------------------------------
    # High-Risk Cases (1 to 10)
    # ----------------------------------------------------
    {
        "id": 1,
        "query": "My Apple account was hacked",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "account_security",
        "intent": "apple_id_icloud",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 2,
        "query": "Someone accessed my account",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "account_security",
        "intent": "apple_id_icloud",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 3,
        "query": "I was charged twice",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "payment_billing",
        "intent": "other_unclear",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 4,
        "query": "I want a refund",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "payment_billing",
        "intent": "other_unclear",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 5,
        "query": "Someone used my card",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "payment_billing",
        "intent": "other_unclear",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 6,
        "query": "My iCloud account is locked",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "account_access",
        "intent": "apple_id_icloud",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 7,
        "query": "I cannot access my account",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "account_access",
        "intent": "apple_id_icloud",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 8,
        "query": "I want my personal data deleted",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "privacy_personal_data",
        "intent": "other_unclear",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 9,
        "query": "I will take legal action",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "legal_or_policy",
        "intent": "other_unclear",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 10,
        "query": "My account was compromised",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "account_security",
        "intent": "apple_id_icloud",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },

    # ----------------------------------------------------
    # Routine Supported Cases (11 to 18)
    # ----------------------------------------------------
    {
        "id": 11,
        "query": "My iPhone battery drains quickly",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "battery_charging",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 12,
        "query": "I cannot connect to WiFi",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "wifi_connectivity",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 13,
        "query": "My iPhone speaker is not working",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "audio_speaker",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 14,
        "query": "My screen is black",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "screen_display",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 15,
        "query": "I cannot download an app from the App Store",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "app_store_downloads",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 16,
        "query": "My iPhone camera is not working",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "device_hardware",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 17,
        "query": "My iPhone is stuck on the Apple logo",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "device_hardware",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 18,
        "query": "My SIM card is not working",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "calls_cellular",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },

    # ----------------------------------------------------
    # Context Tests (19 to 20)
    # ----------------------------------------------------
    {
        "id": 19,
        "query": "My phone is locked on the Apple logo",
        "expected_decision": "AUTO_HANDLE",
        "expected_reason_code": "routine_supported_issue",
        "intent": "device_hardware",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 20,
        "query": "My account was hacked and I cannot log in",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "account_security",
        "intent": "apple_id_icloud",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },

    # ----------------------------------------------------
    # Technical Reliability & Safety Fallback Tests (21 to 27)
    # ----------------------------------------------------
    {
        "id": 21,
        "query": "My WiFi connection drops frequently",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "low_intent_confidence",
        "intent": "wifi_connectivity",
        "intent_confidence": 0.40,  # Below 0.60 threshold
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 22,
        "query": "My battery is draining fast",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "insufficient_evidence",
        "intent": "battery_charging",
        "intent_confidence": 0.95,
        "evidence_used": False,  # Missing / rejected evidence
        "evidence_supported": False,
        "safe_to_answer": False,
        "response_type": "insufficient_evidence"
    },
    {
        "id": 23,
        "query": "My battery is draining fast",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "unsafe_evidence",
        "intent": "battery_charging",
        "intent_confidence": 0.95,
        "evidence_used": False,
        "evidence_supported": False,
        "safe_to_answer": False,
        "response_type": "insufficient_evidence",
        "evidence_rejection_reason": "resolution_divergence: battery query with bluetooth/wifi resolution"
    },
    {
        "id": 24,
        "query": "What is this?",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "unclear_issue",
        "intent": "other_unclear",
        "intent_confidence": 0.85,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 25,
        "query": "Speaker is broken",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "insufficient_evidence",
        "intent": "audio_speaker",
        "intent_confidence": 0.90,
        "evidence_used": False,
        "evidence_supported": False,
        "safe_to_answer": False,
        "response_type": "insufficient_evidence"
    },
    {
        "id": 26,
        "query": "Can you fix my Wi-Fi?",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "unsupported_response",
        "intent": "wifi_connectivity",
        "intent_confidence": 0.90,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": False,
        "response_type": "troubleshooting",
        "grounded": False,
        "unsupported_claims": ["unsupported refund promise"]
    },
    {
        "id": 27,
        "query": "My phone battery is swollen and smoking",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "sensitive_transaction",
        "intent": "battery_charging",
        "intent_confidence": 0.95,
        "evidence_used": True,
        "evidence_supported": True,
        "safe_to_answer": True,
        "response_type": "troubleshooting"
    },
    {
        "id": 28,
        "query": "My battery is swollen and getting hot",
        "expected_decision": "ESCALATE",
        "expected_reason_code": "sensitive_transaction",
        "intent": "battery_charging",
        "intent_confidence": 0.95,
        "evidence_used": False,
        "evidence_supported": False,
        "safe_to_answer": False,
        "response_type": "escalation"
    }
]


def run_unit_tests():
    print("=" * 80)
    print("ESCALATION POLICY UNIT TEST SUITE (28 Tests)")
    print("=" * 80)

    records = []
    passed_count = 0
    failed_count = 0

    for test in TEST_CASES:
        t_id = test["id"]
        q = test["query"]
        res = decide_escalation(
            customer_message=q,
            predicted_intent=test.get("intent", "other_unclear"),
            intent_confidence=test.get("intent_confidence", 1.0),
            evidence_used=test.get("evidence_used", True),
            evidence_supported=test.get("evidence_supported", True),
            safe_to_answer=test.get("safe_to_answer", True),
            response_type=test.get("response_type", "troubleshooting"),
            unsupported_claims=test.get("unsupported_claims", []),
            grounded=test.get("grounded", True),
            evidence_rejection_reason=test.get("evidence_rejection_reason", "")
        )

        dec_match = (res["decision"] == test["expected_decision"])
        code_match = (res["reason_code"] == test["expected_reason_code"])
        test_passed = dec_match and code_match

        if test_passed:
            passed_count += 1
            status = "PASS"
        else:
            failed_count += 1
            status = "FAIL"

        print(f"[{status}] Test #{t_id:02d}: '{q}'")
        print(f"       Got: {res['decision']} ({res['reason_code']}) | Expected: {test['expected_decision']} ({test['expected_reason_code']}) [Risk: {res['risk_level']}]")

        # Invariant check: never AUTO_HANDLE on high-risk domains or unsafe states
        high_risk_codes = ["account_security", "payment_billing", "legal_or_policy", "privacy_personal_data", "sensitive_transaction", "account_access"]
        if res["reason_code"] in high_risk_codes:
            assert res["decision"] == "ESCALATE", f"CRITICAL SAFETY VIOLATION: {res['reason_code']} was auto-handled!"

        if not test.get("evidence_used", True) or not test.get("safe_to_answer", True):
            assert res["decision"] == "ESCALATE", f"CRITICAL SAFETY VIOLATION: Unsafe/insufficient evidence query was auto-handled!"

        records.append({
            "test_id": t_id,
            "query": q,
            "expected_decision": test["expected_decision"],
            "expected_reason_code": test["expected_reason_code"],
            "actual_decision": res["decision"],
            "actual_reason_code": res["reason_code"],
            "risk_level": res["risk_level"],
            "confidence": res["confidence"],
            "reason": res["reason"],
            "passed": test_passed
        })

    # Save Results CSV
    results_df = pd.DataFrame(records)
    results_df.to_csv(TEST_RESULTS_FILE, index=False)
    print(f"\nSaved test results to: {TEST_RESULTS_FILE}")

    # Summary
    total_tests = len(TEST_CASES)
    summary_data = [{
        "total_tests": total_tests,
        "passed_tests": passed_count,
        "failed_tests": failed_count,
        "pass_rate": round(passed_count / total_tests, 4),
        "safety_invariants_verified": True
    }]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(TEST_SUMMARY_FILE, index=False)
    print(f"Saved test summary to: {TEST_SUMMARY_FILE}")

    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Tests Evaluated      : {total_tests}")
    print(f"Tests Passed               : {passed_count} ({passed_count / total_tests * 100:.1f}%)")
    print(f"Tests Failed               : {failed_count} ({failed_count / total_tests * 100:.1f}%)")
    print(f"Safety Invariants Verified : 100% Passed (Zero high-risk auto-handles)")
    print("=" * 80 + "\n")

    if failed_count > 0:
        raise AssertionError(f"{failed_count} escalation tests failed!")


if __name__ == "__main__":
    run_unit_tests()
