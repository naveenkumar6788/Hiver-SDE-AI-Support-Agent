"""
src/agent/escalation.py

Deterministic, Safety-First Escalation Decision Module for AppleSupport AI Agent.

Core Principle:
Safety over automated coverage. Prevent unsafe, inappropriate, or hallucinated
automated replies by routing high-risk or unsupported queries to official Apple Support.

Decisions:
- AUTO_HANDLE: Safe, verified routine troubleshooting/diagnostic assistance.
- ESCALATE: High-risk domains (security, billing, legal, privacy, account access)
            or technical unreliability (insufficient/unsafe evidence, ungrounded replies).
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configurable Intent Confidence Threshold
INTENT_CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# Transparent Domain Heuristic Checkers
# ============================================================

def is_security_case(text: str) -> Tuple[bool, str]:
    """Check for account takeover, hacking, unauthorized access, or compromise."""
    if not text:
        return False, ""
    lower = text.lower()
    patterns = [
        (r"\b(hack(ed|ing)?|someone hacked)\b", "account hacking detected"),
        (r"\b(someone (accessed|logged into|used) my account)\b", "unauthorized account access"),
        (r"\b(unauthorized (access|login|activity|use))\b", "unauthorized access attempt"),
        (r"\b(someone changed my password)\b", "unauthorized credential change"),
        (r"\b(suspicious activity|account (was |is )?compromised|compromised account)\b", "suspicious activity or compromise"),
        (r"\b(security breach)\b", "security breach report")
    ]
    for pat, desc in patterns:
        if re.search(pat, lower):
            return True, desc
    return False, ""


def is_payment_case(text: str) -> Tuple[bool, str]:
    """Check for billing disputes, unauthorized charges, refunds, or financial issues."""
    if not text:
        return False, ""
    lower = text.lower()
    patterns = [
        (r"\b(charged (twice|wrong|extra|again)|duplicate charge)\b", "duplicate or incorrect charge"),
        (r"\b(unauthorized (charge|transaction|payment|fee))\b", "unauthorized transaction"),
        (r"\b(refund|money back|credit my account)\b", "refund request"),
        (r"\b(money (taken|deducted) from my (account|card|bank))\b", "unauthorized fund deduction"),
        (r"\b(someone used my (card|credit card|debit card))\b", "payment card compromise"),
        (r"\b(credit card fraud|bank dispute|billing dispute|billing problem|billing issue)\b", "financial or billing dispute"),
        (r"\b(subscription billing|charged without my permission)\b", "subscription billing issue")
    ]
    for pat, desc in patterns:
        if re.search(pat, lower):
            return True, desc
    return False, ""


def is_legal_case(text: str) -> Tuple[bool, str]:
    """Check for legal threats, lawsuits, regulatory complaints, or lawyer involvement."""
    if not text:
        return False, ""
    lower = text.lower()
    patterns = [
        (r"\b(legal action|take legal action|legal complaint)\b", "legal action notice"),
        (r"\b(lawsuit|sue apple|sue you|suing)\b", "litigation / lawsuit threat"),
        (r"\b(attorney|lawyer|legal counsel|legal team)\b", "legal counsel involvement"),
        (r"\b(court|regulatory complaint|ftc complaint|consumer protection bureau)\b", "formal regulatory or court complaint")
    ]
    for pat, desc in patterns:
        if re.search(pat, lower):
            return True, desc
    return False, ""


def is_privacy_case(text: str) -> Tuple[bool, str]:
    """Check for personal data exposure, GDPR/privacy requests, or data deletion demands."""
    if not text:
        return False, ""
    lower = text.lower()
    patterns = [
        (r"\b(personal data (deleted|exported|leaked|exposed)|delete my (personal )?data)\b", "personal data deletion or export request"),
        (r"\b(someone has my (private|personal) (information|data|photos))\b", "private data exposure"),
        (r"\b(privacy (complaint|violation|breach)|gdpr)\b", "privacy rights or violation complaint"),
        (r"\b(data leak|data exposure)\b", "data breach complaint")
    ]
    for pat, desc in patterns:
        if re.search(pat, lower):
            return True, desc
    return False, ""


def is_sensitive_transaction(text: str) -> Tuple[bool, str]:
    """Check for stolen hardware with account lock, activation lock bypass, or safety hazard."""
    if not text:
        return False, ""
    lower = text.lower()
    patterns = [
        (r"\b(stolen (device|phone|iphone|ipad|mac)|lost (my )?phone.*stolen)\b", "reported stolen device"),
        (r"\b(activation lock bypass|bypass (icloud|activation lock))\b", "device ownership / activation lock"),
        (r"\b(swollen\s+battery|(battery|phone|iphone)\s+(is\s+|was\s+)?(swollen|swelling|bulging)|(battery|phone|iphone)\s+(is\s+|was\s+)?(getting\s+hot|overheating)|battery\s+(is\s+|was\s+)?(leaking|punctured)|smoke\s+from\s+(the\s+)?(battery|phone|iphone)|(battery|phone|iphone)\s+(is\s+|was\s+)?smoking|smoking\s+iphone|smoke|smoking|spark(ing)?|burn(ing|t)?|battery\s+(exploded|explosion)|fire\s+hazard|exploded)\b", "hardware safety hazard"),
        (r"\b(identity verification for recovery|security-sensitive changes)\b", "sensitive security recovery")
    ]
    for pat, desc in patterns:
        if re.search(pat, lower):
            return True, desc
    return False, ""


def is_account_access_case(text: str) -> Tuple[bool, str]:
    """
    Check for account recovery, account lockouts, or authentication lockouts.
    CRITICAL CONTEXT RULE:
    - 'My iCloud account is locked' -> TRUE (account access lockout)
    - 'My phone is locked on the Apple logo' -> FALSE (device hardware/boot issue)
    """
    if not text:
        return False, ""
    lower = text.lower()

    # Rule out device boot / display lock (e.g., 'phone is locked on the Apple logo')
    device_boot_patterns = [
        r"\b(phone|iphone|ipad|device|screen)\s+(is\s+)?(locked|stuck|frozen)\s+(on|at)\s+(the\s+)?(apple\s*logo|logo|boot|spinning wheel|itunes)\b",
        r"\b(locked|stuck|frozen)\s+(on|at)\s+(the\s+)?apple\s*logo\b",
        r"\b(lock|locked)\s+screen\b"
    ]
    if any(re.search(p, lower) for p in device_boot_patterns):
        # Unless the text also explicitly mentions iCloud or Apple ID account recovery
        if not re.search(r"\b(icloud|apple\s*id|appleid|my account)\b", lower):
            return False, ""

    patterns = [
        (r"\b(icloud|apple\s*id|appleid|my account|account)\s+(is\s+)?(locked|disabled)\b", "locked or disabled account"),
        (r"\b(locked out of (my )?(account|icloud|apple\s*id|appleid))\b", "account lockout"),
        (r"\b(cannot|unable to|can't) (access|log in to|sign in to|login to|signin to) (my )?(account|icloud|apple\s*id)\b", "unable to access account"),
        (r"\b(account recovery|cannot verify identity|identity verification|two-factor.*locked out)\b", "account recovery authentication failure"),
        (r"\b(forgot (my )?apple\s*id password.*locked)\b", "password lockout")
    ]
    for pat, desc in patterns:
        if re.search(pat, lower):
            return True, desc
    return False, ""


def is_unclear_case(text: str) -> Tuple[bool, str]:
    """
    Check for vague, context-free user messages such as 'What is this?', 'Help', 'Fix this'.
    """
    if not text:
        return True, "empty customer message"
    lower = text.lower()

    # Remove user handles and links
    cleaned = re.sub(r"@\w+", " ", lower)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    tokens = cleaned.split()

    # Only flag as unclear if message is very short (<= 5 tokens) and matches vague intent
    if len(tokens) <= 5:
        vague_phrases = [
            r"^(what is this|what'?s this|help|please help|fix this|this is broken|what is going on|what'?s going on|broken|not working)$",
            r"^(help me|can you help|i need help)$"
        ]
        text_joined = " ".join(tokens)
        for pat in vague_phrases:
            if re.match(pat, text_joined):
                return True, f"vague customer message: '{text_joined}'"

    return False, ""


# ============================================================
# Main Deterministic Escalation Decision Function
# ============================================================

def decide_escalation(
    customer_message: str,
    predicted_intent: str = "other_unclear",
    intent_confidence: float = 1.0,
    evidence_used: bool = True,
    evidence_supported: bool = True,
    safe_to_answer: bool = True,
    response_type: str = "troubleshooting",
    evidence_relevance: float = 0.80,
    evidence_quality: float = 0.80,
    problem_match_score: float = 0.80,
    unsupported_claims: Any = None,
    grounded: bool = True,
    evidence_rejection_reason: str = ""
) -> Dict[str, Any]:
    """
    Decide whether to AUTO_HANDLE or ESCALATE a customer query.

    Deterministic Priority Order:
    1. account_security
    2. payment_billing
    3. legal_or_policy
    4. privacy_personal_data
    5. sensitive_transaction
    6. account_access
    7. unsupported_response
    8. unsafe_evidence
    9. insufficient_evidence
    10. low_intent_confidence
    11. unclear_issue
    12. routine_supported_issue -> AUTO_HANDLE

    Returns:
        {
            "decision": "AUTO_HANDLE" | "ESCALATE",
            "reason_code": str,
            "reason": str,
            "confidence": float,
            "risk_level": "low" | "medium" | "high"
        }
    """
    text = customer_message or ""

    # 1. High-Risk: Account Security
    is_sec, sec_reason = is_security_case(text)
    if is_sec:
        return {
            "decision": "ESCALATE",
            "reason_code": "account_security",
            "reason": f"Account security concern: {sec_reason}",
            "confidence": 1.0,
            "risk_level": "high"
        }

    # 2. High-Risk: Payment / Billing
    is_pay, pay_reason = is_payment_case(text)
    if is_pay:
        return {
            "decision": "ESCALATE",
            "reason_code": "payment_billing",
            "reason": f"Financial / billing matter: {pay_reason}",
            "confidence": 1.0,
            "risk_level": "high"
        }

    # 3. High-Risk: Legal or Policy
    is_leg, leg_reason = is_legal_case(text)
    if is_leg:
        return {
            "decision": "ESCALATE",
            "reason_code": "legal_or_policy",
            "reason": f"Legal or regulatory inquiry: {leg_reason}",
            "confidence": 1.0,
            "risk_level": "high"
        }

    # 4. High-Risk: Privacy / Personal Data
    is_priv, priv_reason = is_privacy_case(text)
    if is_priv:
        return {
            "decision": "ESCALATE",
            "reason_code": "privacy_personal_data",
            "reason": f"Privacy or personal data request: {priv_reason}",
            "confidence": 1.0,
            "risk_level": "high"
        }

    # 5. High-Risk: Sensitive Transaction / Safety Hazard
    is_trans, trans_reason = is_sensitive_transaction(text)
    if is_trans:
        return {
            "decision": "ESCALATE",
            "reason_code": "sensitive_transaction",
            "reason": f"Sensitive transaction / hardware safety hazard: {trans_reason}",
            "confidence": 1.0,
            "risk_level": "high"
        }

    # 6. High/Medium-Risk: Account Access
    is_acc, acc_reason = is_account_access_case(text)
    if is_acc:
        return {
            "decision": "ESCALATE",
            "reason_code": "account_access",
            "reason": f"Account access lockout: {acc_reason}",
            "confidence": 0.95,
            "risk_level": "high"
        }

    # 7. Quality: Unsupported Response (Ungrounded or containing unsupported claims)
    has_unsupported_claims = False
    if isinstance(unsupported_claims, list):
        has_unsupported_claims = len(unsupported_claims) > 0
    elif isinstance(unsupported_claims, str):
        s = unsupported_claims.strip()
        has_unsupported_claims = (s != "" and s != "[]" and s != "None")

    if not grounded or has_unsupported_claims:
        return {
            "decision": "ESCALATE",
            "reason_code": "unsupported_response",
            "reason": f"Generated reply contains ungrounded or unsupported claims: {unsupported_claims}",
            "confidence": 1.0,
            "risk_level": "high"
        }

    # 8. Evidence Quality: Explicitly Unsafe Evidence
    rej_str = str(evidence_rejection_reason).lower()
    is_explicitly_unsafe = (
        ("divergence" in rej_str)
        or ("mismatch" in rej_str)
        or ("generic" in rej_str)
        or (problem_match_score < 0.60 and evidence_used)
        or (evidence_relevance < 0.60 and evidence_used)
    )
    if is_explicitly_unsafe:
        return {
            "decision": "ESCALATE",
            "reason_code": "unsafe_evidence",
            "reason": f"Retrieved evidence rejected by safety gate: {evidence_rejection_reason or 'threshold failure'}",
            "confidence": 0.90,
            "risk_level": "medium"
        }

    # 9. Conservative Safety: Insufficient Evidence
    if not evidence_used or not evidence_supported or not safe_to_answer or response_type == "insufficient_evidence":
        return {
            "decision": "ESCALATE",
            "reason_code": "insufficient_evidence",
            "reason": f"Conservative safety fallback: insufficient evidence ({evidence_rejection_reason or 'evidence not usable'})",
            "confidence": 0.90,
            "risk_level": "medium"
        }

    # 10. Reliability: Low Intent Confidence
    if intent_confidence < INTENT_CONFIDENCE_THRESHOLD:
        return {
            "decision": "ESCALATE",
            "reason_code": "low_intent_confidence",
            "reason": f"Intent confidence ({intent_confidence:.2f}) below threshold ({INTENT_CONFIDENCE_THRESHOLD:.2f})",
            "confidence": round(1.0 - intent_confidence, 4),
            "risk_level": "medium"
        }

    # 11. Ambiguity: Unclear Issue
    is_unc, unc_reason = is_unclear_case(text)
    if is_unc:
        return {
            "decision": "ESCALATE",
            "reason_code": "unclear_issue",
            "reason": f"Customer request is ambiguous without substantive problem description: {unc_reason}",
            "confidence": 0.85,
            "risk_level": "medium"
        }

    # 12. Routine Supported Issue: Safe to Auto-Handle
    return {
        "decision": "AUTO_HANDLE",
        "reason_code": "routine_supported_issue",
        "reason": f"Routine technical issue with verified safe evidence support ({predicted_intent})",
        "confidence": round(intent_confidence, 4),
        "risk_level": "low"
    }


# ============================================================
# Self-Test CLI
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ESCALATION POLICY MODULE SELF-TEST")
    print("=" * 70)

    sample_queries = [
        ("My Apple account was hacked", "account_security", True),
        ("I was charged twice", "payment_billing", True),
        ("My iCloud account is locked", "account_access", True),
        ("My phone is locked on the Apple logo", "routine_supported_issue", False),
        ("My iPhone battery drains quickly", "routine_supported_issue", False),
        ("I want a refund", "payment_billing", True),
        ("I will take legal action", "legal_or_policy", True),
    ]

    for q, expected_reason, expected_escalate in sample_queries:
        res = decide_escalation(
            customer_message=q,
            predicted_intent="battery_charging" if "battery" in q else "device_hardware",
            intent_confidence=0.95,
            evidence_used=True,
            evidence_supported=True,
            safe_to_answer=True,
            response_type="troubleshooting"
        )
        is_esc = (res["decision"] == "ESCALATE")
        match = (is_esc == expected_escalate) and (res["reason_code"] == expected_reason)
        status = "PASS" if match else "FAIL"
        print(f"[{status}] '{q}' -> {res['decision']} ({res['reason_code']}) [Risk: {res['risk_level']}]")

    print("=" * 70)
