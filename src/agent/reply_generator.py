import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.evidence_relevance import (
    extract_problem_terms,
    detect_problem_conflict,
    is_specific_evidence_safe,
    calculate_final_evidence_score,
)


# ============================================================
# DOMAIN PATTERNS
# ============================================================

DOMAIN_PATTERNS = {
    "battery": [
        r"\bbattery\b",
        r"\bdrain(?:ing|ed)?\b",
        r"\bcharge(?:r|ing|s)?\b",
        r"\bcharging\b",
        r"\bwon't charge\b",
        r"\bnot charging\b",
        r"\bpower drain\b",
        r"\bbattery life\b",
        r"\blow power mode\b",
    ],

    "wifi": [
        r"\bwifi\b",
        r"\bwi-fi\b",
        r"\bwireless network\b",
        r"\brouter\b",
        r"\bhotspot\b",
        r"\bwlan\b",
    ],

    "cellular_calls": [
        r"\bcall(?:s|ing)?\b",
        r"\bphone call\b",
        r"\bcan't call\b",
        r"\bcannot call\b",
        r"\bno service\b",
        r"\bsignal\b",
        r"\bcellular\b",
        r"\bmobile data\b",
        r"\bnetwork coverage\b",
        r"\bcarrier\b",
    ],

    "carrier_unlock": [
        r"\bcarrier unlock\b",
        r"\bunlock.*carrier\b",
        r"\bunlock.*iphone\b",
        r"\bcarrier\b.*\bsim\b",
        r"\banother carrier\b",
        r"\bdifferent carrier\b",
    ],

    "sim_hardware": [
        r"\bsim\b",
        r"\bsim card\b",
        r"\bsim tray\b",
        r"\bsim drawer\b",
        r"\bsim slot\b",
    ],

    "screen": [
        r"\bscreen\b",
        r"\bdisplay\b",
        r"\bblack screen\b",
        r"\bblank screen\b",
        r"\btouchscreen\b",
        r"\btouch screen\b",
        r"\bdisplay not\b",
    ],

    "speaker_audio": [
        r"\bspeaker\b",
        r"\bsound\b",
        r"\baudio\b",
        r"\bvolume\b",
        r"\bmicrophone\b",
        r"\bmic\b",
        r"\bno sound\b",
        r"\bsound not\b",
    ],

    "camera": [
        r"\bcamera\b",
        r"\bcamera app\b",
        r"\bpictures?\b",
        r"\bphotos?\b",
        r"\btake a photo\b",
    ],

    "app_store": [
        r"\bapp store\b",
        r"\bapple store\b",
        r"\bdownload.*app\b",
        r"\binstall.*app\b",
        r"\bcan't download\b",
        r"\bcannot download\b",
        r"\bcan't install\b",
        r"\bcannot install\b",
    ],

    "youtube": [
        r"\byoutube\b",
        r"\byoutube app\b",
    ],

    "apple_music": [
        r"\bapple music\b",
        r"\bitunes\b",
        r"\bmusic app\b",
        r"\bmusic\b",
    ],

    "apple_id": [
        r"\bapple id\b",
        r"\bicloud\b",
        r"\bi-cloud\b",
        r"\bforgot.*password\b",
        r"\bpassword.*apple\b",
        r"\baccount.*locked\b",
        r"\baccount.*lock\b",
        r"\bactivation lock\b",
    ],

    "account_security": [
        r"\bfake account\b",
        r"\bfake email\b",
        r"\bfake e-mail\b",
        r"\bsuspicious email\b",
        r"\bsuspicious e-mail\b",
        r"\bphishing\b",
        r"\bscam\b",
        r"\bscammer\b",
        r"\bfraud\b",
        r"\bfake apple\b",
        r"\bpretending to be apple\b",
        r"\bpretend.*apple\b",
        r"\bsecurity alert\b",
        r"\bsuspicious message\b",
        r"\bsuspicious text\b",
    ],

    "ios_update": [
        r"\bios\b.*\bupdate\b",
        r"\bupdate\b.*\bios\b",
        r"\bsoftware update\b",
        r"\bsoftware upgrade\b",
        r"\bupdate.*iphone\b",
        r"\bupdate.*ipad\b",
        r"\binstall.*update\b",
        r"\bprevious ios\b",
        r"\brevert.*ios\b",
    ],

    "apple_logo_boot": [
        r"\bapple logo\b",
        r"\bstuck.*logo\b",
        r"\bstuck on.*apple\b",
        r"\bboot loop\b",
        r"\bkeeps restarting\b",
        r"\bkeeps rebooting\b",
        r"\bstuck.*boot\b",
    ],

    "power_on": [
        r"\bwon't turn on\b",
        r"\bwill not turn on\b",
        r"\bdoesn't turn on\b",
        r"\bdoes not turn on\b",
        r"\bnot turning on\b",
        r"\bpower on\b",
        r"\bturn on\b",
    ],

    "app_problem": [
        r"\bapp\b",
        r"\bcrash(?:es|ed|ing)?\b",
        r"\bfreez(?:e|es|ing)\b",
        r"\bnot opening\b",
        r"\bwon't open\b",
        r"\bdoesn't open\b",
        r"\bapp problem\b",
        r"\bapplication\b",
    ],
}


# ============================================================
# RESPONSE DOMAIN PATTERNS
# ============================================================

RESPONSE_DOMAIN_PATTERNS = {
    "battery": [
        r"\bbattery\b",
        r"\bcharging\b",
        r"\bcharge\b",
        r"\bpower drain\b",
        r"\bbattery usage\b",
        r"\blow power mode\b",
    ],

    "wifi": [
        r"\bwifi\b",
        r"\bwi-fi\b",
        r"\bwireless network\b",
        r"\brouter\b",
        r"\bhotspot\b",
    ],

    "cellular_calls": [
        r"\bcellular\b",
        r"\bmobile data\b",
        r"\bsignal\b",
        r"\bno service\b",
        r"\bcarrier\b",
        r"\bcalls?\b",
    ],

    "sim_hardware": [
        r"\bsim\b",
        r"\bsim card\b",
        r"\bsim tray\b",
        r"\bsim drawer\b",
        r"\bsim slot\b",
    ],

    "carrier_unlock": [
        r"\bcarrier unlock\b",
        r"\bunlock.*carrier\b",
        r"\bcarrier\b.*\bunlock\b",
    ],

    "screen": [
        r"\bscreen\b",
        r"\bdisplay\b",
        r"\btouchscreen\b",
        r"\btouch screen\b",
    ],

    "speaker_audio": [
        r"\bspeaker\b",
        r"\bsound\b",
        r"\baudio\b",
        r"\bvolume\b",
        r"\bmicrophone\b",
        r"\bmic\b",
    ],

    "camera": [
        r"\bcamera\b",
        r"\bphotos?\b",
        r"\bpictures?\b",
    ],

    "app_store": [
        r"\bapp store\b",
        r"\bdownload\b",
        r"\binstall\b",
    ],

    "apple_music": [
        r"\bapple music\b",
        r"\bitunes\b",
        r"\bmusic\b",
        r"\bsongs?\b",
    ],

    "apple_id": [
        r"\bapple id\b",
        r"\bicloud\b",
        r"\bpassword\b",
        r"\baccount\b",
        r"\bactivation lock\b",
    ],

    "ios_update": [
        r"\bios\b",
        r"\bsoftware update\b",
        r"\bupdate\b",
    ],

    "apple_logo_boot": [
        r"\bapple logo\b",
        r"\bboot\b",
        r"\brestart\b",
        r"\breboot\b",
    ],

    "power_on": [
        r"\bturn on\b",
        r"\bpower on\b",
        r"\bpower\b",
    ],

    "youtube": [
        r"\byoutube\b",
    ],

    "app_problem": [
        r"\bapp\b",
        r"\bcrash(?:es|ed|ing)?\b",
        r"\bfreez(?:e|es|ing)\b",
    ],
}


# ============================================================
# INTENT -> DOMAIN MAPPING
# ============================================================

INTENT_DOMAINS = {
    "battery_charging": {
        "battery"
    },

    "ios_software_update": {
        "ios_update"
    },

    "wifi_connectivity": {
        "wifi"
    },

    "app_problems": {
        "app_problem",
        "youtube",
        "apple_music"
    },

    "app_store_downloads": {
        "app_store",
        "youtube"
    },

    "apple_music_itunes": {
        "apple_music"
    },

    "apple_id_icloud": {
        "apple_id"
    },

    "screen_display": {
        "screen"
    },

    "calls_cellular": {
        "cellular_calls",
        "carrier_unlock",
        "sim_hardware"
    },

    "audio_speaker": {
        "speaker_audio"
    },

    "device_hardware": {
        "camera",
        "apple_logo_boot",
        "power_on",
        "screen",
        "speaker_audio"
    },

    "other_unclear": set(),
}


# ============================================================
# CONFLICTING DOMAINS
# ============================================================

CONFLICTING_DOMAINS = {
    ("battery", "wifi"),
    ("battery", "screen"),
    ("battery", "camera"),
    ("battery", "youtube"),
    ("battery", "apple_music"),
    ("battery", "speaker_audio"),
    ("battery", "apple_id"),
    ("battery", "carrier_unlock"),
    ("battery", "sim_hardware"),

    ("wifi", "battery"),
    ("wifi", "speaker_audio"),
    ("wifi", "camera"),
    ("wifi", "screen"),
    ("wifi", "apple_music"),
    ("wifi", "apple_id"),

    ("screen", "speaker_audio"),
    ("screen", "apple_music"),
    ("screen", "battery"),
    ("screen", "wifi"),
    ("screen", "camera"),

    ("youtube", "apple_music"),
    ("youtube", "camera"),
    ("youtube", "battery"),
    ("youtube", "wifi"),
    ("youtube", "screen"),
    ("youtube", "speaker_audio"),

    ("apple_music", "youtube"),
    ("apple_music", "screen"),
    ("apple_music", "battery"),
    ("apple_music", "wifi"),

    ("sim_hardware", "battery"),
    ("sim_hardware", "screen"),
    ("sim_hardware", "camera"),
    ("sim_hardware", "speaker_audio"),

    ("carrier_unlock", "battery"),
    ("carrier_unlock", "screen"),
    ("carrier_unlock", "camera"),

    ("apple_id", "camera"),
    ("apple_id", "battery"),
    ("apple_id", "screen"),
    ("apple_id", "speaker_audio"),
}


# ============================================================
# COMPATIBLE HIERARCHIES
# ============================================================

COMPATIBLE_HIERARCHIES = {
    ("youtube", "app_problem"),
    ("app_problem", "youtube"),

    ("youtube", "app_store"),
    ("app_store", "youtube"),

    ("apple_music", "app_problem"),
    ("app_problem", "apple_music"),

    ("carrier_unlock", "cellular_calls"),
    ("cellular_calls", "carrier_unlock"),

    ("sim_hardware", "cellular_calls"),
    ("cellular_calls", "sim_hardware"),

    ("apple_logo_boot", "power_on"),
    ("power_on", "apple_logo_boot"),

    ("app_problem", "app_store"),
    ("app_store", "app_problem"),

    ("ios_update", "apple_logo_boot"),
}


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s'\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def clean_historical_response(text: str) -> str:
    """Clean Twitter artifacts from historical AppleSupport responses."""

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# DOMAIN DETECTION
# ============================================================

def detect_domains(text: str) -> List[str]:
    """Detect domains in a customer message or query."""

    if not text:
        return []

    normalized = normalize_text(text)

    found = []

    for domain, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                found.append(domain)
                break

    # Important precedence rules.
    #
    # Specific application/security/account cases should win
    # over generic "app", "music", etc.

    if "account_security" in found:
        return ["account_security"]

    if "carrier_unlock" in found:
        return ["carrier_unlock"]

    if "sim_hardware" in found:
        return ["sim_hardware"]

    if "apple_logo_boot" in found:
        return ["apple_logo_boot"]

    if "youtube" in found:
        return ["youtube"]

    if "apple_music" in found:
        return ["apple_music"]

    if "app_store" in found:
        return ["app_store"]

    if "ios_update" in found:
        return ["ios_update"]

    return found


def detect_response_domains(response: str) -> List[str]:
    """
    Detect domains in a historical AppleSupport response.
    """

    if not response:
        return []

    normalized = normalize_text(response)

    found = []

    for domain, patterns in RESPONSE_DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                found.append(domain)
                break

    return found


def are_domains_compatible(
    query_domains: List[str],
    response_domains: List[str]
) -> Tuple[bool, str]:
    """Check domain compatibility between query and response."""

    if not query_domains or not response_domains:
        return False, "missing_domains"

    # Direct match
    if set(query_domains) & set(response_domains):
        return True, "direct_domain_match"

    # Hierarchical match
    for q in query_domains:
        for r in response_domains:
            if (q, r) in COMPATIBLE_HIERARCHIES:
                return True, f"compatible_hierarchy: {q} and {r}"

    # Explicit conflicts
    for q in query_domains:
        for r in response_domains:
            if (
                (q, r) in CONFLICTING_DOMAINS
                or
                (r, q) in CONFLICTING_DOMAINS
            ):
                return False, f"conflicting_domains: {q} vs {r}"

    return False, "unrelated_domains"


# ============================================================
# RESPONSE CLASSIFICATION HELPERS
# ============================================================

ACTION_WORDS = [
    "try",
    "check",
    "restart",
    "force restart",
    "reset",
    "remove",
    "install",
    "reinstall",
    "update",
    "turn on",
    "turn off",
    "toggle",
    "settings >",
    "settings",
    "delete",
    "restore",
    "back up",
    "sign out",
    "sign in",
    "unpair",
    "pair",
    "clean",
    "disconnect",
    "reconnect",
    "charge",
    "plug in",
    "inspect",
    "reseat",
    "insert",
    "follow these steps",
    "support guide",
    "article",
]


GENERIC_BOILERPLATE_PATTERNS = [
    r"^we(?:'| a)?d (?:love|like|be happy|be glad) to help(?:\s+with this|\s+out)?[\.\!\?]?$",
    r"^(?:please\s+)?send us a dm(?:\s+so we can help|\s+to get started)?[\.\!\?]?$",
    r"^(?:please\s+)?dm us[\.\!\?]?$",
    r"^we('?| )?ve responded to your dm[\.\!\?]?$",
    r"^we('?| )?ve received your dm[\.\!\?]?$",
    r"^look for us there[\.\!\?]?$",
    r"^reach out (?:in|via) dm[\.\!\?]?$",
    r"^join us in dm[\.\!\?]?$",
    r"^let us know (?:how it goes|if that helps|if you need anything)[\.\!\?]?$",
    r"^thanks for reaching out[\.\!\?]?$",
    r"^we're here to help[\.\!\?]?$",
    r"^(?:have you tried\s+|try\s+)?restarting\s+(?:your\s+)?(?:device|phone|iphone|ipad|mac|it)(?:\s+and\s+see\s+if\s+that\s+helps|\s+to\s+see\s+if\s+that\s+helps|\s+to\s+see\s+if\s+that\s+resolves\s+the\s+issue)?[\.\!\?]?$",
    r"^(?:please\s+)?(?:try to\s+)?restart\s+(?:your\s+)?(?:device|phone|iphone|ipad|mac|it)(?:\s+and\s+see\s+if\s+that\s+helps|\s+to\s+see\s+if\s+that\s+helps|\s+to\s+see\s+if\s+that\s+resolves\s+the\s+issue)?[\.\!\?]?$",
    r"^(?:please\s+)?contact\s+(?:apple\s+)?support[\.\!\?]?$",
    # DM-redirect and reach-out patterns that don't include specific troubleshooting
    r"^(?:please\s+)?(?:feel free to\s+)?reach out to us[\.\!\?]?$",
    r"^(?:please\s+)?reach out to (?:apple|our|the)\s+(?:support|team)[\.\!\?]?$",
    r"^let us know (?:in|via|through|by) (?:a\s+)?dm[\.\!\?]?$",
    r"^(?:please\s+)?send us a (?:direct\s+)?message[\.\!\?]?$",
    r"^we(?:'?d|\s+would) (?:love|like) to (?:chat|talk|discuss) (?:more\s+)?(?:in|via|through) dm[\.\!\?]?$",
    r"^(?:please\s+)?head over to our dm[\.\!\?]?$",
]


def is_generic_response(response: str) -> bool:
    """Check if response is purely generic."""

    if not response:
        return True

    normalized = normalize_text(response)

    if not normalized:
        return True

    for pat in GENERIC_BOILERPLATE_PATTERNS:
        if re.search(pat, normalized):
            return True

    words = normalized.split()

    if len(words) <= 6:
        if not any(
            w in normalized
            for w in [
                "restart",
                "settings",
                "update",
                "battery",
                "wifi",
                "sim"
            ]
        ):
            return True

    return False


def response_contains_action(text: str) -> bool:
    """Check if response contains actionable troubleshooting directives."""

    if not text:
        return False

    normalized = normalize_text(text)

    return any(
        word in normalized
        for word in ACTION_WORDS
    )


def is_diagnostic_question(text: str) -> bool:
    """Check if response contains substantive diagnostic inquiry."""

    if not text or "?" not in text:
        return False

    normalized = normalize_text(text)

    diag_patterns = [
        r"\bdoes this happen\b",
        r"\bwhat (?:version|model|happens|error)\b",
        r"\bwhich (?:version|model|device|app)\b",
        r"\bwhen did this (?:start|begin)\b",
        r"\bis this happening\b",
        r"\bare you seeing\b",
        r"\bhave you tried\b",
        r"\bcan you test\b",
        r"\bare you able to\b",
        r"\bdo you know\b",
    ]

    return any(
        re.search(p, normalized)
        for p in diag_patterns
    )


def extract_guidance_components(
    response: str,
    query_domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyze response to verify usable guidance."""

    cleaned = clean_historical_response(response)

    if not cleaned:
        return {
            "usable": False,
            "category": "empty",
            "text": "",
            "reason": "empty_response"
        }

    if is_generic_response(cleaned):
        return {
            "usable": False,
            "category": "generic",
            "text": "",
            "reason": "generic_response"
        }

    resp_domains = detect_response_domains(cleaned)

    has_action = response_contains_action(cleaned)
    has_diag = is_diagnostic_question(cleaned)

    if query_domains:
        compatible, _ = are_domains_compatible(
            query_domains,
            resp_domains
        )

        if compatible and (has_action or has_diag):
            category = (
                "clarification"
                if has_diag and not has_action
                else "specific_guidance"
            )

            return {
                "usable": True,
                "category": category,
                "text": cleaned,
                "reason": "domain_specific_guidance"
            }

    if has_action:
        return {
            "usable": True,
            "category": "specific_guidance",
            "text": cleaned,
            "reason": "actionable_guidance"
        }

    if has_diag:
        return {
            "usable": True,
            "category": "clarification",
            "text": cleaned,
            "reason": "diagnostic_question"
        }

    return {
        "usable": False,
        "category": "unsupported",
        "text": "",
        "reason": "no_actionable_guidance"
    }


# ============================================================
# SENSITIVE TOPICS
# ============================================================

def check_sensitive_topic(text: str) -> Tuple[bool, str]:
    """Identify high-risk topics requiring human escalation."""

    normalized = normalize_text(text)

    # 1. Account security / phishing / hacking / takeover
    security_patterns = [
        (r"\b(fake account|fake email|fake e-mail|suspicious email|suspicious e-mail|phishing|scam|scammer|fraud|fake apple|pretending to be apple|security alert|suspicious message|suspicious text)\b", "account_security_or_phishing"),
        (r"\b(hack(ed|ing)?|someone hacked)\b", "account_security"),
        (r"\b(someone (accessed|logged into|used) my account)\b", "account_security"),
        (r"\b(unauthorized (access|login|activity|use))\b", "account_security"),
        (r"\b(someone changed my password)\b", "account_security"),
        (r"\b(suspicious activity|account (was |is )?compromised|compromised account|security breach)\b", "account_security"),
        (r"\b((apple id|account|icloud) (is |was |has been )?(locked|disabled))\b", "account_security"),
        (r"\b((locked|disabled) for security reasons)\b", "account_security"),
        (r"\b(locked out of (my )?(apple id|icloud|account))\b", "account_security"),
        (r"\b(account (is |was )?locked|apple id (is |was )?locked|account disabled)\b", "account_security"),
    ]

    for p, reason in security_patterns:
        if re.search(p, normalized):
            return True, reason

    # 2. Financial / payment / billing disputes / refunds
    payment_patterns = [
        (r"\b(charged (twice|wrong|extra|again)|duplicate charge|billed twice|overcharged)\b", "payment_billing"),
        (r"\b(unauthorized (charge|transaction|payment|fee)|unrecognized charge)\b", "payment_billing"),
        (r"\b(money (taken|deducted) from my (account|card|bank))\b", "payment_billing"),
        (r"\b(someone used my (card|credit card|debit card))\b", "payment_billing"),
        (r"\b(credit card fraud|bank dispute|billing dispute|charged without my permission)\b", "payment_billing"),
        (r"\b(credit card|debit card|card number|bank account|chargeback)\b", "payment_billing"),
        (r"\b(refund|money back|credit my account)\b", "refund"),
    ]

    for pat, reason in payment_patterns:
        if re.search(pat, normalized):
            return True, reason

    # 3. Legal / Law enforcement
    legal_patterns = [
        (r"\b(legal action|take legal action|legal complaint|lawsuit|sue apple|sue you|suing|attorney|lawyer|legal counsel|legal team|regulatory complaint|ftc complaint)\b", "legal"),
        (r"\bpolice\b", "legal"),
    ]

    for pat, reason in legal_patterns:
        if re.search(pat, normalized):
            return True, reason

    # 4. Privacy / Data breach
    privacy_patterns = [
        (r"\b(personal data (deleted|exported|leaked|exposed)|delete my (personal )?data|private data exposed|privacy violation|gdpr|data leak|data breach)\b", "privacy"),
    ]

    for pat, reason in privacy_patterns:
        if re.search(pat, normalized):
            return True, reason

    # 5. Stolen device / Hardware hazard
    hazard_patterns = [
        (r"\b(stolen (device|phone|iphone|ipad|mac)|lost (my )?phone.*stolen|my (phone|iphone) was stolen)\b", "device_theft"),
        (r"\b(activation lock bypass|bypass (icloud|activation lock))\b", "device_ownership"),
        (r"\b(swollen battery|battery (smoking|swelling|exploded)|smoking iphone|fire hazard)\b", "hardware_hazard"),
    ]

    for pat, reason in hazard_patterns:
        if re.search(pat, normalized):
            return True, reason

    return False, ""


# ============================================================
# RESOLUTION DETECTION
# ============================================================

def is_message_resolved(text: str) -> bool:
    """
    Check whether the customer explicitly confirms resolution.

    IMPORTANT:
    "thanks" or "thank you" alone does NOT mean the issue
    has been resolved.
    """

    normalized = normalize_text(text)

    # Inquiries or confirmation questions are NOT resolutions
    if "?" in str(text):
        if re.search(r"\b(is that correct|is this true|is it|can i|right\?)\b", normalized):
            return False

    strong_resolution_patterns = [
        r"\bthat did it\b",
        r"\bthis did it\b",
        r"\bit worked\b",
        r"\bworks now\b",
        r"\bworking now\b",
        r"\bproblem solved\b",
        r"\bsolved\b",
        r"\bfixed now\b",
        r"\bfixed it\b",
        r"\bgot it working\b",
        r"\blooks good now\b",
        r"\bissue is resolved\b",
        r"\bproblem is resolved\b",
        r"\bthat fixed it\b",
        r"\bfigured it out\b",
        r"\bno longer need help\b",
        r"\balready fixed\b",
        r"\balready solved\b",
        r"\b(?:i think i )?found the answer\b",
        r"\bnever mind\b",
        r"\bgot it,?\s*thanks?\b",
        r"\ball fixed\b",
        r"\ball good now\b",
    ]

    if any(re.search(pattern, normalized) for pattern in strong_resolution_patterns):
        return True

    return False


# ============================================================
# INTENT PREDICTION
# ============================================================

def predict_intent(
    text: str,
    supplied_intent: Optional[str] = None
) -> Tuple[str, float]:
    """Heuristic intent prediction for standalone use."""

    domains = detect_domains(text)

    if "account_security" in domains:
        return "other_unclear", 1.0

    if supplied_intent:
        return supplied_intent, 1.0

    if "battery" in domains:
        return "battery_charging", 0.95

    if "ios_update" in domains:
        return "ios_software_update", 0.95

    if "wifi" in domains:
        return "wifi_connectivity", 0.95

    if "app_store" in domains:
        return "app_store_downloads", 0.95

    if "apple_music" in domains:
        return "apple_music_itunes", 0.95

    if "apple_id" in domains:
        return "apple_id_icloud", 0.90

    if "screen" in domains:
        return "screen_display", 0.95

    if "speaker_audio" in domains:
        return "audio_speaker", 0.95

    if "camera" in domains:
        return "device_hardware", 0.85

    if (
        "cellular_calls" in domains
        or "carrier_unlock" in domains
        or "sim_hardware" in domains
    ):
        return "calls_cellular", 0.90

    if (
        "apple_logo_boot" in domains
        or "power_on" in domains
    ):
        return "device_hardware", 0.90

    if (
        "youtube" in domains
        or "app_problem" in domains
    ):
        return "app_problems", 0.85

    return "other_unclear", 0.20


# ============================================================
# DEBUG PRINTER
# ============================================================

def _print_debug_evidence(
    query: str,
    query_domains: List[str],
    hist_customer: str,
    hist_response: str,
    accepted: bool,
    reason: str
):
    print("\n" + "=" * 50)

    print("QUERY:")
    print(query)

    print("\nQUERY DOMAINS:")
    print(query_domains)

    print("\nHISTORICAL CUSTOMER:")
    print(
        hist_customer[:120]
        if hist_customer
        else "None"
    )

    print("\nHISTORICAL CUSTOMER DOMAINS:")
    print(
        detect_domains(hist_customer)
    )

    print("\nHISTORICAL RESPONSE:")
    print(
        hist_response[:120]
        if hist_response
        else "None"
    )

    print("\nRESPONSE DOMAINS:")
    print(
        detect_response_domains(hist_response)
    )

    print("\nRESULT:")
    print(
        "ACCEPTED"
        if accepted
        else "REJECTED"
    )

    print("\nREASON:")
    print(reason)

    print("=" * 50 + "\n")


# ============================================================
# EVIDENCE SAFETY GATE
# ============================================================

def is_evidence_safe(
    query: str,
    evidence_candidate: Any = None,
    query_intent: Optional[str] = None,
    evidence_customer: Optional[str] = None,
    evidence_response: Optional[str] = None,
    query_domains: Optional[List[str]] = None,
    debug: bool = False,
    **kwargs
) -> Tuple[bool, str]:
    """
    Deterministically evaluates whether historical evidence
    is safe and applicable.
    """

    cand = (
        evidence_candidate
        if evidence_candidate is not None
        else kwargs.get("candidate", None)
    )

    cust_text = ""
    resp_text = ""

    if isinstance(cand, (dict, pd.Series)):
        cust_text = str(
            cand.get(
                "customer_text",
                cand.get(
                    "text",
                    cand.get(
                        "query",
                        ""
                    )
                )
            )
        )

        resp_text = str(
            cand.get(
                "historical_response",
                cand.get(
                    "response",
                    cand.get(
                        "response_text",
                        ""
                    )
                )
            )
        )

        if isinstance(query_intent, (list, set)):
            query_domains = list(query_intent)

    elif isinstance(cand, str):
        cust_text = cand

        if (
            isinstance(query_intent, str)
            and evidence_response is None
        ):
            resp_text = query_intent

    if evidence_customer is not None:
        cust_text = str(evidence_customer)

    if evidence_response is not None:
        resp_text = str(evidence_response)

    # --------------------------------------------------------
    # 1. Empty evidence
    # --------------------------------------------------------

    if not resp_text.strip():
        if debug:
            _print_debug_evidence(
                query,
                query_domains or [],
                cust_text,
                resp_text,
                False,
                "empty_evidence"
            )

        return False, "empty_evidence"

    # --------------------------------------------------------
    # 2. Generic response
    # --------------------------------------------------------

    if is_generic_response(resp_text):
        if debug:
            _print_debug_evidence(
                query,
                query_domains or [],
                cust_text,
                resp_text,
                False,
                "generic_response"
            )

        return False, "generic_response"

    # --------------------------------------------------------
    # 3. Domains
    # --------------------------------------------------------

    q_domains = (
        query_domains
        if query_domains is not None
        else detect_domains(query)
    )

    hist_cust_domains = detect_domains(
        cust_text
    )

    resp_domains = detect_response_domains(
        resp_text
    )

    # --------------------------------------------------------
    # 4. Intent/domain consistency
    # --------------------------------------------------------

    if (
        query_intent
        and isinstance(query_intent, str)
        and query_intent in INTENT_DOMAINS
    ):
        expected_domains = INTENT_DOMAINS[
            query_intent
        ]

        if (
            expected_domains
            and q_domains
            and not set(q_domains) & expected_domains
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    f"intent_domain_mismatch: {query_intent}"
                )

            return (
                False,
                f"intent_domain_mismatch: {query_intent}"
            )

    # --------------------------------------------------------
    # 5. Normalized text
    # --------------------------------------------------------

    q_norm = normalize_text(query)
    r_norm = normalize_text(resp_text)

    # --------------------------------------------------------
    # 6. Action / diagnostic checks
    # --------------------------------------------------------

    has_action = response_contains_action(
        resp_text
    )

    has_diag = is_diagnostic_question(
        resp_text
    )

    has_link = bool(
        re.search(
            r"https?://\S+",
            resp_text
        )
    ) or (
        "support.apple.com"
        in resp_text.lower()
    ) or (
        "apple.co"
        in resp_text.lower()
    )

    # --------------------------------------------------------
    # 7. Battery vs Wi-Fi/Bluetooth
    # --------------------------------------------------------

    if "battery" in q_domains:

        if (
            (
                "wifi" in resp_domains
                or "wi-fi" in r_norm
                or "bluetooth" in r_norm
            )
            and
            "battery" not in resp_domains
            and
            "charging" not in r_norm
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "response_domain_mismatch: resolution_divergence"
                )

            return (
                False,
                "response_domain_mismatch: resolution_divergence"
            )

    # --------------------------------------------------------
    # 8. Screen divergence
    # --------------------------------------------------------

    if "screen" in q_domains:

        if (
            "autocorrect" in r_norm
            or "keyboard" in r_norm
            or "letter i" in r_norm
            or "typing" in r_norm
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "issue_mismatch: resolution_divergence"
                )

            return (
                False,
                "issue_mismatch: resolution_divergence"
            )

        if (
            "facetime" in r_norm
            and "screen" not in r_norm
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "issue_mismatch: facetime_divergence"
                )

            return (
                False,
                "issue_mismatch: facetime_divergence"
            )

    # --------------------------------------------------------
    # 9. SIM / carrier divergence
    # --------------------------------------------------------

    if (
        "sim_hardware" in q_domains
        or (
            "carrier" in q_norm
            and "sim" in q_norm
        )
    ):

        if (
            "recovery mode" in r_norm
            or "won't power on" in r_norm
            or "restore in recovery" in r_norm
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "issue_mismatch: resolution_divergence"
                )

            return (
                False,
                "issue_mismatch: resolution_divergence"
            )

    # --------------------------------------------------------
    # 10. YouTube vs Apple Music
    # --------------------------------------------------------

    if "youtube" in q_domains:

        if (
            "apple_music" in resp_domains
            or "apple music" in r_norm
            or "itunes" in r_norm
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "application_mismatch"
                )

            return False, "application_mismatch"

    # --------------------------------------------------------
    # Carrier unlock vs physical SIM tray divergence
    # --------------------------------------------------------
    # Carrier unlock vs physical SIM tray divergence
    # --------------------------------------------------------
    if (
        ("carrier" in q_norm or "carrier lock" in q_norm or "unlock" in q_norm or "unlocking" in q_norm)
        and ("unlock" in q_norm or "unlocking" in q_norm)
    ):
        if (
            ("paperclip" in r_norm or "sim tray" in r_norm or "eject the sim" in r_norm or "sized sim card" in r_norm or "sim card size" in r_norm or "sim drawer" in r_norm)
            and not any(k in r_norm for k in ["carrier unlock", "unlocking", "carrier lock", "unlock policy"])
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "carrier_unlock_vs_tray_divergence"
                )
            return False, "carrier_unlock_vs_tray_divergence"

    # --------------------------------------------------------
    # App Store password settings vs download/Wi-Fi error
    # --------------------------------------------------------
    if (
        any(k in q_norm for k in ["require a password", "password for already purchased", "password for any new apps", "password settings", "free apps password", "allow no password"])
    ):
        if (
            any(k in r_norm for k in ["different wi-fi", "wi-fi network", "downloading those applications", "app store working again", "restart your device"])
            and not any(k in r_norm for k in ["password", "settings > itunes & app store", "apple id password", "require password"])
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "app_password_vs_download_network_divergence"
                )
            return False, "app_password_vs_download_network_divergence"

    # --------------------------------------------------------
    # USB car music playback vs restart divergence
    # --------------------------------------------------------
    if (
        "usb" in q_norm
        and any(k in q_norm for k in ["song starts", "song", "music", "car", "listening to", "auto play", "autoplay"])
    ):
        if (
            any(k in r_norm for k in [
                "restart your computer", "restarting your computer",
                "restarted your iphone", "restart your device", "restarting your device",
                "have you restarted", "have you tried restarting",
                "try restarting", "tried restarting",
            ])
            and not any(k in r_norm for k in ["usb", "carplay", "stereo", "vehicle", "head unit", "aux", "audio"])
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "usb_car_playback_vs_restart_divergence"
                )
            return False, "usb_car_playback_vs_restart_divergence"

    # --------------------------------------------------------
    # Battery health inquiry vs generic battery performance update
    # --------------------------------------------------------
    if "battery health" in q_norm or "figure out the battery health" in q_norm:
        if (
            ("article and send us a dm" in r_norm or "tell us, what's happening with the battery" in r_norm)
            and not any(k in r_norm for k in ["battery health", "maximum capacity", "service", "settings > battery"])
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "battery_health_vs_generic_performance_divergence"
                )
            return False, "battery_health_vs_generic_performance_divergence"

    # --------------------------------------------------------
    # App download vs app crash divergence
    # --------------------------------------------------------
    if (
        any(k in q_norm for k in ["download", "install", "cannot download", "won't download", "unable to download"])
        and not any(k in q_norm for k in ["crash", "freez", "force close", "quits unexpectedly", "shut down"])
    ):
        if (
            any(k in r_norm for k in ["force close", "crash", "quits unexpectedly", "stops responding"])
            and not any(k in r_norm for k in ["download", "install", "app store", "storage", "purchase"])
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "app_download_vs_crash_divergence"
                )
            return False, "app_download_vs_crash_divergence"

    if (
        any(k in q_norm for k in ["crash", "freez", "force close", "quits unexpectedly"])
        and not any(k in q_norm for k in ["download", "install", "unable to download"])
    ):
        if (
            any(k in r_norm for k in ["unable to download", "cannot download", "app store download", "payment method for download"])
            and not any(k in r_norm for k in ["crash", "freez", "close", "reinstall", "restart"])
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "app_crash_vs_download_divergence"
                )
            return False, "app_crash_vs_download_divergence"

    # --------------------------------------------------------
    # iMessage vs Wi-Fi divergence
    # --------------------------------------------------------
    if (
        ("imessage" in q_norm or "message activation" in q_norm or "imessage waiting for activation" in q_norm)
        and not any(k in q_norm for k in ["wifi", "wi-fi", "internet connection", "router"])
    ):
        if (
            ("reset network settings" in r_norm or "forget this network" in r_norm or "router" in r_norm)
            and not any(k in r_norm for k in ["imessage", "message", "apple id", "send & receive", "activation"])
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "imessage_vs_wifi_divergence"
                )
            return False, "imessage_vs_wifi_divergence"

    # --------------------------------------------------------
    # 11. Apple ID divergence
    # --------------------------------------------------------

    if (
        "apple_id" in q_domains
        or "icloud" in q_norm
        or "apple id" in q_norm
    ):

        if (
            "camera" in resp_domains
            or "camera" in r_norm
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "issue_mismatch: response_domain_mismatch"
                )

            return (
                False,
                "issue_mismatch: response_domain_mismatch"
            )

        if (
            "turn on" in r_norm
            or "power on" in r_norm
            or "won't power on" in r_norm
        ):
            if debug:
                _print_debug_evidence(
                    query,
                    q_domains,
                    cust_text,
                    resp_text,
                    False,
                    "account_problem_power_guidance_mismatch"
                )

            return (
                False,
                "account_problem_power_guidance_mismatch"
            )

    # --------------------------------------------------------
    # 12. Specific Problem Match conflict
    # --------------------------------------------------------

    conflict_detected, conflict_reason = detect_problem_conflict(
        query=query,
        query_problems=extract_problem_terms(query),
        hist_customer=cust_text,
        hist_response=resp_text,
    )

    if conflict_detected:

        if debug:
            _print_debug_evidence(
                query,
                q_domains,
                cust_text,
                resp_text,
                False,
                conflict_reason
            )

        return False, conflict_reason

    # --------------------------------------------------------
    # 13. Explicit conflicting domains
    # --------------------------------------------------------

    for q_dom in q_domains:

        for r_dom in resp_domains:

            if (
                q_dom,
                r_dom
            ) in CONFLICTING_DOMAINS:

                if debug:
                    _print_debug_evidence(
                        query,
                        q_domains,
                        cust_text,
                        resp_text,
                        False,
                        f"conflicting_domains: {q_dom} vs {r_dom}"
                    )

                return (
                    False,
                    f"conflicting_domains: {q_dom} vs {r_dom}"
                )

    # --------------------------------------------------------
    # 14. General domain match
    # --------------------------------------------------------

    if q_domains:

        compatible, comp_reason = are_domains_compatible(
            q_domains,
            resp_domains
        )

        if not compatible:

            # Historical customer can still provide domain
            # alignment if the response itself does not contain
            # obvious domain terminology.

            if (
                not resp_domains
                and
                set(q_domains) & set(hist_cust_domains)
                and
                (has_action or has_link)
            ):
                pass

            elif not resp_domains:

                if debug:
                    _print_debug_evidence(
                        query,
                        q_domains,
                        cust_text,
                        resp_text,
                        False,
                        "no_response_domain"
                    )

                return False, "no_response_domain"

            else:

                if debug:
                    _print_debug_evidence(
                        query,
                        q_domains,
                        cust_text,
                        resp_text,
                        False,
                        f"response_domain_mismatch: {comp_reason}"
                    )

                return (
                    False,
                    f"response_domain_mismatch: {comp_reason}"
                )

    # --------------------------------------------------------
    # 15. Actionable guidance
    # --------------------------------------------------------

    if not (
        has_action
        or has_diag
        or has_link
    ):

        if debug:
            _print_debug_evidence(
                query,
                q_domains,
                cust_text,
                resp_text,
                False,
                "no_actionable_guidance"
            )

        return False, "no_actionable_guidance"

    # --------------------------------------------------------
    # 16. Specific evidence safety
    # --------------------------------------------------------

    spec_safe, spec_reason = is_specific_evidence_safe(
        query=query,
        evidence_customer=cust_text,
        evidence_response=resp_text,
        current_intent=query_intent,
        query_domains=q_domains,
        resp_domains=resp_domains,
    )

    if (
        not spec_safe
        and
        (
            "problem" in spec_reason
            or "conflict" in spec_reason
        )
    ):

        if debug:
            _print_debug_evidence(
                query,
                q_domains,
                cust_text,
                resp_text,
                False,
                spec_reason
            )

        return False, spec_reason

    # --------------------------------------------------------
    # ACCEPTED
    # --------------------------------------------------------

    if debug:
        _print_debug_evidence(
            query,
            q_domains,
            cust_text,
            resp_text,
            True,
            "domain_match + actionable_guidance"
        )

    return True, "domain_match + actionable_guidance"


# ============================================================
# EVIDENCE SCORING
# ============================================================

def score_evidence_candidate(
    query: str,
    evidence_customer: str,
    evidence_response: str,
    base_score: float = 0.0
) -> Dict[str, Any]:

    query_domains = detect_domains(
        query
    )

    hist_cust_domains = detect_domains(
        evidence_customer
    )

    resp_domains = detect_response_domains(
        evidence_response
    )

    safe, safety_reason = is_evidence_safe(
        query=query,
        evidence_customer=evidence_customer,
        evidence_response=evidence_response,
        query_domains=query_domains
    )

    guidance = extract_guidance_components(
        evidence_response,
        query_domains
    )

    resp_domain_match = bool(
        set(query_domains)
        &
        set(resp_domains)
    )

    cust_domain_match = bool(
        set(query_domains)
        &
        set(hist_cust_domains)
    )

    if not safe:

        score = base_score - 1.0

    else:

        score = float(base_score)

        if resp_domain_match:
            score += 0.30

        elif cust_domain_match:
            score += 0.15

        if guidance["usable"]:
            score += 0.20

        if guidance["category"] == "specific_guidance":
            score += 0.10

    spec_eval = calculate_final_evidence_score(
        query=query,
        historical_customer_message=evidence_customer,
        historical_response=evidence_response,
        current_intent="",
        historical_intent="",
        query_domains=query_domains,
        resp_domains=resp_domains,
    )

    # ------------------------------------------------------------------
    # SPECIFICITY BONUS / PENALTY
    # Reward evidence that specifically addresses the customer's sub-problem;
    # apply a mild penalty for very-low-specificity guidance.
    # This ONLY affects ranking within safe candidates —
    # it does NOT change safety gates or hard rejection thresholds.
    # ------------------------------------------------------------------
    if safe:
        cust_sim = float(
            spec_eval.get("customer_problem_similarity", 0.0) or 0.0
        )
        resp_sim = float(
            spec_eval.get("response_problem_similarity", 0.0) or 0.0
        )
        specificity = (cust_sim + resp_sim) / 2.0

        if specificity >= 0.45:
            score += 0.25  # strong specificity: evidence closely matches sub-problem
        elif specificity >= 0.30:
            score += 0.12  # moderate specificity bonus
        elif specificity < 0.12 and guidance["category"] != "specific_guidance":
            score -= 0.10  # mild penalty: very-low-specificity generic guidance

    return {
        "score": score,
        "safe": safe,
        "reason": safety_reason,
        "guidance": guidance,
        "query_domains": query_domains,
        "hist_cust_domains": hist_cust_domains,
        "resp_domains": resp_domains,

        "customer_problem_similarity":
            spec_eval.get(
                "customer_problem_similarity",
                0.0
            ),

        "response_problem_similarity":
            spec_eval.get(
                "response_problem_similarity",
                0.0
            ),

        "domain_match":
            spec_eval.get(
                "domain_match",
                0.0
            ),

        "action_match":
            spec_eval.get(
                "action_match",
                0.0
            ),

        "conflict_detected":
            spec_eval.get(
                "conflict_detected",
                False
            ),

        "final_evidence_score":
            spec_eval.get(
                "final_evidence_score",
                score
            ),

        "evidence_rejection_reason":
            safety_reason,
    }


def rerank_evidence(
    query: str,
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    scored = []

    for candidate in candidates:

        customer_text = candidate.get(
            "customer_text",
            candidate.get(
                "text",
                candidate.get(
                    "query",
                    ""
                )
            )
        )

        response_text = candidate.get(
            "historical_response",
            candidate.get(
                "response",
                candidate.get(
                    "response_text",
                    ""
                )
            )
        )

        base_score = float(
            candidate.get(
                "final_score",
                candidate.get(
                    "retrieval_score",
                    candidate.get(
                        "evidence_relevance_score",
                        candidate.get(
                            "similarity",
                            0.0
                        )
                    )
                )
            )
        )

        result = score_evidence_candidate(
            query=query,
            evidence_customer=customer_text,
            evidence_response=response_text,
            base_score=base_score
        )

        c = dict(candidate)

        c["evidence_score"] = result["score"]

        c["evidence_safe"] = result["safe"]

        c["rejection_reason"] = result["reason"]

        c["guidance_category"] = (
            result["guidance"]["category"]
        )

        c["query_domains"] = (
            result["query_domains"]
        )

        c["response_domains"] = (
            result["resp_domains"]
        )

        c["customer_problem_similarity"] = (
            result.get(
                "customer_problem_similarity",
                candidate.get(
                    "customer_problem_similarity",
                    0.0
                )
            )
        )

        c["response_problem_similarity"] = (
            result.get(
                "response_problem_similarity",
                candidate.get(
                    "response_problem_similarity",
                    0.0
                )
            )
        )

        c["domain_match"] = (
            result.get(
                "domain_match",
                candidate.get(
                    "domain_match",
                    0.0
                )
            )
        )

        c["action_match"] = (
            result.get(
                "action_match",
                candidate.get(
                    "action_match",
                    0.0
                )
            )
        )

        c["conflict_detected"] = (
            result.get(
                "conflict_detected",
                candidate.get(
                    "conflict_detected",
                    False
                )
            )
        )

        c["final_evidence_score"] = (
            result.get(
                "final_evidence_score",
                candidate.get(
                    "final_evidence_score",
                    0.0
                )
            )
        )

        c["evidence_rejection_reason"] = (
            result["reason"]
        )

        scored.append(c)

    # Safe evidence first.
    #
    # Within safe evidence, prefer the highest
    # evidence score.

    scored.sort(
        key=lambda x: (
            1 if x["evidence_safe"] else 0,
            float(
                x.get(
                    "final_evidence_score",
                    0.0
                )
            ),
            float(
                x.get(
                    "evidence_score",
                    0.0
                )
            )
        ),
        reverse=True
    )

    return scored


# ============================================================
# INTENT-AWARE SAFE CLARIFICATION
# ============================================================

INTENT_CLARIFICATION = {

    "battery_charging":
        "Could you tell us what is happening with the battery or charging, such as fast draining or not charging?",

    "ios_update":
        "Could you tell us what is happening with the iOS update, such as downloading, installing, or failing?",

    "ios_software_update":
        "Could you tell us what is happening with the iOS update, such as downloading, installing, or failing?",

    "wifi_connectivity":
        "Could you tell us what is happening with Wi-Fi, such as not connecting or frequently disconnecting?",

    "app_problems":
        "Could you tell us which app is affected and what happens when you try to use it?",

    "app_store_downloads":
        "Could you tell us whether the issue is with downloading or installing an app?",

    "apple_music_itunes":
        "Could you tell us what is happening with Apple Music or iTunes?",

    "apple_id_icloud":
        "Could you tell us whether the issue is with signing in, your password, or iCloud?",

    "screen_display":
        "Could you tell us what is happening with the screen, such as a black display, touch issue, or flickering?",

    "calls_cellular":
        "Could you tell us whether the issue is with making calls, cellular data, or getting a signal?",

    "audio_speaker":
        "Could you tell us whether the issue is with the speaker, microphone, or overall audio?",

    "device_hardware":
        "Could you provide more details about the hardware issue, including what happens when you use the device?",

    "other_unclear":
        "Could you provide a little more detail about the issue you're experiencing?",
}


def generate_safe_clarification(
    query: str,
    intent: str
) -> str:
    """
    Generate a safe diagnostic question.

    Prefers specific contextual problem targeting, then intent clarification,
    then domain fallback.
    """
    norm_q = normalize_text(query)

    # 1. Targeted sub-problem clarifications for specific customer contexts
    if "battery health" in norm_q or "figure out the battery health" in norm_q:
        return "Could you tell us what battery percentage or message you see in Settings > Battery > Battery Health?"

    if ("unlock" in norm_q or "unlocking" in norm_q) and ("carrier" in norm_q or "sim" in norm_q):
        return "Could you clarify if you're looking to carrier unlock your iPhone to use with another network, or having trouble with the SIM tray?"

    if "password" in norm_q and any(w in norm_q for w in ["purchased", "new apps", "free apps", "app store"]):
        return "Could you clarify whether you're trying to adjust password requirements for free or purchased apps in Settings > iTunes & App Store?"

    if "usb" in norm_q and any(w in norm_q for w in ["song starts", "car", "stereo", "music"]):
        return "Could you clarify if music automatically starts playing when connecting to your vehicle or computer via USB?"

    if any(w in norm_q for w in ["calendar", "whatsapp", "instagram", "facebook", "twitter", "snapchat"]):
        return "Could you tell us what occurs when you use the app, including any error messages you see?"

    # 2. Intent-aware clarification
    if intent in INTENT_CLARIFICATION:
        return INTENT_CLARIFICATION[intent]

    # --------------------------------------------------------
    # Domain fallback
    # --------------------------------------------------------

    domains = detect_domains(query)

    if "battery" in domains:
        return (
            "Could you tell us what is happening with the "
            "battery or charging, such as fast draining or "
            "not charging?"
        )

    if "wifi" in domains:
        return (
            "Could you tell us what is happening with Wi-Fi, "
            "such as not connecting or frequently disconnecting?"
        )

    if "screen" in domains:
        return (
            "Could you tell us what is happening with the "
            "screen, such as a black display, touch issue, "
            "or flickering?"
        )

    if "speaker_audio" in domains:
        return (
            "Could you tell us whether the issue is with the "
            "speaker, microphone, or overall audio?"
        )

    if "camera" in domains:
        return (
            "Could you tell us what is happening with the "
            "camera, such as not opening or not taking photos?"
        )

    if "app_store" in domains:
        return (
            "Could you tell us whether the issue is with "
            "downloading or installing an app?"
        )

    if "youtube" in domains:
        return (
            "Could you tell us which part of YouTube is affected "
            "and what happens when you try to use it?"
        )

    if "apple_music" in domains:
        return (
            "Could you tell us what is happening with "
            "Apple Music or iTunes?"
        )

    if "apple_id" in domains:
        return (
            "Could you tell us whether the issue is with "
            "signing in, your password, or iCloud?"
        )

    if "ios_update" in domains:
        return (
            "Could you tell us what is happening with the "
            "software update, such as downloading, installing, "
            "or failing?"
        )

    if "carrier_unlock" in domains:
        return (
            "Are you trying to use an iPhone with a SIM "
            "from another carrier?"
        )

    if "sim_hardware" in domains:
        return (
            "Could you tell us what is happening with the "
            "SIM card, tray, or SIM slot?"
        )

    if "cellular_calls" in domains:
        return (
            "Could you tell us whether the issue is with "
            "making calls, cellular data, or getting a signal?"
        )

    if "apple_logo_boot" in domains:
        return (
            "Could you tell us whether the device is stuck "
            "on the Apple logo or repeatedly restarting?"
        )

    if "power_on" in domains:
        return (
            "Could you tell us what happens when you try "
            "to turn the device on?"
        )

    if "app_problem" in domains:
        return (
            "Could you tell us which app is affected and "
            "what happens when you try to use it?"
        )

    return (
        "Could you provide a little more detail about the "
        "issue you're experiencing, including your device "
        "model and iOS version?"
    )


# ============================================================
# GROUNDED REPLY BUILDER
# ============================================================

def build_grounded_reply(
    query: str,
    intent: str,
    evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Sensitive topic
    # --------------------------------------------------------

    sensitive, sensitive_reason = check_sensitive_topic(
        query
    )

    if sensitive:

        return {
            "reply":
                "This may involve account security or sensitive "
                "information. Please contact official Apple Support "
                "directly so a specialist can assist you.",

            "response_type":
                "escalation",

            "grounded":
                True,

            "unsupported_claims":
                [],

            "safe_to_answer":
                False,

            "escalate":
                True,

            "escalation_decision":
                "ESCALATE",

            "escalation_reason":
                sensitive_reason,

            "evidence_used":
                False,

            "evidence_safe":
                False,

            "evidence":
                None,

            "evidence_id":
                None,
        }

    # --------------------------------------------------------
    # Already resolved
    # --------------------------------------------------------

    if is_message_resolved(query):

        norm_q = normalize_text(query)
        if "bug" in norm_q:
            resolved_msg = "Glad to hear restarting resolved it! If the issue reoccurs or appears to be a software bug, let us know what iOS version you're running so we can look into it."
        elif any(w in norm_q for w in ["thank", "you helped", "that helped"]):
            resolved_msg = "You're welcome! Glad to hear that helped. Let us know if you need anything else."
        else:
            resolved_msg = "Glad to hear your issue is resolved. Let us know if you need anything else in the future."

        return {
            "reply":
                resolved_msg,

            "response_type":
                "resolved_acknowledgement",

            "grounded":
                True,

            "unsupported_claims":
                [],

            "safe_to_answer":
                True,

            "escalate":
                False,

            "escalation_decision":
                "AUTO_HANDLE",

            "escalation_reason":
                "",

            "evidence_used":
                False,

            "evidence_safe":
                False,

            "evidence":
                None,

            "evidence_id":
                None,
        }

    # --------------------------------------------------------
    # No safe evidence
    # --------------------------------------------------------

    if (
        not evidence
        or
        not evidence.get(
            "evidence_safe",
            False
        )
    ):

        rej_reason = (
            evidence.get(
                "rejection_reason",
                "no_evidence"
            )
            if evidence
            else "no_evidence"
        )

        return {
            "reply":
                generate_safe_clarification(
                    query,
                    intent
                ),

            "response_type":
                "insufficient_evidence",

            "grounded":
                True,

            "unsupported_claims":
                [],

            "safe_to_answer":
                True,

            "escalate":
                False,

            "escalation_decision":
                "AUTO_HANDLE",

            "escalation_reason":
                "",

            "evidence_used":
                False,

            "evidence_safe":
                False,

            "evidence":
                None,

            "evidence_id":
                None,

            "rejection_reason":
                rej_reason,
        }

    # --------------------------------------------------------
    # Extract evidence response
    # --------------------------------------------------------

    evidence_response = evidence.get(
        "response_text",
        evidence.get(
            "response",
            evidence.get(
                "historical_response",
                ""
            )
        )
    )

    evidence_id = evidence.get(
        "tweet_id",
        evidence.get(
            "response_id",
            evidence.get(
                "id"
            )
        )
    )

    query_domains = detect_domains(
        query
    )

    guidance = extract_guidance_components(
        evidence_response,
        query_domains
    )

    useful_text = guidance[
        "text"
    ].strip()

    # --------------------------------------------------------
    # Evidence contains no usable guidance
    # --------------------------------------------------------

    if (
        not useful_text
        or
        len(useful_text) < 10
    ):

        return {
            "reply":
                generate_safe_clarification(
                    query,
                    intent
                ),

            "response_type":
                "insufficient_evidence",

            "grounded":
                True,

            "unsupported_claims":
                [],

            "safe_to_answer":
                True,

            "escalate":
                False,

            "escalation_decision":
                "AUTO_HANDLE",

            "escalation_reason":
                "",

            "evidence_used":
                False,

            "evidence_safe":
                False,

            "evidence":
                None,

            "evidence_id":
                None,

            "rejection_reason":
                "evidence_has_no_extractable_guidance",
        }

    # --------------------------------------------------------
    # Construct response
    # --------------------------------------------------------

    if guidance["category"] == "clarification":

        reply = (
            "To narrow this down, "
            + useful_text
        )

        response_type = (
            "clarification_question"
        )

    else:

        reply = (
            "Based on a similar Apple Support case, "
            + useful_text
        )

        response_type = (
            "troubleshooting"
        )

    reply = re.sub(
        r"\s+",
        " ",
        reply
    ).strip()

    return {
        "reply":
            reply,

        "response_type":
            response_type,

        "grounded":
            True,

        "unsupported_claims":
            [],

        "safe_to_answer":
            True,

        "escalate":
            False,

        "escalation_decision":
            "AUTO_HANDLE",

        "escalation_reason":
            "",

        "evidence_used":
            True,

        "evidence_safe":
            True,

        "evidence":
            evidence,

        "evidence_id":
            evidence_id,

        "rejection_reason":
            "",
    }


# ============================================================
# DETERMINISTIC GROUNDING CHECK
# ============================================================

def perform_deterministic_grounding_check(
    query: str,
    reply: str,
    evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Verify that technical directives in the reply are
    supported by accepted evidence.
    """

    unsupported_claims = []

    if not reply:
        return {
            "grounded": False,
            "unsupported_claims": [
                "empty_reply"
            ]
        }

    sensitive, _ = check_sensitive_topic(
        query
    )

    if sensitive:
        return {
            "grounded": True,
            "unsupported_claims": []
        }

    if is_message_resolved(query):
        return {
            "grounded": True,
            "unsupported_claims": []
        }

    # Safe clarification responses don't introduce
    # unsupported technical directives.

    reply_lower = reply.lower()

    if (
        "could you" in reply_lower
        or
        "to narrow this down" in reply_lower
    ):
        return {
            "grounded": True,
            "unsupported_claims": []
        }

    if not evidence:

        if (
            "based on a similar"
            in reply_lower
        ):
            unsupported_claims.append(
                "claim_based_on_missing_evidence"
            )

        return {
            "grounded":
                len(unsupported_claims) == 0,

            "unsupported_claims":
                unsupported_claims
        }

    # Evidence must have passed the safety gate.

    if not evidence.get(
        "evidence_safe",
        False
    ):

        unsupported_claims.append(
            "unsafe_evidence: "
            +
            str(
                evidence.get(
                    "rejection_reason",
                    ""
                )
            )
        )

    return {
        "grounded":
            len(unsupported_claims) == 0,

        "unsupported_claims":
            unsupported_claims
    }


# ============================================================
# MAIN REPLY GENERATOR
# ============================================================

class ReplyGenerator:
    """
    Deterministic Grounded Reply Generator.

    Pipeline:

        Customer message
              ↓
        Intent prediction
              ↓
        Historical retrieval
              ↓
        Evidence reranking
              ↓
        Evidence safety gate
              ↓
        Grounded reply
              ↓
        Deterministic grounding check
              ↓
        Escalation decision
    """

    def __init__(
        self,
        retriever=None
    ):

        if retriever is None:

            try:

                from src.retrieval.retriever import (
                    AppleSupportRetriever
                )

                self.retriever = (
                    AppleSupportRetriever()
                )

            except Exception:

                self.retriever = None

        else:

            self.retriever = retriever

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    def _predict_intent_internal(
        self,
        query: str,
        intent: Optional[str] = None
    ) -> Tuple[str, float]:

        if intent:
            return intent, 1.0

        if (
            self.retriever
            and
            hasattr(
                self.retriever,
                "predict_intent"
            )
        ):

            pred = self.retriever.predict_intent(
                query
            )

            if isinstance(pred, dict):

                return (
                    pred.get(
                        "intent",
                        "other_unclear"
                    ),
                    float(
                        pred.get(
                            "confidence",
                            0.5
                        )
                    )
                )

            elif isinstance(pred, tuple):

                return (
                    pred[0],
                    float(
                        pred[1]
                    )
                    if len(pred) > 1
                    else 0.5
                )

            return (
                str(pred),
                0.5
            )

        return predict_intent(
            query,
            intent
        )

    # --------------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------------

    def retrieve_evidence(
        self,
        query: str,
        intent: Optional[str] = None,
        conversation_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        if self.retriever is None:
            return []

        try:

            results = self.retriever.retrieve(
                query,
                top_k=top_k,
                intent=intent,
                exclude_conversation_id=conversation_id,
            )

        except TypeError:

            results = self.retriever.retrieve(
                query,
                top_k=top_k
            )

        if results is None:
            return []

        if hasattr(
            results,
            "empty"
        ):

            if results.empty:
                return []

            candidates = (
                results.to_dict(
                    orient="records"
                )
            )

        else:

            if not results:
                return []

            candidates = list(
                results
            )

        if not candidates:
            return []

        return rerank_evidence(
            query,
            candidates
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    def generate(
        self,
        query: str,
        intent: Optional[str] = None,
        conversation_id: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # 1. Sensitive topic
        # ----------------------------------------------------

        sensitive, sensitive_reason = (
            check_sensitive_topic(
                query
            )
        )

        if sensitive:

            intent_name, confidence = (
                self._predict_intent_internal(
                    query,
                    intent
                )
            )

            result = build_grounded_reply(
                query,
                intent_name,
                None
            )

            result["intent"] = (
                intent_name
            )

            result["predicted_intent"] = (
                intent_name
            )

            result["intent_confidence"] = (
                confidence
            )

            result["domains"] = (
                detect_domains(query)
            )

            result["evidence_retrieved"] = 0

            result["evidence_candidates"] = 0

            result["sensitive"] = True

            result["sensitive_reason"] = (
                sensitive_reason
            )

            result["evidence"] = None

            result["evidence_used"] = False

            result["evidence_safe"] = False

            result["evidence_supported"] = False

            result["safe_to_answer"] = False

            result["escalate"] = True

            result["escalation_decision"] = (
                "ESCALATE"
            )

            result["escalation_reason"] = (
                sensitive_reason
            )

            return result

        # ----------------------------------------------------
        # 2. Resolved acknowledgement
        # ----------------------------------------------------

        if is_message_resolved(query):

            intent_name, confidence = (
                self._predict_intent_internal(
                    query,
                    intent
                )
            )

            result = build_grounded_reply(
                query,
                intent_name,
                None
            )

            result["intent"] = (
                intent_name
            )

            result["predicted_intent"] = (
                intent_name
            )

            result["intent_confidence"] = (
                confidence
            )

            result["domains"] = (
                detect_domains(query)
            )

            result["evidence_retrieved"] = 0

            result["evidence_candidates"] = 0

            result["sensitive"] = False

            result["evidence"] = None

            result["evidence_used"] = False

            result["evidence_safe"] = False

            result["evidence_supported"] = False

            result["safe_to_answer"] = True

            result["escalate"] = False

            result["escalation_decision"] = (
                "AUTO_HANDLE"
            )

            return result

        # ----------------------------------------------------
        # 3. Intent prediction
        # ----------------------------------------------------

        intent_name, confidence = (
            self._predict_intent_internal(
                query,
                intent
            )
        )

        # ----------------------------------------------------
        # 4. Retrieval
        # ----------------------------------------------------

        candidates = self.retrieve_evidence(
            query=query,
            intent=intent_name,
            conversation_id=conversation_id,
            top_k=top_k
        )

        # ----------------------------------------------------
        # 5. Accepted safe evidence
        # ----------------------------------------------------

        accepted = [
            c
            for c in candidates
            if c.get(
                "evidence_safe",
                False
            )
        ]

        # ----------------------------------------------------
        # IMPORTANT FIX:
        # Do not blindly take accepted[0].
        #
        # Select the strongest safe evidence according to
        # the calculated evidence scores.
        # ----------------------------------------------------

        if accepted:

            accepted.sort(
                key=lambda c: (
                    float(
                        c.get(
                            "final_evidence_score",
                            0.0
                        )
                    ),
                    float(
                        c.get(
                            "response_problem_similarity",
                            0.0
                        )
                    ),
                    float(
                        c.get(
                            "customer_problem_similarity",
                            0.0
                        )
                    ),
                    float(
                        c.get(
                            "evidence_score",
                            0.0
                        )
                    )
                ),
                reverse=True
            )

            evidence = accepted[0]

        else:

            evidence = None

        # ----------------------------------------------------
        # 6. Build reply
        # ----------------------------------------------------

        result = build_grounded_reply(
            query,
            intent_name,
            evidence
        )

        # ----------------------------------------------------
        # 7. Grounding verification
        # ----------------------------------------------------

        grounding = (
            perform_deterministic_grounding_check(
                query=query,
                reply=result["reply"],
                evidence=evidence
            )
        )

        result["grounded"] = (
            grounding["grounded"]
        )

        result["unsupported_claims"] = (
            grounding[
                "unsupported_claims"
            ]
        )

        # ----------------------------------------------------
        # 8. Grounding failure fallback
        # ----------------------------------------------------

        if not result["grounded"]:

            result["reply"] = (
                generate_safe_clarification(
                    query,
                    intent_name
                )
            )

            result["response_type"] = (
                "insufficient_evidence"
            )

            result["evidence_used"] = (
                False
            )

            result["evidence_safe"] = (
                False
            )

            result["evidence_supported"] = (
                False
            )

            result["evidence"] = None

            result["evidence_id"] = None

            result["safe_to_answer"] = (
                True
            )

            result["escalate"] = (
                False
            )

            result["escalation_decision"] = (
                "AUTO_HANDLE"
            )

        # ----------------------------------------------------
        # 9. Metadata
        # ----------------------------------------------------

        result["intent"] = (
            intent_name
        )

        result["predicted_intent"] = (
            intent_name
        )

        result["intent_confidence"] = (
            confidence
        )

        result["domains"] = (
            detect_domains(query)
        )

        result["evidence_retrieved"] = (
            len(candidates)
        )

        result["evidence_candidates"] = (
            len(accepted)
        )

        result["evidence_candidates_list"] = (
            accepted
        )

        result["sensitive"] = False

        result["sensitive_reason"] = ""

        result["escalation_decision"] = (
            "ESCALATE"
            if result.get(
                "escalate",
                False
            )
            else "AUTO_HANDLE"
        )

        # ----------------------------------------------------
        # Evidence metadata
        # ----------------------------------------------------

        if evidence:

            result["evidence"] = (
                evidence
            )

            result["evidence_id"] = (
                evidence.get(
                    "tweet_id",
                    evidence.get(
                        "response_id",
                        evidence.get(
                            "id",
                            ""
                        )
                    )
                )
            )

            result["selected_evidence_text"] = (
                evidence.get(
                    "response_text",
                    evidence.get(
                        "response",
                        evidence.get(
                            "historical_response",
                            ""
                        )
                    )
                )
            )

            result["evidence_relevance"] = float(
                evidence.get(
                    "evidence_relevance",
                    evidence.get(
                        "evidence_relevance_score",
                        evidence.get(
                            "similarity",
                            0.0
                        )
                    )
                )
            )

            result["evidence_quality"] = float(
                evidence.get(
                    "evidence_quality",
                    evidence.get(
                        "evidence_quality_score",
                        0.85
                    )
                )
            )

            result["problem_match_score"] = float(
                evidence.get(
                    "problem_match_score",
                    evidence.get(
                        "problem_match",
                        0.0
                    )
                )
            )

            result["customer_problem_similarity"] = float(
                evidence.get(
                    "customer_problem_similarity",
                    0.0
                )
            )

            result["response_problem_similarity"] = float(
                evidence.get(
                    "response_problem_similarity",
                    0.0
                )
            )

            result["domain_match"] = float(
                evidence.get(
                    "domain_match",
                    0.0
                )
            )

            result["action_match"] = float(
                evidence.get(
                    "action_match",
                    0.0
                )
            )

            result["conflict_detected"] = bool(
                evidence.get(
                    "conflict_detected",
                    False
                )
            )

            result["final_evidence_score"] = float(
                evidence.get(
                    "final_evidence_score",
                    result[
                        "evidence_relevance"
                    ]
                )
            )

            result["evidence_safe"] = True

            result["evidence_used"] = True

            result["evidence_supported"] = True

            result["safe_to_answer"] = (
                result["grounded"]
                and
                not bool(
                    result[
                        "unsupported_claims"
                    ]
                )
            )

            result["evidence_rejection_reason"] = ""

        else:

            result["evidence"] = None

            result["evidence_id"] = None

            result["selected_evidence_text"] = ""

            result["evidence_relevance"] = 0.0

            result["evidence_quality"] = 0.0

            result["problem_match_score"] = 0.0

            result["customer_problem_similarity"] = 0.0

            result["response_problem_similarity"] = 0.0

            result["domain_match"] = 0.0

            result["action_match"] = 0.0

            result["conflict_detected"] = False

            result["final_evidence_score"] = 0.0

            result["evidence_safe"] = False

            result["evidence_used"] = False

            result["evidence_supported"] = False

            result["safe_to_answer"] = True

            if candidates:

                result["evidence_rejection_reason"] = (
                    candidates[0].get(
                        "rejection_reason",
                        "no_safe_evidence"
                    )
                )

            else:

                result["evidence_rejection_reason"] = (
                    "no_evidence"
                )

        return result


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def generate_reply(
    query: str,
    retriever=None,
    intent: Optional[str] = None,
    conversation_id: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:

    generator = ReplyGenerator(
        retriever=retriever
    )

    return generator.generate(
        query=query,
        intent=intent,
        conversation_id=conversation_id,
        top_k=top_k
    )