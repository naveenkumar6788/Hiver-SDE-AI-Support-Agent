"""
src/retrieval/evidence_relevance.py

Hardened deterministic specific problem match and evidence relevance scoring module for AppleSupport customer support.
Evaluates whether a retrieved historical customer message + AppleSupport response
is genuinely useful, actionable, and problem-aligned for answering the current customer query.

Features:
1. Distinguishes Specific Problem Match from superficial Entity/Intent Match.
2. Extracts fine-grained problem types and failure modes.
3. Decouples query_domains, historical_customer_domains, and historical_response_domains.
4. Detects specific problem conflicts (e.g. inappropriate ads vs downloading, downgrade vs update, heating vs restore, photo loss vs account login).
5. Computes composite scoring:
   final_score = 0.30 * customer_problem_similarity + 0.30 * response_problem_similarity + 0.15 * intent_match + 0.15 * domain_match + 0.10 * action_match
6. Enforces hard safety gates to conservatively reject misaligned evidence.
"""

import re
from typing import Dict, List, Set, Tuple, Any, Optional

# ============================================================
# Configurable Weights, Penalties, and Thresholds
# ============================================================

# Relevance score weights (sum = 1.0)
RELEVANCE_WEIGHTS: Dict[str, float] = {
    "customer_problem_similarity": 0.30,
    "response_problem_similarity": 0.30,
    "intent_match": 0.15,
    "domain_match": 0.15,
    "action_match": 0.10,
}

# Explicit Penalties
PRODUCT_MISMATCH_PENALTY: float = 0.35
ISSUE_MISMATCH_PENALTY: float = 0.30
GENERIC_RESPONSE_PENALTY: float = 0.30

# Hard Gate Caps and Thresholds
MAX_GENERIC_RELEVANCE: float = 0.40
MIN_EVIDENCE_RELEVANCE: float = 0.55
MIN_EVIDENCE_QUALITY: float = 0.40
MIN_SPECIFIC_PROBLEM_MATCH: float = 0.40

# ============================================================
# Reusable Entity Extractor Taxonomy
# ============================================================

APP_ENTITIES: Dict[str, List[str]] = {
    "youtube": ["youtube"],
    "music": ["apple music", "applemusic", "music app", "itunes music", "music", "itunes"],
    "app_store": ["app store", "appstore"],
    "safari": ["safari", "browser"],
    "maps": ["maps", "apple maps", "google maps"],
    "mail": ["mail app", "email app", "mail", "email"],
    "messages": ["imessage", "messages app", "text message", "sms"],
    "camera": ["camera", "camera app", "photos app", "photo library", "photos", "pictures"],
    "calendar": ["calendar", "calendar app"],
    "notes": ["notes app", "notes"],
    "whatsapp": ["whatsapp"],
    "instagram": ["instagram"],
    "facebook": ["facebook"],
    "twitter": ["twitter"],
    "spotify": ["spotify"],
    "netflix": ["netflix"],
}

CONNECTIVITY_ENTITIES: Dict[str, List[str]] = {
    "wifi": ["wifi", "wi-fi", "wireless", "router", "hotspot", "wlan", "ssid"],
    "bluetooth": ["bluetooth", "airpods", "pair bluetooth"],
    "cellular": ["cellular", "lte", "4g", "3g", "carrier", "mobile data", "signal", "reception", "no service"],
    "sim": ["sim", "sim card", "esim", "sim tray", "sim drawer", "puk"],
    "calls": ["call", "calls", "calling", "dial", "voicemail", "phone call"]
}

HARDWARE_ENTITIES: Dict[str, List[str]] = {
    "battery": ["battery", "drain", "draining", "charge", "charging", "charger", "percentage", "battery life"],
    "speaker": ["speaker", "speakers", "sound", "volume", "audio", "mic", "microphone", "crackling", "earpiece", "headphones"],
    "screen": ["screen", "display", "touch", "touchscreen", "black screen", "flicker", "flickering", "oled", "lcd"],
    "apple_logo": ["apple logo", "boot loop", "bootloop", "stuck on logo", "flashing logo", "won't turn on", "wont turn on", "power on", "startup loop"],
    "keyboard": ["keyboard", "autocorrect", "typing", "keypad", "predictive"],
    "buttons": ["home button", "power button", "volume button", "side button", "sleep wake button", "vibrate switch"]
}

ACCOUNT_ENTITIES: Dict[str, List[str]] = {
    "icloud": ["icloud", "icloud drive", "cloud storage", "icloud backup"],
    "apple_id": ["apple id", "appleid", "account", "login", "password", "passcode", "iforgot", "two-factor", "2fa", "verification code"]
}

DEVICE_ENTITIES: Dict[str, List[str]] = {
    "iphone": [r"\biphone(\s*(7|8|x|xr|xs|11|12|13|14|15|6|6s|5|se|plus|pro|max))?\b"],
    "ipad": [r"\bipad(\s*(pro|mini|air))?\b"],
    "mac": [r"\b(mac|macbook|macbook pro|macbook air|imac|macos)\b"],
    "apple_watch": [r"\b(apple watch|iwatch|watch)\b"]
}

ALL_ENTITIES: Dict[str, List[str]] = {
    **APP_ENTITIES,
    **CONNECTIVITY_ENTITIES,
    **HARDWARE_ENTITIES,
    **ACCOUNT_ENTITIES
}

def extract_domain_entities(text: str) -> Set[str]:
    """Extract all recognized domain entity keys present in text."""
    if not text:
        return set()
    clean = re.sub(r"[^\w\s\-]", " ", text.lower())
    clean = f" {clean} "
    found = set()

    for ent, keywords in ALL_ENTITIES.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, clean):
                found.add(ent)
                break

    for dev, patterns in DEVICE_ENTITIES.items():
        for p in patterns:
            if re.search(p, clean):
                found.add(dev)
                break

    return found

# Explicit conflicting domain pairs (Query Domain vs Response Domain)
CONFLICTING_DOMAINS: Set[Tuple[str, str]] = {
    ("battery", "wifi"),
    ("wifi", "battery"),
    ("battery", "speaker_audio"),
    ("speaker_audio", "battery"),
    ("screen", "speaker_audio"),
    ("speaker_audio", "screen"),
    ("screen", "wifi"),
    ("wifi", "screen"),
    ("apple_id", "camera"),
    ("camera", "apple_id"),
    ("youtube", "apple_music"),
    ("apple_music", "youtube"),
    ("sim_hardware", "apple_logo_boot"),
    ("apple_logo_boot", "sim_hardware"),
    ("speaker_audio", "mac"),
    ("mac", "speaker_audio"),
    ("messages", "wifi"),
    ("wifi", "messages"),
    ("cellular", "wifi"),
    ("wifi", "cellular"),
    ("iphone", "mac"),
    ("mac", "iphone"),
}

# ============================================================
# Specific Problem Taxonomy & Extractor
# ============================================================

SPECIFIC_PROBLEM_PATTERNS: Dict[str, Dict[str, str]] = {
    "ads_in_apps": {
        "pattern": r"\b(inappropriate ads?|ads? in (kid'?s? )?games?|ad supported|block ads?|stop ads?|remove ads?|unwanted ads?|pop[- ]?ups?)\b",
        "category": "app_ads"
    },
    "ios_downgrade": {
        "pattern": r"\b(downgrade|revert back to ios|rollback|previous ios|return to ios|download the previous ios|go back to ios)\b",
        "category": "update_downgrade"
    },
    "screen_orientation": {
        "pattern": r"\b(horizontal (position|mode|screen)|landscape (mode|position|screen)|screen rotat(e|ion|ing)|auto[- ]rotate|orientation)\b",
        "category": "screen_display"
    },
    "device_heating": {
        "pattern": r"\b(heating (issue|problem)|overheat(ing)?|phone gets? (very )?hot|device (is )?hot|burning hot|thermal)\b",
        "category": "hardware_thermal"
    },
    "imessage_send_address": {
        "pattern": r"\b(messages? coming from (my )?email|imessage from email|sending from (my )?email|sent as email|messages? showing email)\b",
        "category": "messaging_address"
    },
    "photo_loss": {
        "pattern": r"\b(photos? (are )?(magically )?deleted|pictures? (are )?(magically )?deleted|all my photos? (are )?gone|lost all my (photos?|pictures?)|photos? disappeared|pictures? missing)\b",
        "category": "photo_library"
    },
    "port_connection_failure": {
        "pattern": r"\b(connections? to host.*failed|default ports? failed|port \d+ failed|connection refused|ssh.*failed|socket error)\b",
        "category": "network_port"
    },
    "battery_drain": {
        "pattern": r"\b(battery (is )?drain(ing)?|battery dies? (so )?(quickly|fast)|drains? (so )?fast|battery life (is )?poor|depleting fast|percentage drops?)\b",
        "category": "battery"
    },
    "battery_charging": {
        "pattern": r"\b(won'?t charge|not charging|charging slow(ly)?|charger|cable|charging port)\b",
        "category": "battery"
    },
    "app_download": {
        "pattern": r"\b(cannot download|can'?t download|won'?t download|cannot install|can'?t install|downloading those applications|app store download)\b",
        "category": "app_store"
    },
    "app_crash": {
        "pattern": r"\b(crash(es|ed|ing)?|keeps? crashing|force close|freez(e|es|ing)?|quits? unexpectedly)\b",
        "category": "app"
    },
    "account_locked": {
        "pattern": r"\b(account (is )?locked|locked out|disabled in app store|security lock|iforgot|reset password)\b",
        "category": "account"
    },
    "sim_failure": {
        "pattern": r"\b(invalid sim|no sim|sim card (not working|failure)|sim drawer won'?t open|sim tray)\b",
        "category": "sim"
    },
    "screen_black": {
        "pattern": r"\b(black screen|display black|screen went black|blank screen|won'?t turn on)\b",
        "category": "screen"
    },
    "screen_touch": {
        "pattern": r"\b(unresponsive|touch.*not working|ghost touch|touchscreen not responding)\b",
        "category": "screen"
    },
    "audio_speaker": {
        "pattern": r"\b(speaker (not working|broke|broken)|no sound|volume low|quiet|mic(rophone)? not working)\b",
        "category": "audio"
    },
    "wifi_connection": {
        "pattern": r"\b(cannot connect to wi[- ]?fi|can'?t connect.*wi[- ]?fi|wi[- ]?fi drops|wi[- ]?fi disconnecting|forget network)\b",
        "category": "wifi"
    },
    "cellular_calls": {
        "pattern": r"\b(cannot make calls?|can'?t make calls?|call failed|no service|no signal|carrier settings)\b",
        "category": "cellular"
    },
    "storage_full": {
        "pattern": r"\b(storage (is )?full|not enough storage|other storage|manage storage|storage space)\b",
        "category": "storage"
    },
    "autocorrect_keyboard": {
        "pattern": r"\b(autocorrect|keyboard typing|letter i glitch|predictive text|keyboard lag)\b",
        "category": "keyboard"
    },
    "apple_logo_boot": {
        "pattern": r"\b(apple logo|stuck on apple logo|boot loop|bootloop|flashing apple logo|reboot loop)\b",
        "category": "boot"
    },
    "device_restore": {
        "pattern": r"\b(restore in itunes|recovery mode|dfu mode|restore your iphone|restore in recovery)\b",
        "category": "restore"
    },
}

# Generic AppleSupport boilerplate patterns
GENERIC_BOILERPLATE_PATTERNS = [
    r"^we(?:'| a)?d (?:love|like|be happy|be glad) to help(?:\s+with this|\s+out)?[\.\!\?]?$",
    r"^(?:please\s+)?send us a dm(?:\s+so we can help|\s+to get started)?[\.\!\?]?$",
    r"^(?:please\s+)?dm us[\.\!\?]?$",
    r"^we('?| )?ve responded to your dm[\.\!\?]?$",
    r"^we('?| )?ve received your dm[\.\!\?]?$",
    r"^look for us there[\.\!\?]?$",
    r"^check out this link[\.\!\?]?$",
    r"^follow the steps here[\.\!\?]?$",
    r"^feel free to reach back out[\.\!\?]?$",
]

ACTION_WORDS = [
    "try", "check", "restart", "force restart", "reset", "remove",
    "install", "reinstall", "update", "turn on", "turn off", "toggle",
    "settings >", "settings", "delete", "restore", "back up", "sign out",
    "sign in", "unpair", "pair", "clean", "disconnect", "reconnect",
    "charge", "plug in", "inspect", "reseat", "insert", "follow these steps",
    "tap", "navigate to", "open", "clear",
]

DIAGNOSTIC_PATTERNS = [
    r"\bwhat (ios )?version\b",
    r"\bwhich (ios )?version\b",
    r"\bversion of ios\b",
    r"\bwhich model\b",
    r"\bwhat model\b",
    r"\bwhen did this (start|begin)\b",
    r"\bis this happening with (all|both|any)\b",
    r"\bare you seeing (an? |the )?(error|message|alert|code|screen|no service)\b",
    r"\bhave you (tried|tested|attempted)\b",
    r"\bdoes this (also )?happen (when|on|with)\b",
    r"\bare there any (app )?updates\b",
    r"\bare you downloading (updates|new)\b",
    r"\bcan you test if\b",
    r"\bare you able to\b",
    r"\bwhat happens when you\b",
]


# ============================================================
# 1. extract_problem_terms()
# ============================================================

def extract_problem_terms(text: str) -> Dict[str, Any]:
    """
    Extract specific problem types, keywords, and primary category from text.
    """
    if not text:
        return {
            "specific_problems": [],
            "categories": [],
            "primary_category": "unclear",
            "problem_terms": set(),
            "has_specific_problem": False,
        }

    clean = text.lower()
    detected_problems = []
    detected_cats = []
    problem_terms = set()

    for prob_name, prob_meta in SPECIFIC_PROBLEM_PATTERNS.items():
        if re.search(prob_meta["pattern"], clean):
            detected_problems.append(prob_name)
            detected_cats.append(prob_meta["category"])
            problem_terms.add(prob_name)

    primary_cat = detected_cats[0] if detected_cats else "unclear"

    return {
        "specific_problems": detected_problems,
        "categories": detected_cats,
        "primary_category": primary_cat,
        "problem_terms": problem_terms,
        "has_specific_problem": len(detected_problems) > 0,
    }


# ============================================================
# 2. detect_response_problem()
# ============================================================

def detect_response_problem(response: str) -> Dict[str, Any]:
    """
    Inspect ONLY the historical AppleSupport response to identify what
    specific problem it is providing guidance or troubleshooting for.
    """
    if not response:
        return {
            "addressed_problems": [],
            "addressed_categories": [],
            "actionable": False,
            "has_diagnostic": False,
            "is_generic": True,
            "action_terms": set(),
        }

    clean = response.lower()

    # Check boilerplate
    clean_lines = [line.strip() for line in clean.splitlines() if line.strip()]
    is_generic = False
    for line in clean_lines:
        line_clean = re.sub(r"@\w+", "", line).strip()
        if any(re.search(bp, line_clean) for bp in GENERIC_BOILERPLATE_PATTERNS):
            is_generic = True
            break

    # Extract actions
    actions_found = set()
    for word in ACTION_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", clean):
            actions_found.add(word)

    has_action = len(actions_found) > 0
    has_diag = any(re.search(p, clean) for p in DIAGNOSTIC_PATTERNS)
    has_link = bool(re.search(r"https?://\S+", response)) or "support.apple.com" in clean or "apple.co" in clean

    # Detect addressed problems in response
    addressed_problems = []
    addressed_cats = []
    for prob_name, prob_meta in SPECIFIC_PROBLEM_PATTERNS.items():
        if re.search(prob_meta["pattern"], clean):
            addressed_problems.append(prob_name)
            addressed_cats.append(prob_meta["category"])

    # Specific response domain signals
    if re.search(r"\bbattery usage|low power mode|draining\b", clean):
        if "battery_drain" not in addressed_problems:
            addressed_problems.append("battery_drain")
            addressed_cats.append("battery")

    if re.search(r"\bdelete the update|download the (software )?update again\b", clean):
        if "update_install" not in addressed_problems:
            addressed_problems.append("update_install")
            addressed_cats.append("update")

    if re.search(r"\bforget this network|wi[- ]?fi\b", clean) and "settings > wi-fi" in clean:
        if "wifi_connection" not in addressed_problems:
            addressed_problems.append("wifi_connection")
            addressed_cats.append("wifi")

    if re.search(r"\bremove the sim|sim card\b", clean):
        if "sim_failure" not in addressed_problems:
            addressed_problems.append("sim_failure")
            addressed_cats.append("sim")

    return {
        "addressed_problems": addressed_problems,
        "addressed_categories": addressed_cats,
        "actionable": has_action or has_link,
        "has_diagnostic": has_diag,
        "is_generic": is_generic and not has_action and not has_link,
        "action_terms": actions_found,
    }


# ============================================================
# 3. calculate_customer_problem_similarity()
# ============================================================

def calculate_customer_problem_similarity(
    query: str,
    hist_customer: str,
    query_problems: Optional[Dict[str, Any]] = None,
    hist_problems: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """
    Evaluate similarity between current customer problem and historical customer problem.
    """
    if query_problems is None:
        query_problems = extract_problem_terms(query)
    if hist_problems is None:
        hist_problems = extract_problem_terms(hist_customer)

    q_probs = set(query_problems["specific_problems"])
    h_probs = set(hist_problems["specific_problems"])

    q_cats = set(query_problems["categories"])
    h_cats = set(hist_problems["categories"])

    # Direct specific problem match
    shared_probs = q_probs & h_probs
    if shared_probs:
        prob_str = ", ".join(sorted(shared_probs))
        return 0.95, f"exact_problem_match: {prob_str}"

    # Same category match
    shared_cats = q_cats & h_cats
    if shared_cats:
        cat_str = ", ".join(sorted(shared_cats))
        return 0.75, f"same_category_match: {cat_str}"

    # Word overlap on key technical nouns
    q_words = set(re.findall(r"\b[a-z]{3,}\b", query.lower())) - {"the", "and", "with", "this", "that", "have", "from", "for", "phone", "iphone", "apple", "applesupport"}
    h_words = set(re.findall(r"\b[a-z]{3,}\b", hist_customer.lower())) - {"the", "and", "with", "this", "that", "have", "from", "for", "phone", "iphone", "apple", "applesupport"}
    word_overlap = q_words & h_words

    if len(word_overlap) >= 3:
        return 0.65, f"technical_keyword_overlap ({len(word_overlap)} terms)"
    elif len(word_overlap) >= 1:
        return 0.40, f"weak_keyword_overlap ({len(word_overlap)} terms)"

    return 0.15, "no_problem_overlap"


# ============================================================
# 4. calculate_response_problem_similarity()
# ============================================================

def calculate_response_problem_similarity(
    query: str,
    hist_response: str,
    query_problems: Optional[Dict[str, Any]] = None,
    resp_problems: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """
    Evaluate whether the historical response actually addresses the current query's specific problem.
    """
    if query_problems is None:
        query_problems = extract_problem_terms(query)
    if resp_problems is None:
        resp_problems = detect_response_problem(hist_response)

    q_probs = set(query_problems["specific_problems"])
    r_probs = set(resp_problems["addressed_problems"])

    q_cats = set(query_problems["categories"])
    r_cats = set(resp_problems["addressed_categories"])

    # Direct problem resolution
    shared_probs = q_probs & r_probs
    if shared_probs:
        prob_str = ", ".join(sorted(shared_probs))
        return 0.95, f"response_resolves_problem: {prob_str}"

    # Category alignment with actionable steps
    shared_cats = q_cats & r_cats
    if shared_cats and resp_problems["actionable"]:
        cat_str = ", ".join(sorted(shared_cats))
        return 0.80, f"response_actionable_in_category: {cat_str}"
    elif shared_cats and resp_problems["has_diagnostic"]:
        cat_str = ", ".join(sorted(shared_cats))
        return 0.65, f"response_diagnostic_in_category: {cat_str}"

    # Check if response has useful general troubleshooting while query has no detected specific problem
    if not q_probs and resp_problems["actionable"]:
        return 0.50, "general_actionable_response"
    elif not q_probs and resp_problems["has_diagnostic"]:
        return 0.45, "general_diagnostic_response"

    # Mismatch
    if q_probs and r_probs and not (q_probs & r_probs):
        q_p = list(q_probs)[0]
        r_p = list(r_probs)[0]
        return 0.15, f"response_addresses_different_problem: {q_p} vs {r_p}"

    return 0.20, "unaligned_response"


# ============================================================
# 5. calculate_domain_match()
# ============================================================

def calculate_domain_match(
    query_domains: List[str],
    resp_domains: List[str],
) -> Tuple[float, str]:
    """
    Decoupled domain match: strictly inspects query_domains vs resp_domains.
    """
    if not query_domains or not resp_domains:
        return 0.50, "neutral_domain"

    shared = set(query_domains) & set(resp_domains)
    if shared:
        dom_str = ", ".join(sorted(shared))
        return 1.0, f"domain_match: {dom_str}"

    # Check explicit conflicts
    for q_dom in query_domains:
        for r_dom in resp_domains:
            if (q_dom, r_dom) in CONFLICTING_DOMAINS:
                return 0.0, f"domain_conflict: {q_dom} vs {r_dom}"

    return 0.20, "disjoint_domains"


# ============================================================
# 6. calculate_action_match()
# ============================================================

def calculate_action_match(
    query_problems: Dict[str, Any],
    hist_response: str,
) -> Tuple[float, str]:
    """
    Check if actions in the historical response are appropriate for the customer's problem.
    """
    resp_meta = detect_response_problem(hist_response)

    if not resp_meta["actionable"] and not resp_meta["has_diagnostic"]:
        return 0.0, "no_actionable_or_diagnostic_content"

    q_probs = query_problems.get("specific_problems", [])

    # Action compatibility rules
    if "battery_drain" in q_probs:
        if any(w in hist_response.lower() for w in ["battery usage", "background app refresh", "low power mode", "settings > battery"]):
            return 1.0, "targeted_battery_action"
        elif resp_meta["actionable"]:
            return 0.60, "general_action"

    if "wifi_connection" in q_probs:
        if any(w in hist_response.lower() for w in ["forget this network", "settings > wi-fi", "restart your router", "reset network settings"]):
            return 1.0, "targeted_wifi_action"

    if "sim_failure" in q_probs:
        if any(w in hist_response.lower() for w in ["remove", "paperclip", "tray", "reseat", "carrier settings"]):
            return 1.0, "targeted_sim_action"

    if resp_meta["actionable"]:
        return 0.75, "actionable_instructions_present"
    elif resp_meta["has_diagnostic"]:
        return 0.50, "diagnostic_inquiry_present"

    return 0.20, "low_action_match"


# ============================================================
# 7. detect_problem_conflict()
# ============================================================

def detect_problem_conflict(
    query: str,
    query_problems: Dict[str, Any],
    hist_customer: str,
    hist_response: str,
) -> Tuple[bool, str]:
    """
    Detect explicit specific problem conflicts between query and evidence.
    """
    q_lower = query.lower()
    r_lower = hist_response.lower()
    q_probs = query_problems.get("specific_problems", [])

    # Case 1: Inappropriate ads in games vs app download troubleshooting
    if "ads_in_apps" in q_probs or ("ad supported" in q_lower or "inappropriate ads" in q_lower):
        if any(w in r_lower for w in ["downloading those applications", "app store download", "restarting your device yet"]):
            return True, "problem_conflict: ads_in_apps vs app_downloading"

    # Case 2: iOS Downgrade request vs future update / generic guidance
    if "ios_downgrade" in q_probs or ("previous ios" in q_lower or "revert back to ios" in q_lower or "download the previous ios" in q_lower):
        if any(w in r_lower for w in ["future software update", "work around the issue until", "update to ios", "latest ios", "update to the latest", "an update has been released", "we've released an update", "release an update"]):
            return True, "problem_conflict: ios_downgrade vs update_guidance"
        if any(w in r_lower for w in ["update", "restore", "itunes"]):
            return True, "problem_conflict: ios_downgrade vs update_steps"

    # Case 3: Screen orientation / horizontal mode vs app update
    if "screen_orientation" in q_probs or ("horizontal position" in q_lower or "landscape" in q_lower):
        if any(w in r_lower for w in ["check for an update in the app store", "whatsapp", "update the app"]):
            return True, "problem_conflict: screen_orientation vs app_store_update"

    # Case 4: Heating issue vs screen / setup / restore
    if "device_heating" in q_probs or ("heating issue" in q_lower or "overheating" in q_lower):
        if any(w in r_lower for w in ["screen is dark", "setup for the first time", "restore in recovery"]):
            return True, "problem_conflict: device_heating vs device_restore"

    # Case 5: Messages coming from email vs iOS update
    if "imessage_send_address" in q_probs or ("messages coming from my email" in q_lower):
        if any(w in r_lower for w in ["software update", "ios 11.1"]):
            return True, "problem_conflict: imessage_send_address vs ios_update"

    # Case 6: Photo loss vs generic account login
    if "photo_loss" in q_probs or ("pictures on my phone magically deleted" in q_lower or "photos deleted" in q_lower):
        if any(w in r_lower for w in ["log in at", "manage your apple id", "iforgot"]):
            return True, "problem_conflict: photo_loss vs generic_account_login"

    # Case 7: Server port connection failure vs generic Wi-Fi security
    if "port_connection_failure" in q_probs or ("ports failed" in q_lower or "connections to host" in q_lower):
        if any(w in r_lower for w in ["security settings", "wpa2", "wifi password"]):
            return True, "problem_conflict: server_port_connection vs wifi_security"

    # Case 8: Autocorrect / keyboard typing vs Wi-Fi / Bluetooth
    if "autocorrect_keyboard" in q_probs or ("autocorrect" in q_lower or "keyboard" in q_lower):
        if any(w in r_lower for w in ["turn off wi-fi", "bluetooth disconnect", "control center"]):
            return True, "problem_conflict: autocorrect_vs_wifi_bluetooth"

    # Case 9: Battery drain vs asking ONLY for iOS version without battery steps
    if "battery_drain" in q_probs:
        if ("which ios version" in r_lower or "what ios version" in r_lower) and not any(w in r_lower for w in ["battery", "drain", "charge", "settings"]):
            return True, "problem_conflict: battery_drain vs pure_ios_version_inquiry"

    # Case 10: SIM failure vs recovery mode / power on
    if "sim_failure" in q_probs:
        if any(w in r_lower for w in ["won't power on", "recovery mode", "restore your device"]):
            return True, "problem_conflict: sim_failure vs power_on_recovery"

    # Case 11: Headphones / audio hardware vs macOS
    if any(w in q_lower for w in ["headphones", "earbuds", "airpods", "earphones", "headset"]):
        if any(w in r_lower for w in ["macos", "mac book", "macbook", "imac"]):
            return True, "problem_conflict: headphones vs macos"

    # Case 12: MMS / SMS messaging vs Wi-Fi
    if any(w in q_lower for w in ["mms", "sms", "text message", "send an mms"]):
        if any(w in r_lower for w in ["wi-fi", "wifi", "router", "network settings"]) and not any(w in r_lower for w in ["mms", "sms", "cellular", "carrier"]):
            return True, "problem_conflict: mms_sms vs wifi"

    # Case 13: Carrier unlocking vs physical SIM tray or SIM card size
    if any(w in q_lower for w in ["unlocking", "carrier unlock", "unlock my iphone", "unlock phone"]):
        if any(w in r_lower for w in ["sized sim card", "sim card size", "sim tray", "paperclip", "sim drawer"]):
            return True, "problem_conflict: carrier_unlock vs physical_sim_tray_or_size"

    # Case 14: App Store password settings vs download/Wi-Fi error
    if any(w in q_lower for w in ["require a password", "password for already purchased", "password for any new apps", "password settings", "free apps password"]):
        if any(w in r_lower for w in ["different wi-fi", "wi-fi network", "downloading those applications", "app store working again"]):
            return True, "problem_conflict: app_store_passwords vs download_network"

    # Case 15: USB car music playback vs restart
    if "usb" in q_lower and any(w in q_lower for w in ["song starts", "song", "music", "car", "listening to"]):
        if any(w in r_lower for w in ["restart your computer", "restarted your iphone", "restart your device", "have you restarted"]):
            return True, "problem_conflict: usb_car_playback vs restart"

    # Case 16: Battery health inquiry vs generic battery performance update
    if "battery health" in q_lower:
        if ("article and send us a dm" in r_lower or "performance" in r_lower) and not any(w in r_lower for w in ["battery health", "maximum capacity", "service", "settings > battery"]):
            return True, "problem_conflict: battery_health vs generic_battery_performance"

    return False, ""


# ============================================================
# 8. calculate_final_evidence_score()
# ============================================================

def calculate_final_evidence_score(
    query: str,
    historical_customer_message: str,
    historical_response: str,
    current_intent: str,
    historical_intent: str,
    query_domains: Optional[List[str]] = None,
    resp_domains: Optional[List[str]] = None,
    semantic_similarity: float = 0.0,
    evidence_quality_score: float = 0.50,
    actionable: bool = False,
    generic_response: bool = False,
) -> Dict[str, Any]:
    """
    Compute fine-grained multi-signal evidence score with hard safety gating.
    """
    # 1. Extract problem terms
    q_problems = extract_problem_terms(query)
    h_problems = extract_problem_terms(historical_customer_message)
    r_problems = detect_response_problem(historical_response)

    # 2. Compute component signals
    cust_prob_sim, cust_prob_reason = calculate_customer_problem_similarity(
        query, historical_customer_message, q_problems, h_problems
    )
    resp_prob_sim, resp_prob_reason = calculate_response_problem_similarity(
        query, historical_response, q_problems, r_problems
    )

    intent_match = bool(current_intent and historical_intent and current_intent == historical_intent)
    intent_match_score = 1.0 if intent_match else (0.50 if (current_intent == "other_unclear" or historical_intent == "other_unclear") else 0.10)

    if query_domains is None:
        q_ents = list(extract_domain_entities(query))
        query_domains = q_ents if q_ents else list(q_problems["categories"])
    if resp_domains is None:
        r_ents = list(extract_domain_entities(historical_response))
        resp_domains = r_ents if r_ents else list(r_problems["addressed_categories"])

    domain_match_score, domain_match_reason = calculate_domain_match(query_domains, resp_domains)
    action_match_score, action_match_reason = calculate_action_match(q_problems, historical_response)

    # 3. Detect problem conflicts
    conflict_detected, conflict_reason = detect_problem_conflict(
        query, q_problems, historical_customer_message, historical_response
    )

    # 4. Compute conceptual scoring formula
    # final_score = 0.30*cust_prob + 0.30*resp_prob + 0.15*intent + 0.15*domain + 0.10*action
    raw_final_score = (
        RELEVANCE_WEIGHTS["customer_problem_similarity"] * cust_prob_sim +
        RELEVANCE_WEIGHTS["response_problem_similarity"] * resp_prob_sim +
        RELEVANCE_WEIGHTS["intent_match"] * intent_match_score +
        RELEVANCE_WEIGHTS["domain_match"] * domain_match_score +
        RELEVANCE_WEIGHTS["action_match"] * action_match_score
    )

    # Penalties
    total_penalty = 0.0
    if generic_response or r_problems["is_generic"]:
        total_penalty += GENERIC_RESPONSE_PENALTY

    if conflict_detected:
        total_penalty += PRODUCT_MISMATCH_PENALTY

    final_score = max(0.0, min(1.0, raw_final_score - total_penalty))

    # 5. Hard Rejection Gates
    hard_reject = False
    rejection_reason = ""

    if conflict_detected:
        hard_reject = True
        rejection_reason = conflict_reason
    elif domain_match_score == 0.0:
        hard_reject = True
        rejection_reason = "explicit_domain_conflict"
    elif r_problems["is_generic"] and not r_problems["actionable"]:
        hard_reject = True
        rejection_reason = "generic_response_without_actionable_guidance"
    elif q_problems["has_specific_problem"] and resp_prob_sim < MIN_SPECIFIC_PROBLEM_MATCH:
        hard_reject = True
        rejection_reason = "response_problem_mismatch"
    elif not r_problems["actionable"] and not r_problems["has_diagnostic"]:
        hard_reject = True
        rejection_reason = "no_useful_guidance"
    elif final_score < MIN_EVIDENCE_RELEVANCE:
        hard_reject = True
        rejection_reason = "low_evidence_relevance"

    safe_and_accepted = (not hard_reject) and (final_score >= MIN_EVIDENCE_RELEVANCE)

    return {
        "final_evidence_score": round(final_score, 4),
        "customer_problem_similarity": round(cust_prob_sim, 4),
        "customer_problem_reason": cust_prob_reason,
        "response_problem_similarity": round(resp_prob_sim, 4),
        "response_problem_reason": resp_prob_reason,
        "intent_match": intent_match,
        "intent_match_score": round(intent_match_score, 4),
        "domain_match": round(domain_match_score, 4),
        "domain_match_reason": domain_match_reason,
        "action_match": round(action_match_score, 4),
        "action_match_reason": action_match_reason,
        "conflict_detected": conflict_detected,
        "conflict_reason": conflict_reason,
        "generic_response": r_problems["is_generic"] or generic_response,
        "evidence_safe": safe_and_accepted,
        "evidence_rejection_reason": rejection_reason if hard_reject else "accepted: problem_aligned_evidence",
    }


# ============================================================
# 9. is_specific_evidence_safe()
# ============================================================

def is_specific_evidence_safe(
    query: str,
    evidence_customer: str,
    evidence_response: str,
    current_intent: Optional[str] = None,
    historical_intent: Optional[str] = None,
    query_domains: Optional[List[str]] = None,
    resp_domains: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Hard safety gate verifying that evidence addresses the specific customer problem.
    """
    eval_result = calculate_final_evidence_score(
        query=query,
        historical_customer_message=evidence_customer,
        historical_response=evidence_response,
        current_intent=current_intent or "",
        historical_intent=historical_intent or "",
        query_domains=query_domains,
        resp_domains=resp_domains,
    )
    return eval_result["evidence_safe"], eval_result["evidence_rejection_reason"]


# ============================================================
# 10. calculate_evidence_relevance() (Compatibility Wrapper)
# ============================================================

def calculate_evidence_relevance(
    query: str,
    historical_customer_message: str,
    historical_response: str,
    current_intent: str,
    historical_intent: str,
    semantic_similarity: float = 0.0,
    evidence_quality_score: float = 0.50,
    actionable: bool = False,
    generic_response: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes evidence relevance preserving backward compatibility with retriever and reply generator,
    augmented with specific problem match signals.
    """
    score_info = calculate_final_evidence_score(
        query=query,
        historical_customer_message=historical_customer_message,
        historical_response=historical_response,
        current_intent=current_intent,
        historical_intent=historical_intent,
        semantic_similarity=semantic_similarity,
        evidence_quality_score=evidence_quality_score,
        actionable=actionable,
        generic_response=generic_response,
    )

    relevance_accepted = score_info["evidence_safe"]
    evidence_score = score_info["final_evidence_score"]
    problem_match_score = (score_info["customer_problem_similarity"] + score_info["response_problem_similarity"]) / 2.0

    return {
        "evidence_relevance_score": evidence_score,
        "relevance_reason": score_info["evidence_rejection_reason"],
        "problem_match_score": round(problem_match_score, 4),
        "problem_match_reason": score_info["response_problem_reason"],
        "entity_overlap": [],
        "entity_mismatch": score_info["conflict_detected"],
        "issue_overlap": [],
        "issue_mismatch": score_info["conflict_detected"] or (score_info["response_problem_similarity"] < 0.35),
        "usable_guidance": not score_info["generic_response"],
        "relevance_accepted": relevance_accepted,
        "evidence_safe": relevance_accepted,
        # Diagnostic fields
        "customer_problem_similarity": score_info["customer_problem_similarity"],
        "response_problem_similarity": score_info["response_problem_similarity"],
        "intent_match": score_info["intent_match"],
        "domain_match": score_info["domain_match"],
        "action_match": score_info["action_match"],
        "conflict_detected": score_info["conflict_detected"],
        "generic_response": score_info["generic_response"],
        "final_evidence_score": score_info["final_evidence_score"],
        "evidence_rejection_reason": score_info["evidence_rejection_reason"],
    }
