import pandas as pd
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

GOLDEN_FILE = Path("golden_set/final_golden.csv")
THREAD_FILE = Path("data/processed/applesupport_threads.csv")

OUTPUT_FILE = Path("golden_set/bulk_context_labeled.csv")
UNCERTAIN_FILE = Path("golden_set/bulk_context_uncertain.csv")


# ============================================================
# INTENTS
# ============================================================

INTENTS = {
    1: "battery_charging",
    2: "ios_software_update",
    3: "wifi_connectivity",
    4: "app_problems",
    5: "app_store_downloads",
    6: "apple_music_itunes",
    7: "apple_id_icloud",
    8: "screen_display",
    9: "calls_cellular",
    10: "audio_speaker",
    11: "device_hardware",
    12: "other_unclear",
}


# ============================================================
# KEYWORDS
# ============================================================

RULES = {

    "battery_charging": [
        "battery",
        "battery drain",
        "battery life",
        "battery dying",
        "charging",
        "charge",
        "won't charge",
        "not charging",
        "battery percentage",
        "battery percent",
        "power drain",
    ],

    "ios_software_update": [
        "ios update",
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17",
        "ios 18",
        "ios 19",
        "software update",
        "software upgrade",
        "update my iphone",
        "update my ipad",
        "updating",
        "update failed",
        "update error",
        "update taking",
        "update took",
        "download update",
        "install update",
        "after update",
        "since update",
        "updated to",
        "downgrade ios",
        "beta",
        "gm update",
        "autocorrect",
        "typing i",
        "question mark",
    ],

    "wifi_connectivity": [
        "wifi",
        "wi-fi",
        "wireless",
        "internet connection",
        "connect to wifi",
        "connected to wifi",
        "wifi connection",
        "wifi issue",
        "wifi issues",
        "wifi not",
        "router",
        "network connection",
    ],

    "app_problems": [
        "app crashes",
        "app crash",
        "apps crash",
        "app crashing",
        "apps crashing",
        "app doesn't work",
        "app not working",
        "app won't work",
        "application",
        "crashing app",
        "app freezes",
        "app freezing",
        "video won't",
        "videos won't",
        "facebook",
        "twitter",
        "instagram",
        "whatsapp",
    ],

    "app_store_downloads": [
        "app store",
        "appstore",
        "download apps",
        "download an app",
        "install apps",
        "install an app",
        "can't download app",
        "cannot download app",
        "apps won't download",
        "apps not downloading",
    ],

    "apple_music_itunes": [
        "apple music",
        "itunes",
        "music app",
        "listening to music",
        "listen to music",
        "music stops",
        "music stop",
        "music cuts",
        "music playback",
        "songs",
        "song stops",
    ],

    "apple_id_icloud": [
        "apple id",
        "icloud",
        "i cloud",
        "icloud account",
        "apple account",
        "password",
        "verification code",
        "two factor",
        "2fa",
        "trusted device",
        "sign in",
        "login",
        "log in",
        "account verification",
        "phishing",
    ],

    "screen_display": [
        "screen",
        "display",
        "touch screen",
        "touchscreen",
        "brightness",
        "pixels",
        "pixel",
        "question marks",
        "screen position",
        "horizontal position",
        "assistive touch",
    ],

    "calls_cellular": [
        "cellular",
        "cell signal",
        "cell connection",
        "no signal",
        "poor signal",
        "weak signal",
        "mobile data",
        "phone calls",
        "make calls",
        "receive calls",
        "outbound calls",
        "can't call",
        "cannot call",
        "calling",
        "call dropped",
        "call drops",
    ],

    "audio_speaker": [
        "speaker",
        "speakers",
        "airpods",
        "airpod",
        "headphones",
        "headphone",
        "bluetooth audio",
        "audio",
        "sound",
        "no sound",
        "sound cuts",
        "audio cuts",
        "audio skips",
    ],

    "device_hardware": [
        "hardware",
        "iphone broken",
        "ipad broken",
        "phone broken",
        "device broken",
        "button doesn't work",
        "button not working",
        "power button",
        "home button",
        "overheating",
        "heating",
        "phone gets hot",
        "graphics card",
        "charger",
        "charging port",
        "physical damage",
        "screen broken",
    ],
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize common variations
    text = text.replace("wi-fi", "wifi")
    text = text.replace("i-phone", "iphone")
    text = text.replace("appstore", "app store")

    return text


# ============================================================
# GET CONVERSATION CONTEXT
# ============================================================

def build_context_map(threads):

    print("Building conversation context...")

    threads["tweet_id"] = threads["tweet_id"].astype(str)
    threads["conversation_id"] = threads["conversation_id"].astype(str)

    context_map = {}

    grouped = threads.groupby("conversation_id", sort=False)

    for conversation_id, group in grouped:

        # Sort chronologically if possible
        if "created_at" in group.columns:
            group = group.sort_values("created_at")

        messages = []

        for _, row in group.iterrows():

            text = row.get("text", "")

            if pd.isna(text):
                continue

            author = str(row.get("author_id", ""))

            # AppleSupport rows are support messages.
            # Everything else is treated as customer context.
            is_support = (
                str(row.get("inbound", "")).lower() == "false"
                or "applesupport" in author.lower()
            )

            role = "SUPPORT" if is_support else "CUSTOMER"

            messages.append(
                f"{role}: {text}"
            )

        context_map[conversation_id] = "\n".join(messages)

    print(f"Conversation contexts built: {len(context_map)}")

    return context_map


# ============================================================
# RULE SCORING
# ============================================================

def score_intents(text, context):

    target = clean_text(text)
    full_context = clean_text(context)

    # Give target message higher weight.
    combined = target + " " + target + " " + full_context

    scores = {}

    for intent, keywords in RULES.items():

        score = 0

        for keyword in keywords:

            keyword = keyword.lower()

            # Stronger weight for target
            if keyword in target:
                score += 5

            # Context evidence
            if keyword in full_context:
                score += 1

        scores[intent] = score

    return scores


# ============================================================
# SPECIAL CONTEXT RULES
# ============================================================

def context_override(text, context):

    target = clean_text(text)
    ctx = clean_text(context)

    # --------------------------------------------------------
    # Cellular vs WiFi
    # --------------------------------------------------------

    if (
        ("cell signal" in ctx)
        or ("poor cell connection" in ctx)
        or ("no signal" in ctx)
        or ("weak cell" in ctx)
    ):
        if "connection" in target or "network" in target:
            return "calls_cellular"

    # --------------------------------------------------------
    # Verification / trusted device
    # --------------------------------------------------------

    if (
        "verification code" in target
        or "trusted device" in target
        or "two factor" in target
        or "2fa" in target
    ):
        return "apple_id_icloud"

    # --------------------------------------------------------
    # Music
    # --------------------------------------------------------

    if (
        "music" in target
        or "itunes" in target
        or "airpods" in target
    ):
        if (
            "music" in target
            or "itunes" in target
            or "airpods" in target
        ):
            return "apple_music_itunes"

    # --------------------------------------------------------
    # Explicit WiFi
    # --------------------------------------------------------

    if "wifi" in target or "wi-fi" in target:
        return "wifi_connectivity"

    # --------------------------------------------------------
    # Explicit cellular
    # --------------------------------------------------------

    if (
        "cellular" in target
        or "cell signal" in target
        or "mobile data" in target
        or "no signal" in target
    ):
        return "calls_cellular"

    # --------------------------------------------------------
    # Explicit iOS update
    # --------------------------------------------------------

    update_terms = [
        "ios update",
        "software update",
        "download update",
        "install update",
        "updated to",
        "after update",
        "since update",
        "update failed",
        "update error",
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17",
        "ios 18",
        "ios 19",
        "downgrade ios",
        "beta",
    ]

    if any(term in target for term in update_terms):
        return "ios_software_update"

    # --------------------------------------------------------
    # Battery
    # --------------------------------------------------------

    if "battery" in target or "charging" in target:
        return "battery_charging"

    # --------------------------------------------------------
    # Apple ID
    # --------------------------------------------------------

    if (
        "apple id" in target
        or "icloud" in target
        or "verification code" in target
        or "trusted device" in target
        or "login" in target
        or "sign in" in target
    ):
        return "apple_id_icloud"

    # --------------------------------------------------------
    # App Store
    # --------------------------------------------------------

    if "app store" in target:

        # If specifically about iOS update, update wins
        if (
            "update" in target
            or "ios" in target
            or "software" in target
        ):
            return "ios_software_update"

        return "app_store_downloads"

    # --------------------------------------------------------
    # Screen
    # --------------------------------------------------------

    if (
        "screen" in target
        or "display" in target
        or "assistive touch" in target
    ):
        return "screen_display"

    # --------------------------------------------------------
    # Calls
    # --------------------------------------------------------

    if (
        "call" in target
        or "calls" in target
        or "cellular" in target
        or "signal" in target
    ):
        return "calls_cellular"

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    if (
        "speaker" in target
        or "headphone" in target
        or "airpods" in target
        or "audio" in target
        or "sound" in target
    ):
        return "audio_speaker"

    # --------------------------------------------------------
    # Hardware
    # --------------------------------------------------------

    if (
        "hardware" in target
        or "broken" in target
        or "overheating" in target
        or "heating" in target
        or "charger" in target
        or "button" in target
    ):
        return "device_hardware"

    return None


# ============================================================
# CLASSIFY ONE ROW
# ============================================================

def classify_row(row, context_map):

    tweet_id = str(row["tweet_id"])
    conversation_id = str(row["conversation_id"])

    target = row.get("text", "")

    context = context_map.get(conversation_id, "")

    # First use explicit/context rules
    override = context_override(target, context)

    if override:
        return override, "high"

    # Otherwise use scoring
    scores = score_intents(target, context)

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_intent, best_score = ranked[0]

    second_score = ranked[1][1]

    # Nothing meaningful found
    if best_score == 0:
        return "other_unclear", "low"

    # Very close competition
    if best_score - second_score <= 1:
        return best_intent, "medium"

    return best_intent, "high"


# ============================================================
# MAIN
# ============================================================

print("\nLoading golden set...")

golden = pd.read_csv(GOLDEN_FILE)

print(f"Golden rows: {len(golden)}")


print("\nLoading conversation threads...")

threads = pd.read_csv(THREAD_FILE)

print(f"Thread rows: {len(threads)}")


# ------------------------------------------------------------
# Ensure columns exist
# ------------------------------------------------------------

for column in [
    "human_intent",
    "human_notes",
    "label_status"
]:

    if column not in golden.columns:
        golden[column] = ""

    golden[column] = (
        golden[column]
        .fillna("")
        .astype(str)
    )


# ------------------------------------------------------------
# Context map
# ------------------------------------------------------------

context_map = build_context_map(threads)


# ------------------------------------------------------------
# Preserve existing human labels
# ------------------------------------------------------------

reviewed_mask = golden["label_status"].astype(str).str.lower().isin([
    "reviewed",
    "human_reviewed"
])

print(
    f"\nAlready human-reviewed: "
    f"{reviewed_mask.sum()}"
)


# ------------------------------------------------------------
# Bulk classification
# ------------------------------------------------------------

golden["bulk_intent"] = ""
golden["bulk_confidence"] = ""
golden["bulk_reason"] = ""


for index, row in golden.iterrows():

    # NEVER overwrite human labels
    if reviewed_mask.iloc[index]:

        golden.loc[index, "bulk_intent"] = (
            golden.loc[index, "human_intent"]
        )

        golden.loc[index, "bulk_confidence"] = "human"

        golden.loc[index, "bulk_reason"] = (
            "Existing human-reviewed label preserved."
        )

        continue

    intent, confidence = classify_row(
        row,
        context_map
    )

    golden.loc[index, "bulk_intent"] = intent
    golden.loc[index, "bulk_confidence"] = confidence

    golden.loc[index, "bulk_reason"] = (
        f"Automatically assigned using target message "
        f"and conversation context; confidence={confidence}."
    )


# ------------------------------------------------------------
# Status
# ------------------------------------------------------------

golden["bulk_label_status"] = "bulk_provisional"

golden.loc[
    reviewed_mask,
    "bulk_label_status"
] = "human_reviewed"


# ------------------------------------------------------------
# Save complete output
# ------------------------------------------------------------

golden.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ------------------------------------------------------------
# Create uncertainty file
# ------------------------------------------------------------

uncertain = golden[
    (
        golden["bulk_confidence"] == "low"
    )
    |
    (
        golden["bulk_confidence"] == "medium"
    )
].copy()


uncertain.to_csv(
    UNCERTAIN_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("BULK CONTEXT LABELING COMPLETE")
print("=" * 70)

print(f"\nTotal rows: {len(golden)}")

print(
    f"Human reviewed: "
    f"{reviewed_mask.sum()}"
)

print(
    f"Bulk provisional: "
    f"{len(golden) - reviewed_mask.sum()}"
)

print(
    f"Uncertain cases: "
    f"{len(uncertain)}"
)


print("\nBulk intent distribution:")

print(
    golden["bulk_intent"]
    .value_counts(dropna=False)
)


print("\nConfidence distribution:")

print(
    golden["bulk_confidence"]
    .value_counts(dropna=False)
)


print("\nOutput files:")

print(
    f"  Complete: {OUTPUT_FILE}"
)

print(
    f"  Uncertain: {UNCERTAIN_FILE}"
)


print("\nIMPORTANT:")
print("Existing human labels were preserved.")
print("Bulk labels are PROVISIONAL.")
print("Do NOT call the bulk labels human-labelled yet.")