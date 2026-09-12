import pandas as pd
import numpy as np
import os

SOURCE_FILE = "data/processed/apple_customer_messages.csv"

EXISTING_FILES = [
    "golden_set/human_validation_50_reviewed.csv",
    "golden_set/human_bulk_100.csv",
]

OUTPUT_FILE = "golden_set/human_bulk_16.csv"

RANDOM_STATE = 84
SAMPLE_SIZE = 16

INTENTS = [
    "battery_charging",
    "ios_software_update",
    "wifi_connectivity",
    "app_problems",
    "app_store_downloads",
    "apple_music_itunes",
    "apple_id_icloud",
    "screen_display",
    "calls_cellular",
    "audio_speaker",
    "device_hardware",
    "other_unclear",
]

print("=" * 70)
print("CREATING 16 NEW HUMAN-LABEL REVIEW EXAMPLES")
print("=" * 70)

df = pd.read_csv(SOURCE_FILE)

print(f"\nCustomer messages available: {len(df)}")

# --------------------------------------------------
# Remove ALL previously used human examples
# --------------------------------------------------

used_ids = set()

for existing_file in EXISTING_FILES:

    if os.path.exists(existing_file):

        existing = pd.read_csv(
            existing_file,
            dtype=str
        )

        if "tweet_id" in existing.columns:

            ids = set(
                existing["tweet_id"]
                .dropna()
                .astype(str)
                .tolist()
            )

            used_ids.update(ids)

            print(
                f"Loaded {len(ids)} IDs from "
                f"{existing_file}"
            )

print(f"\nTotal previously used IDs: {len(used_ids)}")

df = df[
    ~df["tweet_id"]
    .astype(str)
    .isin(used_ids)
].copy()

print(
    f"Remaining unused customer messages: {len(df)}"
)

# --------------------------------------------------
# Remove duplicates / empty messages
# --------------------------------------------------

df = df.drop_duplicates(
    subset=["tweet_id"]
).copy()

df = df[
    df["text"].notna()
    & (df["text"].astype(str).str.strip() != "")
].copy()

# --------------------------------------------------
# Candidate group detection
# --------------------------------------------------

df["text_lower"] = df["text"].astype(str).str.lower()


def candidate_group(text):

    t = text.lower()

    if any(x in t for x in [
        "battery", "charge", "charging", "charger",
        "drain", "dies", "power"
    ]):
        return "battery_charging"

    if any(x in t for x in [
        "ios", "update", "upgrade", "downgrade",
        "ios 11", "ios11", "software"
    ]):
        return "ios_software_update"

    if any(x in t for x in [
        "wifi", "wi-fi", "internet", "router",
        "network", "connect"
    ]):
        return "wifi_connectivity"

    if any(x in t for x in [
        "app", "application", "crash", "freeze",
        "freezing", "bug"
    ]):
        return "app_problems"

    if any(x in t for x in [
        "download", "app store", "itunes store",
        "install", "waiting"
    ]):
        return "app_store_downloads"

    if any(x in t for x in [
        "music", "itunes", "song", "playlist",
        "apple music"
    ]):
        return "apple_music_itunes"

    if any(x in t for x in [
        "apple id", "icloud", "account", "login",
        "password", "locked", "verification", "2fa"
    ]):
        return "apple_id_icloud"

    if any(x in t for x in [
        "screen", "display", "touch", "brightness",
        "keyboard", "letter", "pixel"
    ]):
        return "screen_display"

    if any(x in t for x in [
        "call", "calling", "phone call", "cellular",
        "signal", "sim", "carrier"
    ]):
        return "calls_cellular"

    if any(x in t for x in [
        "speaker", "sound", "audio", "airpods",
        "headphones", "bluetooth"
    ]):
        return "audio_speaker"

    if any(x in t for x in [
        "broken", "heating", "overheating", "hardware",
        "button", "camera", "phone won't",
        "phone wont"
    ]):
        return "device_hardware"

    return "other_unclear"


df["candidate_group"] = df["text"].apply(
    candidate_group
)

# --------------------------------------------------
# Sample approximately evenly
# --------------------------------------------------

rng = np.random.default_rng(
    RANDOM_STATE
)

available_groups = [
    g for g in INTENTS
    if (
        df["candidate_group"] == g
    ).sum() > 0
]

per_group = SAMPLE_SIZE // len(
    available_groups
)

remainder = SAMPLE_SIZE % len(
    available_groups
)

parts = []

for i, group in enumerate(
    available_groups
):

    group_df = df[
        df["candidate_group"] == group
    ]

    n = per_group

    if i < remainder:
        n += 1

    n = min(
        n,
        len(group_df)
    )

    if n > 0:

        sampled = group_df.sample(
            n=n,
            random_state=RANDOM_STATE + i
        )

        parts.append(sampled)

result = pd.concat(
    parts,
    ignore_index=True
)

# --------------------------------------------------
# Fill if sampling produced fewer than 16
# --------------------------------------------------

if len(result) < SAMPLE_SIZE:

    remaining = df[
        ~df["tweet_id"].astype(str).isin(
            result["tweet_id"].astype(str)
        )
    ]

    extra_n = min(
        SAMPLE_SIZE - len(result),
        len(remaining)
    )

    if extra_n > 0:

        extra = remaining.sample(
            n=extra_n,
            random_state=999
        )

        result = pd.concat(
            [result, extra],
            ignore_index=True
        )

# --------------------------------------------------
# Shuffle
# --------------------------------------------------

result = result.sample(
    frac=1,
    random_state=123
).reset_index(drop=True)

result["review_number"] = range(
    1,
    len(result) + 1
)

result["human_intent"] = ""
result["human_notes"] = ""
result["label_status"] = ""

# --------------------------------------------------
# Keep useful columns
# --------------------------------------------------

result = result[
    [
        "review_number",
        "tweet_id",
        "created_at",
        "conversation_id",
        "text",
        "candidate_group",
        "human_intent",
        "human_notes",
        "label_status",
    ]
]

# --------------------------------------------------
# Save
# --------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("CREATED")
print("=" * 70)

print(
    f"Examples created: {len(result)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print()
print("Candidate-group distribution:")

print(
    result["candidate_group"]
    .value_counts()
    .to_string()
)

print()
print("These are NOT automatically labeled.")
print("You will provide the human labels.")