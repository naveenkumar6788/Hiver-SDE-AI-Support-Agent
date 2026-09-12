import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/processed/apple_customer_messages.csv")
OUTPUT_FILE = Path("golden_set/golden_candidates.csv")


# Candidate keywords for creating a diverse sample.
# These are NOT final labels.
KEYWORDS = {
    "battery_charging": [
        "battery", "charging", "charge", "battery life", "battery health"
    ],
    "ios_software_update": [
        "ios", "update", "updated", "updating", "software", "upgrade"
    ],
    "wifi_connectivity": [
        "wifi", "wi-fi", "internet", "network", "connection", "connect"
    ],
    "app_problems": [
        "app", "crash", "crashes", "freezes", "freezing"
    ],
    "app_store_downloads": [
        "app store", "download", "downloads", "install", "itunes store"
    ],
    "apple_music_itunes": [
        "apple music", "itunes", "music", "playlist", "song", "songs"
    ],
    "apple_id_icloud": [
        "apple id", "icloud", "account", "password", "login", "sign in"
    ],
    "screen_display": [
        "screen", "display", "brightness", "touch", "black screen"
    ],
    "calls_cellular": [
        "call", "calls", "calling", "phone call", "signal", "cellular"
    ],
    "audio_speaker": [
        "speaker", "sound", "audio", "volume", "microphone", "airpods"
    ],
    "device_hardware": [
        "camera", "button", "sim", "hardware", "broken", "device"
    ],
}


def contains_keyword(text, keywords):
    text = str(text).lower()
    return any(keyword in text for keyword in keywords)


def main():

    print("Loading customer messages...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Customer messages loaded: {len(df):,}")

    # Remove empty messages
    df = df[df["text"].notna()].copy()

    # Remove exact duplicate tweet IDs if any
    df = df.drop_duplicates(subset=["tweet_id"])

    # Create candidate groups
    candidates = []

    for intent, keywords in KEYWORDS.items():

        mask = df["text"].apply(
            lambda text: contains_keyword(text, keywords)
        )

        group = df[mask].copy()

        # Random sample from this candidate group.
        # We deliberately keep the number small because this is only
        # a candidate pool for later manual labeling.
        sample_size = min(30, len(group))

        if sample_size > 0:
            sample = group.sample(
                n=sample_size,
                random_state=42
            )

            sample["candidate_group"] = intent

            candidates.append(sample)

            print(
                f"{intent:25s}: "
                f"{len(group):6,} matches -> "
                f"{sample_size:2} candidates"
            )

    # Add a random group for unclear / miscellaneous messages
    used_ids = set()

    if candidates:
        used_ids = set(
            pd.concat(candidates)["tweet_id"].astype(str)
        )

    remaining = df[
        ~df["tweet_id"].astype(str).isin(used_ids)
    ]

    other_sample_size = min(30, len(remaining))

    if other_sample_size > 0:

        other_sample = remaining.sample(
            n=other_sample_size,
            random_state=42
        )

        other_sample["candidate_group"] = "other_unclear"

        candidates.append(other_sample)

        print(
            f"{'other_unclear':25s}: "
            f"{len(remaining):6,} available -> "
            f"{other_sample_size:2} candidates"
        )

    # Combine everything
    golden_candidates = pd.concat(
        candidates,
        ignore_index=True
    )

    # Shuffle final candidate dataset
    golden_candidates = golden_candidates.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Add empty column for human labeling
    golden_candidates["golden_intent"] = ""

    # Add notes column for labeling decisions
    golden_candidates["label_notes"] = ""

    # Keep only useful columns
    columns = [
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "conversation_id",
        "candidate_group",
        "golden_intent",
        "label_notes",
    ]

    golden_candidates = golden_candidates[columns]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    golden_candidates.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Golden candidate dataset created.")
    print(f"Rows: {len(golden_candidates):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()