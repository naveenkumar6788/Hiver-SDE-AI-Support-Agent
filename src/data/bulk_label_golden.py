import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path("golden_set/golden_candidates.csv")
OUTPUT_FILE = Path("golden_set/golden_labeled.csv")


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


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text, words):
    return any(word in text for word in words)


def classify(text):
    text = clean_text(text)

    scores = {i: 0 for i in INTENTS}

    # 1. Battery / Charging
    if contains_any(text, [
        "battery", "batteries", "battery life", "battery drain",
        "charging", "charge", "charger", "charging port",
        "power", "dies", "drain"
    ]):
        scores[1] += 5

    # 2. iOS / Software Update
    if contains_any(text, [
        "ios", "ios 10", "ios 11", "ios 12", "ios 13",
        "update", "upgrade", "downgrade", "software update",
        "latest ios", "previous ios"
    ]):
        scores[2] += 3

    # 3. WiFi / Connectivity
    if contains_any(text, [
        "wifi", "wi-fi", "wireless", "internet connection",
        "router", "network", "bluetooth"
    ]):
        scores[3] += 5

    # 4. App Problems
    if contains_any(text, [
        "app crash", "app crashes", "apps crash", "apps crashes",
        "application", "app doesn't work", "app won't work",
        "app not working", "app problem", "app issue",
        "safari crash", "game", "crash", "freezes", "freezing"
    ]):
        scores[4] += 4

    # 5. App Store / Downloads
    if contains_any(text, [
        "app store", "download app", "download apps",
        "can't download", "cannot download", "downloading",
        "redownload", "install app", "install apps",
        "local apps"
    ]):
        scores[5] += 5

    # 6. Apple Music / iTunes
    if contains_any(text, [
        "apple music", "itunes", "music app", "songs",
        "song", "spotify", "audible", "music"
    ]):
        scores[6] += 5

    # 7. Apple ID / iCloud
    if contains_any(text, [
        "apple id", "icloud", "icloud account",
        "itunes account", "account disabled",
        "account hacked", "password", "verification",
        "two factor", "2fa"
    ]):
        scores[7] += 5

    # 8. Screen / Display
    if contains_any(text, [
        "screen", "display", "touchscreen", "touch screen",
        "cracked screen", "black screen", "screen black",
        "assistive touch", "pixel", "pixels", "bar on screen"
    ]):
        scores[8] += 5

    # 9. Calls / Cellular
    if contains_any(text, [
        "can't call", "cannot call", "can't make calls",
        "cannot make calls", "call", "calls", "calling",
        "phone call", "cellular", "cell signal", "signal",
        "sim", "no service", "mobile network"
    ]):
        scores[9] += 5

    # 10. Audio / Speaker
    if contains_any(text, [
        "speaker", "speakers", "volume", "sound",
        "audio", "earphone", "earphones", "earpod",
        "airpods", "headphones", "ringer"
    ]):
        scores[10] += 5

    # 11. Device / Hardware
    if contains_any(text, [
        "device", "iphone", "ipad", "macbook", "mac",
        "heating", "overheating", "broken", "physical damage",
        "repair", "service center", "hardware", "button",
        "home button", "power button"
    ]):
        scores[11] += 2

    # Important priority adjustments
    #
    # WiFi should beat generic iOS/update mentions.
    if contains_any(text, ["wifi", "wi-fi"]):
        scores[3] += 5
        scores[2] -= 2

    # Battery should beat generic iOS/update mentions.
    if contains_any(text, ["battery", "battery life", "battery drain"]):
        scores[1] += 5
        scores[2] -= 2

    # Calls should beat generic app crash/update mentions
    # when inability to call is explicit.
    if contains_any(text, [
        "can't call", "cannot call",
        "can't make calls", "cannot make calls"
    ]):
        scores[9] += 8

    # App Store should beat generic Apple ID when downloading
    # apps is the actual problem.
    if contains_any(text, [
        "app store", "can't download", "cannot download",
        "download apps", "local apps"
    ]):
        scores[5] += 5

    # Music-specific problems should beat generic app problems.
    if contains_any(text, [
        "apple music", "itunes", "spotify",
        "audible", "music app"
    ]):
        scores[6] += 5

    # Find best score
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # If no useful signal, Other / Unclear
    if best_score <= 0:
        return 12, "No clear intent signal."

    return best_intent, f"Bulk classification based on message content."


def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    print("Loading candidates...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Candidates loaded: {len(df):,}")

    # Preserve manually labeled data if it already exists.
    if OUTPUT_FILE.exists():

        print("Existing golden_labeled.csv found.")
        existing = pd.read_csv(OUTPUT_FILE)

        if "golden_intent" in existing.columns:
            existing_labels = existing[
                ["tweet_id", "golden_intent", "label_notes", "label_status"]
            ].copy()

            df = df.merge(
                existing_labels,
                on="tweet_id",
                how="left"
            )

        else:
            df["golden_intent"] = ""
            df["label_notes"] = ""
            df["label_status"] = ""

    else:
        df["golden_intent"] = ""
        df["label_notes"] = ""
        df["label_status"] = ""

    # Find text column
    if "text" not in df.columns:
        raise ValueError("Could not find 'text' column.")

    processed = 0
    preserved = 0

    for idx, row in df.iterrows():

        current = str(row.get("golden_intent", "")).strip()

        # Preserve manually labeled examples.
        if current and current.lower() not in ["nan", "none"]:
            preserved += 1
            continue

        intent_number, note = classify(row["text"])

        df.at[idx, "golden_intent"] = INTENTS[intent_number]
        df.at[idx, "label_notes"] = note
        df.at[idx, "label_status"] = "bulk_labeled"

        processed += 1

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("BULK LABELING COMPLETE")
    print("=" * 60)
    print(f"Total candidates:       {len(df):,}")
    print(f"Newly bulk labeled:     {processed:,}")
    print(f"Existing labels kept:   {preserved:,}")
    print(f"Output:                 {OUTPUT_FILE}")

    print()
    print("Intent distribution:")
    print(df["golden_intent"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()