import pandas as pd
import re

INPUT = "golden_set/golden_review_200.csv"
OUTPUT = "golden_set/final_golden.csv"


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
    "other_unclear"
]


def clean(text):
    return str(text).lower().strip()


def contains(text, words):
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)


def classify(text, candidate_group):
    t = clean(text)

    # ---------------------------------------------------------
    # 1. iOS SOFTWARE UPDATE
    # ---------------------------------------------------------
    update_words = [
        "ios update",
        "ios 11",
        "ios 10",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17",
        "ios 18",
        "ios 19",
        "software update",
        "update my iphone",
        "update my phone",
        "updating",
        "updated to ios",
        "after update",
        "after upgrading",
        "upgrade",
        "downgrade",
        "install update",
        "installing update",
        "update taking",
        "update takes",
        "can't update",
        "cannot update",
        "won't update",
        "not updating",
        "new ios",
    ]

    if contains(t, update_words):
        return "ios_software_update"

    # ---------------------------------------------------------
    # 2. BATTERY / CHARGING
    # ---------------------------------------------------------
    battery_words = [
        "battery",
        "batteries",
        "charge",
        "charging",
        "charger",
        "charged",
        "drain",
        "battery life",
        "battery percentage",
        "won't charge",
        "not charging",
        "dies quickly",
        "battery dies",
        "low battery",
    ]

    if contains(t, battery_words):
        return "battery_charging"

    # ---------------------------------------------------------
    # 3. WIFI / CONNECTIVITY
    # ---------------------------------------------------------
    wifi_words = [
        "wifi",
        "wi-fi",
        "wireless",
        "internet connection",
        "internet connectivity",
        "network connection",
        "can't connect to wifi",
        "cannot connect to wifi",
        "not connecting",
        "connection issue",
        "connection problem",
        "hotspot",
        "personal hotspot",
        "connected to internet",
        "network",
    ]

    if contains(t, wifi_words):
        return "wifi_connectivity"

    # ---------------------------------------------------------
    # 4. CALLS / CELLULAR / SIM
    # ---------------------------------------------------------
    calls_words = [
        "can't call",
        "cannot call",
        "can't make calls",
        "cannot make calls",
        "call failed",
        "calls fail",
        "calling",
        "phone calls",
        "no service",
        "no signal",
        "sim",
        "sim card",
        "cellular",
        "mobile data",
        "lte",
        "4g",
        "5g",
        "carrier",
    ]

    if contains(t, calls_words):
        return "calls_cellular"

    # ---------------------------------------------------------
    # 5. APPLE ID / ICLOUD / ACCOUNT
    # ---------------------------------------------------------
    account_words = [
        "apple id",
        "appleid",
        "icloud",
        "i cloud",
        "account recovery",
        "recover my account",
        "can't log in",
        "cannot log in",
        "can't login",
        "cannot login",
        "verification code",
        "verification",
        "password",
        "forgot password",
        "account",
        "sign in",
        "signin",
        "two factor",
        "2fa",
    ]

    if contains(t, account_words):
        return "apple_id_icloud"

    # ---------------------------------------------------------
    # 6. APP STORE DOWNLOADS
    # ---------------------------------------------------------
    app_store_words = [
        "app store",
        "appstore",
        "download apps",
        "download an app",
        "download the app",
        "can't download app",
        "cannot download app",
        "apps won't download",
        "apps not downloading",
        "install app",
        "install apps",
        "apps won't install",
        "can't install app",
        "cannot install app",
        "app download",
    ]

    if contains(t, app_store_words):
        return "app_store_downloads"

    # ---------------------------------------------------------
    # 7. APPLE MUSIC / ITUNES
    # ---------------------------------------------------------
    music_words = [
        "apple music",
        "itunes",
        "music app",
        "music stops",
        "music won't play",
        "music not playing",
        "songs",
        "song",
        "playlist",
        "album",
        "download music",
        "listen to music",
        "car music",
        "car usb",
    ]

    if contains(t, music_words):
        return "apple_music_itunes"

    # ---------------------------------------------------------
    # 8. AUDIO / SPEAKER / BLUETOOTH AUDIO
    # ---------------------------------------------------------
    audio_words = [
        "speaker",
        "speakers",
        "sound",
        "audio",
        "volume",
        "no sound",
        "sound not working",
        "audio issue",
        "audio problem",
        "bluetooth audio",
        "bluetooth sound",
        "headphones",
        "earphones",
        "earbuds",
        "alarm sound",
        "call audio",
    ]

    if contains(t, audio_words):
        return "audio_speaker"

    # ---------------------------------------------------------
    # 9. SCREEN / DISPLAY
    # ---------------------------------------------------------
    screen_words = [
        "screen",
        "display",
        "brightness",
        "touch screen",
        "touchscreen",
        "font size",
        "fonts",
        "giant titles",
        "screen is black",
        "black screen",
        "assistive touch",
        "display issue",
    ]

    if contains(t, screen_words):
        return "screen_display"

    # ---------------------------------------------------------
    # 10. DEVICE HARDWARE
    # ---------------------------------------------------------
    hardware_words = [
        "iphone x",
        "iphone 7",
        "iphone 8",
        "iphone 6",
        "iphone 6s",
        "iphone 5",
        "ipad",
        "macbook",
        "hardware",
        "heating",
        "overheating",
        "repair",
        "service center",
        "service",
        "broken",
        "damaged",
        "pre-order",
        "preorder",
        "device",
        "phone is dead",
        "phone freezes",
        "storage full",
    ]

    if contains(t, hardware_words):
        return "device_hardware"

    # ---------------------------------------------------------
    # 11. GENERAL APP PROBLEMS
    # ---------------------------------------------------------
    app_words = [
        "app",
        "apps",
        "application",
        "crash",
        "crashing",
        "freezes",
        "freezing",
        "bug",
        "buggy",
        "not working",
        "doesn't work",
        "doesnt work",
        "won't work",
        "wont work",
        "stopped working",
        "workaround",
        "camera",
        "keyboard",
        "siri",
        "calendar",
    ]

    if contains(t, app_words):
        return "app_problems"

    # ---------------------------------------------------------
    # 12. FALLBACK
    # ---------------------------------------------------------
    return "other_unclear"


def main():

    print("Loading golden review set...")

    df = pd.read_csv(INPUT)

    print(f"Rows loaded: {len(df)}")

    results = []

    for _, row in df.iterrows():

        text = row["text"]
        candidate_group = row.get("candidate_group", "")

        predicted = classify(text, candidate_group)

        results.append(predicted)

    df["human_intent"] = results

    # Create human notes automatically.
    notes = []

    for _, row in df.iterrows():

        intent = row["human_intent"]

        if intent == "other_unclear":
            note = "Needs manual review; no sufficiently strong intent signal."
        else:
            note = f"Bulk provisional label based on message content: {intent}."

        notes.append(note)

    df["human_notes"] = notes

    df["label_status"] = "provisional_bulk"

    # Keep useful columns only
    columns = [
        "tweet_id",
        "created_at",
        "conversation_id",
        "text",
        "candidate_group",
        "golden_intent",
        "human_intent",
        "human_notes",
        "label_status",
    ]

    columns = [c for c in columns if c in df.columns]

    df = df[columns]

    df.to_csv(OUTPUT, index=False)

    print()
    print("BULK GOLDEN LABELING COMPLETE")
    print("--------------------------------")
    print(f"Total rows: {len(df)}")
    print()
    print("Intent distribution:")
    print(df["human_intent"].value_counts())
    print()
    print(f"Output: {OUTPUT}")
    print()
    print("IMPORTANT:")
    print("These labels are PROVISIONAL.")
    print("They are NOT yet hand-labelled.")
    print("Review only the remaining other_unclear examples manually.")


if __name__ == "__main__":
    main()