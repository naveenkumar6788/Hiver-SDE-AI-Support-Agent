import pandas as pd

FILE = "golden_set/human_bulk_16.csv"

INTENTS = {
    "1": "battery_charging",
    "2": "ios_software_update",
    "3": "wifi_connectivity",
    "4": "app_problems",
    "5": "app_store_downloads",
    "6": "apple_music_itunes",
    "7": "apple_id_icloud",
    "8": "screen_display",
    "9": "calls_cellular",
    "10": "audio_speaker",
    "11": "device_hardware",
    "12": "other_unclear",
}

df = pd.read_csv(
    FILE,
    dtype=str
).fillna("")

df["human_intent"] = df["human_intent"].astype(str)
df["human_notes"] = df["human_notes"].astype(str)
df["label_status"] = df["label_status"].astype(str)

print("=" * 80)
print("HUMAN GOLDEN SET — FINAL 16 REVIEW")
print("=" * 80)

print("\nINTENTS:")
for number, intent in INTENTS.items():
    print(f" {number} = {intent}")

print("\nEnter labels like:")
print("1:4 2:11 3:2 4:12")
print("\nYou can paste all 16 labels at once.")
print("Type DONE when finished.")
print("-" * 80)

for _, row in df.iterrows():

    number = int(row["review_number"])

    print(
        f"\n[{number:03d}] Candidate: "
        f"{row['candidate_group']}"
    )

    print(
        f"Tweet ID: {row['tweet_id']}"
    )

    print(
        f"Text: {row['text']}"
    )

    print("-" * 80)

print("\nNOW ENTER YOUR LABELS.")

while True:

    user_input = input("> ").strip()

    if user_input.upper() == "DONE":
        break

    entries = user_input.split()

    changed = False

    for entry in entries:

        if ":" not in entry:
            print(
                f"Invalid format: {entry}"
            )
            continue

        number, label = entry.split(
            ":",
            1
        )

        number = number.strip()
        label = label.strip()

        if number not in df["review_number"].astype(str).values:
            print(
                f"Invalid review number: {number}"
            )
            continue

        if label not in INTENTS:
            print(
                f"Invalid intent number: {label}"
            )
            continue

        intent = INTENTS[label]

        mask = (
            df["review_number"].astype(str)
            == number
        )

        df.loc[
            mask,
            "human_intent"
        ] = intent

        df.loc[
            mask,
            "label_status"
        ] = "human_reviewed"

        changed = True

    if changed:

        df.to_csv(
            FILE,
            index=False
        )

        labeled = (
            df["human_intent"]
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        print(
            f"Saved. Labeled: "
            f"{labeled}/{len(df)}"
        )

print("\n")
print("=" * 80)
print("REVIEW COMPLETE")
print("=" * 80)

labeled = (
    df["human_intent"]
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)

print(
    f"Total examples : {len(df)}"
)

print(
    f"Human labeled  : {labeled}"
)

print(
    f"Remaining      : {len(df) - labeled}"
)

print("\nHuman label distribution:")

print(
    df.loc[
        df["human_intent"].astype(str).str.strip() != "",
        "human_intent"
    ]
    .value_counts()
    .to_string()
)

print(
    f"\nSaved to: {FILE}"
)