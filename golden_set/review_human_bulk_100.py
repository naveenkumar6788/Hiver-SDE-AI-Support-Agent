import pandas as pd

FILE = "golden_set/human_bulk_100.csv"

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

df = pd.read_csv(FILE, dtype=str)

# Make sure these columns can store text
if "human_intent" not in df.columns:
    df["human_intent"] = ""

if "label_status" not in df.columns:
    df["label_status"] = ""

df["human_intent"] = df["human_intent"].fillna("").astype(str)
df["label_status"] = df["label_status"].fillna("").astype(str)

print("=" * 80)
print("HUMAN GOLDEN SET — BULK REVIEW")
print("=" * 80)

print("\nINTENTS:")
for key, value in INTENTS.items():
    print(f"{key:>2} = {value}")

print("\nEnter labels like:")
print("1:4 2:2 3:12 4:7")
print()
print("You can paste all 100 labels at once.")
print()

for _, row in df.iterrows():
    print("-" * 80)
    print(
        f"[{int(row['review_number']):03d}] "
        f"Candidate: {row['candidate_group']}"
    )
    print(f"Tweet ID: {row['tweet_id']}")
    print(f"Text: {row['text']}")

print("-" * 80)
print("\nNOW ENTER YOUR LABELS.")
print("> ", end="")

# Read multiple lines until the user enters DONE
while True:
    line = input().strip()

    if line.upper() == "DONE":
        break

    if line.lower() == "q":
        break

    if not line:
        print("> ", end="")
        continue

    tokens = line.split()

    for token in tokens:

        if ":" not in token:
            print(f"Skipping invalid entry: {token}")
            continue

        left, right = token.split(":", 1)

        try:
            review_number = int(left)
        except ValueError:
            print(f"Invalid review number: {left}")
            continue

        intent_number = right.strip()

        if intent_number not in INTENTS:
            print(
                f"Invalid intent number for "
                f"{review_number}: {intent_number}"
            )
            continue

        mask = df["review_number"].astype(str) == str(review_number)

        if not mask.any():
            print(f"Invalid review number: {review_number}")
            continue

        intent = INTENTS[intent_number]

        df.loc[mask, "human_intent"] = intent
        df.loc[mask, "label_status"] = "human_reviewed"

    # Save after EVERY LINE
    df.to_csv(FILE, index=False)

    labeled = (
        df["human_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    )

    print(
        f"Saved. Labeled: {labeled.sum()}/"
        f"{len(df)}"
    )
    print("> ", end="")

df.to_csv(FILE, index=False)

labeled = (
    df["human_intent"]
    .fillna("")
    .astype(str)
    .str.strip()
    != ""
)

print("\n" + "=" * 80)
print("REVIEW COMPLETE")
print("=" * 80)

print(f"Total examples : {len(df)}")
print(f"Human labeled  : {labeled.sum()}")
print(f"Remaining      : {(~labeled).sum()}")

if labeled.sum() > 0:
    print("\nHuman label distribution:")
    print(
        df.loc[labeled, "human_intent"]
        .value_counts()
        .to_string()
    )

print(f"\nSaved to: {FILE}")