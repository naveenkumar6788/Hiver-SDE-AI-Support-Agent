import pandas as pd
import os

INPUT_FILE = "golden_set/golden_human_160.csv"
OUTPUT_FILE = "golden_set/golden_independently_verified.csv"
BATCH_SIZE = 20

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
    "12": "other_unclear"
}

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

# Load previous progress if available
if os.path.exists(OUTPUT_FILE):
    verified = pd.read_csv(OUTPUT_FILE, dtype=str).fillna("")

    # Keep verified answers
    verified_map = dict(
        zip(
            verified["tweet_id"].astype(str),
            verified["verified_intent"]
        )
    )

    status_map = dict(
        zip(
            verified["tweet_id"].astype(str),
            verified["verification_status"]
        )
    )
else:
    verified_map = {}
    status_map = {}

# Identify the original 44 genuinely reviewed examples
reviewed_file = "golden_set/human_validation_50_reviewed.csv"

if os.path.exists(reviewed_file):
    reviewed = pd.read_csv(reviewed_file, dtype=str).fillna("")
    original_human_ids = set(reviewed["tweet_id"].astype(str))
else:
    original_human_ids = set()

# Initialize columns
df["verification_status"] = ""
df["verified_intent"] = ""

# Restore previous progress
for i, row in df.iterrows():
    tweet_id = str(row["tweet_id"])

    if tweet_id in verified_map:
        df.at[i, "verified_intent"] = verified_map[tweet_id]
        df.at[i, "verification_status"] = status_map.get(
            tweet_id, "independently_verified"
        )

# Automatically preserve the original 44
for i, row in df.iterrows():
    tweet_id = str(row["tweet_id"])

    if tweet_id in original_human_ids:
        df.at[i, "verified_intent"] = row["human_intent"]
        df.at[i, "verification_status"] = "previously_human_reviewed"

# Find rows still needing independent verification
pending_indices = [
    i for i, row in df.iterrows()
    if df.at[i, "verification_status"] == ""
]

print("\n" + "=" * 80)
print("BULK GOLDEN SET VERIFICATION")
print("=" * 80)

print(f"Total examples: {len(df)}")
print(f"Already verified: {len(df) - len(pending_indices)}")
print(f"Remaining: {len(pending_indices)}")

while pending_indices:

    batch = pending_indices[:BATCH_SIZE]

    print("\n" + "=" * 80)
    print(f"NEW BATCH: {len(batch)} examples")
    print("=" * 80)

    # Display messages
    for i in batch:
        print("\n" + "-" * 80)
        print(f"Example {i + 1} / {len(df)}")
        print("-" * 80)
        print(df.at[i, "text"])

    print("\n" + "=" * 80)
    print("INTENTS")
    print("=" * 80)

    for key, value in INTENTS.items():
        print(f"{key} = {value}")

    print("\nEnter labels for this batch.")
    print("Format:")
    print("45=1 46=12 47=3 ...")
    print()

    while True:

        answer = input("Labels: ").strip()

        parts = answer.split()

        labels = {}

        valid = True

        for part in parts:

            if "=" not in part:
                valid = False
                break

            example_number, label = part.split("=", 1)

            if not example_number.isdigit():
                valid = False
                break

            if label not in INTENTS:
                valid = False
                break

            row_number = int(example_number) - 1

            if row_number not in batch:
                valid = False
                break

            labels[row_number] = INTENTS[label]

        if not valid:
            print("\nInvalid format.")
            print("Example:")
            print("45=1 46=12 47=3")
            continue

        if len(labels) != len(batch):
            print(
                f"\nYou must provide exactly {len(batch)} labels "
                f"for this batch."
            )
            continue

        break

    # Save labels
    for i, intent in labels.items():
        df.at[i, "verified_intent"] = intent
        df.at[i, "verification_status"] = "independently_verified"

    # Save immediately
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nBatch saved successfully.")

    pending_indices = [
        i for i, row in df.iterrows()
        if df.at[i, "verification_status"] == ""
    ]

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(f"Total examples: {len(df)}")

print("\nVerification status:")
print(df["verification_status"].value_counts())

print("\nFinal intent distribution:")
print(df["verified_intent"].value_counts())

print(f"\nSaved to:")
print(OUTPUT_FILE)