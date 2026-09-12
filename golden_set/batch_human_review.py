import pandas as pd
import os
import re

INPUT_FILE = "golden_set/human_validation_50.csv"
OUTPUT_FILE = "golden_set/human_validation_50_reviewed.csv"

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

print("=" * 70)
print("HIVER — BULK HUMAN VALIDATION REVIEW")
print("=" * 70)

# Load validation data
golden = pd.read_csv(INPUT_FILE)

# Load previous progress if available
if os.path.exists(OUTPUT_FILE):
    review = pd.read_csv(OUTPUT_FILE)

    # Make sure columns exist
    if "human_intent" not in review.columns:
        review["human_intent"] = ""

    if "human_notes" not in review.columns:
        review["human_notes"] = ""

    if "label_status" not in review.columns:
        review["label_status"] = ""

    # Match previous labels back using tweet_id
    previous = review.set_index("tweet_id")

    golden["human_intent"] = golden["tweet_id"].map(
        previous["human_intent"]
    ).fillna("")

    golden["human_notes"] = golden["tweet_id"].map(
        previous["human_notes"]
    ).fillna("")

    golden["label_status"] = golden["tweet_id"].map(
        previous["label_status"]
    ).fillna("")
else:
    golden["human_intent"] = ""
    golden["human_notes"] = ""
    golden["label_status"] = ""

# Normalize
golden["human_intent"] = golden["human_intent"].fillna("").astype(str)

# Only unlabeled rows
remaining = golden[
    golden["human_intent"].str.strip() == ""
].copy()

print()
print(f"Total validation rows: {len(golden)}")
print(f"Already reviewed: {len(golden) - len(remaining)}")
print(f"Remaining: {len(remaining)}")

if len(remaining) == 0:
    print()
    print("All validation examples are already reviewed.")
    exit()

print()
print("=" * 70)
print("INTENT LABELS")
print("=" * 70)

for number, intent in INTENTS.items():
    print(f"{number:2} = {intent}")

print()
print("=" * 70)
print("REMAINING EXAMPLES")
print("=" * 70)

# Display every remaining example
for i, (_, row) in enumerate(remaining.iterrows(), start=1):
    print()
    print("-" * 70)
    print(f"EXAMPLE {i}")
    print("-" * 70)

    print(f"Tweet ID: {row['tweet_id']}")
    print(f"Conversation ID: {row['conversation_id']}")
    print()
    print("CUSTOMER MESSAGE:")
    print(row["text"])

    if "bulk_intent" in row:
        print()
        print("BULK PREDICTION:")
        print(f"Intent: {row['bulk_intent']}")

    print()

print()
print("=" * 70)
print("ENTER ALL LABELS")
print("=" * 70)

print()
print("Format:")
print("1=2,2=7,3=12,...")
print()
print("The number before '=' is the EXAMPLE number.")
print("The number after '=' is the INTENT number.")
print()
print("Example:")
print("1=2 means Example 1 -> ios_software_update")
print()

while True:
    raw = input("Enter labels: ").strip()

    if not raw:
        print("No labels entered.")
        continue

    pairs = re.findall(r"(\d+)\s*=\s*(\d+)", raw)

    if len(pairs) != len(remaining):
        print()
        print(
            f"ERROR: Expected {len(remaining)} labels "
            f"but found {len(pairs)}."
        )
        print("Please enter one label for every example.")
        continue

    labels = {}

    valid = True

    for example_num, intent_num in pairs:
        example_num = int(example_num)
        intent_num = int(intent_num)

        if example_num < 1 or example_num > len(remaining):
            print(f"ERROR: Invalid example number: {example_num}")
            valid = False
            break

        if intent_num not in INTENTS:
            print(f"ERROR: Invalid intent number: {intent_num}")
            valid = False
            break

        if example_num in labels:
            print(f"ERROR: Duplicate example number: {example_num}")
            valid = False
            break

        labels[example_num] = INTENTS[intent_num]

    if not valid:
        continue

    missing = [
        i for i in range(1, len(remaining) + 1)
        if i not in labels
    ]

    if missing:
        print(f"ERROR: Missing examples: {missing}")
        continue

    break

# Apply labels
remaining_indices = list(remaining.index)

for example_num, intent in labels.items():
    original_index = remaining_indices[example_num - 1]

    golden.loc[original_index, "human_intent"] = intent
    golden.loc[original_index, "label_status"] = "human_reviewed"

    if not str(golden.loc[original_index, "human_notes"]).strip():
        golden.loc[original_index, "human_notes"] = (
            "Bulk human review"
        )

# Save
golden.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 70)
print("BULK REVIEW SAVED")
print("=" * 70)

reviewed_count = (
    golden["human_intent"].str.strip() != ""
).sum()

remaining_count = len(golden) - reviewed_count

print(f"Human reviewed so far: {reviewed_count}")
print(f"Remaining: {remaining_count}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 70)