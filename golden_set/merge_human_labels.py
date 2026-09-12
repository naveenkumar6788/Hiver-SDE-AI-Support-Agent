import pandas as pd
from pathlib import Path

BASE_FILE = Path("golden_set/final_golden.csv")
HUMAN_FILE = Path("golden_set/final_golden_human_reviewed.csv")
OUTPUT_FILE = Path("golden_set/final_golden.csv")

print("Loading files...")

base = pd.read_csv(BASE_FILE)
human = pd.read_csv(HUMAN_FILE)

print(f"Base rows: {len(base)}")
print(f"Human review rows: {len(human)}")

# Make tweet IDs consistent
base["tweet_id"] = base["tweet_id"].astype(str)
human["tweet_id"] = human["tweet_id"].astype(str)

# Make sure required columns exist
for col in ["human_intent", "human_notes", "label_status"]:
    if col not in base.columns:
        base[col] = ""

# Merge human decisions
human_lookup = human.set_index("tweet_id")

reviewed_count = 0

for idx, row in base.iterrows():

    tweet_id = str(row["tweet_id"])

    if tweet_id in human_lookup.index:

        human_row = human_lookup.loc[tweet_id]

        # Only accept rows that actually received a human label
        intent = human_row.get("human_intent", "")

        if pd.notna(intent) and str(intent).strip() != "":
            base.at[idx, "human_intent"] = str(intent).strip()

            notes = human_row.get("human_notes", "")
            if pd.notna(notes):
                base.at[idx, "human_notes"] = str(notes)

            base.at[idx, "label_status"] = "human_reviewed"

            reviewed_count += 1

print()
print("=" * 70)
print("HUMAN LABEL MERGE COMPLETE")
print("=" * 70)

print(f"Human-reviewed rows recovered: {reviewed_count}")
print(f"Total rows: {len(base)}")

print()
print("Label status:")
print(base["label_status"].value_counts(dropna=False))

print()
print("Human-reviewed intents:")

reviewed = base[base["label_status"] == "human_reviewed"]

if len(reviewed) > 0:
    print(reviewed["human_intent"].value_counts())
else:
    print("NONE")

# Save
base.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Saved: {OUTPUT_FILE}")