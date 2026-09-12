import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

files = [
    BASE / "human_validation_50_reviewed.csv",
    BASE / "human_bulk_100.csv",
    BASE / "human_bulk_16.csv",
]

print("=" * 80)
print("MERGING HUMAN GOLDEN SET")
print("=" * 80)

dfs = []

for file in files:
    if not file.exists():
        print(f"\nERROR: File not found: {file}")
        raise SystemExit(1)

    df = pd.read_csv(file, dtype=str).fillna("")

    print(f"\n{file.name}")
    print(f"  Rows: {len(df)}")

    dfs.append(df)

# Combine
combined = pd.concat(dfs, ignore_index=True)

print(f"\nTotal rows before deduplication: {len(combined)}")

# Check duplicate tweet IDs
duplicates = combined[combined.duplicated("tweet_id", keep=False)]

if len(duplicates) > 0:
    print(f"Duplicate rows found: {len(duplicates)}")
else:
    print("Duplicate tweet IDs: 0")

# Remove duplicate tweet IDs
golden = combined.drop_duplicates(
    subset=["tweet_id"],
    keep="first"
).copy()

# Validate human labels
golden["human_intent"] = golden["human_intent"].astype(str).str.strip()

missing_labels = golden[
    (golden["human_intent"] == "") |
    (golden["human_intent"].str.lower() == "nan")
]

if len(missing_labels) > 0:
    print(f"\nWARNING: {len(missing_labels)} rows have missing human labels.")
else:
    print("Missing human labels: 0")

# Keep only properly labeled rows
golden = golden[
    (golden["human_intent"] != "") &
    (golden["human_intent"].str.lower() != "nan")
].copy()

# Save final golden set
output = BASE / "golden_human_160.csv"
golden.to_csv(output, index=False)

print("\n" + "=" * 80)
print("FINAL GOLDEN SET")
print("=" * 80)

print(f"Unique examples : {len(golden)}")

if "conversation_id" in golden.columns:
    print(f"Unique conversations: {golden['conversation_id'].nunique()}")

print("\nIntent distribution:")
print(golden["human_intent"].value_counts())

print(f"\nSaved to:")
print(output)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)