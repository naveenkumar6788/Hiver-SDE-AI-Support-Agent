"""
Reconstructs multi-turn conversation threads for @AppleSupport from the raw TWCS dataset.
Processing is performed at the conversation thread level rather than on isolated tweets to
preserve complete multi-turn dialogue context and prevent cross-turn leakage.
"""

import re
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/raw/twcs/twcs.csv")
OUTPUT_PATH = Path("data/processed/applesupport_conversations.csv")

BRAND = "applesupport"
CHUNK_SIZE = 50_000


def normalize(value):
    """Convert a value to lowercase string."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def extract_tweet_ids(value):
    """Extract tweet IDs from response_tweet_id string."""
    if pd.isna(value):
        return set()
    return set(re.findall(r"\d+", str(value)))


print("=" * 70)
print("PHASE 1: Finding AppleSupport-related tweets")
print("=" * 70)

seed_ids = set()
total_rows = 0
brand_rows = 0

for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE, dtype=str):
    total_rows += len(chunk)

    author = chunk["author_id"].fillna("").str.strip().str.lower()
    text = chunk["text"].fillna("").str.lower()

    mask = (author == BRAND) | (text.str.contains("@applesupport", regex=False))
    matched = chunk.loc[mask]
    brand_rows += len(matched)

    seed_ids.update(matched["tweet_id"].dropna().astype(str).tolist())

    for value in matched["in_response_to_tweet_id"]:
        if pd.notna(value):
            seed_ids.add(str(value))

    for value in matched["response_tweet_id"]:
        seed_ids.update(extract_tweet_ids(value))

print(f"\nDataset rows scanned: {total_rows:,}")
print(f"AppleSupport-related rows: {brand_rows:,}")
print(f"Initial related tweet IDs: {len(seed_ids):,}")

print("\n" + "=" * 70)
print("PHASE 2: Extracting related tweets")
print("=" * 70)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
first_write = True
saved_rows = 0

for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE, dtype=str):
    tweet_ids = chunk["tweet_id"].fillna("").astype(str)
    mask = tweet_ids.isin(seed_ids)
    matched = chunk.loc[mask].copy()

    if len(matched) == 0:
        continue

    matched.to_csv(
        OUTPUT_PATH,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )
    first_write = False
    saved_rows += len(matched)

print("\n" + "=" * 70)
print("RECONSTRUCTION COMPLETE")
print("=" * 70)
print(f"Tweets saved: {saved_rows:,}")
print(f"Output file: {OUTPUT_PATH}")