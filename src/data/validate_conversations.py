import pandas as pd
from pathlib import Path


# ==============================
# Configuration
# ==============================

INPUT_PATH = Path(
    "data/processed/applesupport_conversations.csv"
)

OUTPUT_PATH = Path(
    "data/processed/applesupport_threads.csv"
)


# ==============================
# Load data
# ==============================

print("=" * 70)
print("Loading AppleSupport dataset")
print("=" * 70)

df = pd.read_csv(INPUT_PATH, dtype=str)

print(f"Tweets loaded: {len(df):,}")


# ==============================
# Basic cleaning
# ==============================

df["tweet_id"] = df["tweet_id"].astype(str)

df["in_response_to_tweet_id"] = (
    df["in_response_to_tweet_id"]
    .fillna("")
    .astype(str)
)

df["inbound"] = (
    df["inbound"]
    .fillna("")
    .astype(str)
    .str.lower()
)


# ==============================
# Create tweet lookup
# ==============================

tweet_ids = set(df["tweet_id"])

parent_ids = set(
    df.loc[
        df["in_response_to_tweet_id"] != "",
        "in_response_to_tweet_id"
    ]
)

valid_parent_links = parent_ids.intersection(tweet_ids)

print()
print("=" * 70)
print("Relationship validation")
print("=" * 70)

print(f"Unique tweets: {len(tweet_ids):,}")
print(f"Parent links: {len(parent_ids):,}")
print(f"Valid parent links: {len(valid_parent_links):,}")

if len(parent_ids) > 0:
    coverage = (
        len(valid_parent_links) / len(parent_ids)
    ) * 100

    print(f"Parent link coverage: {coverage:.2f}%")


# ==============================
# Build parent → children mapping
# ==============================

children = {}

for _, row in df.iterrows():

    parent = row["in_response_to_tweet_id"]
    tweet = row["tweet_id"]

    if parent == "":
        continue

    if parent not in children:
        children[parent] = []

    children[parent].append(tweet)


# ==============================
# Find conversation roots
# ==============================

has_parent = set(
    df.loc[
        df["in_response_to_tweet_id"] != "",
        "tweet_id"
    ]
)

root_tweets = tweet_ids - has_parent

print()
print(f"Conversation root tweets: {len(root_tweets):,}")


# ==============================
# Build conversation IDs
# ==============================

conversation_ids = {}

conversation_number = 0

for root in root_tweets:

    conversation_number += 1

    conversation_id = f"apple_{conversation_number:06d}"

    stack = [root]
    visited = set()

    while stack:

        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)

        conversation_ids[current] = conversation_id

        for child in children.get(current, []):
            stack.append(child)


# ==============================
# Attach conversation IDs
# ==============================

df["conversation_id"] = (
    df["tweet_id"].map(conversation_ids)
)


# ==============================
# Statistics
# ==============================

conversation_counts = (
    df.groupby("conversation_id")
    .size()
)

print()
print("=" * 70)
print("Conversation statistics")
print("=" * 70)

print(
    f"Conversations reconstructed: "
    f"{conversation_counts.shape[0]:,}"
)

print(
    f"Average tweets/conversation: "
    f"{conversation_counts.mean():.2f}"
)

print(
    f"Median tweets/conversation: "
    f"{conversation_counts.median():.0f}"
)

print(
    f"Longest conversation: "
    f"{conversation_counts.max():,} tweets"
)


# ==============================
# Save
# ==============================

df = df.sort_values(
    ["conversation_id", "created_at"]
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(f"Saved to: {OUTPUT_PATH}")