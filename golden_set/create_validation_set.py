import pandas as pd
from pathlib import Path

INPUT_FILE = Path("golden_set/bulk_context_labeled.csv")
OUTPUT_FILE = Path("golden_set/human_validation_50.csv")

print("Loading bulk context labels...")

df = pd.read_csv(INPUT_FILE)

print(f"Total rows available: {len(df)}")

# ---------------------------------------------------------
# Separate human-reviewed and provisional examples
# ---------------------------------------------------------

human = df[df["bulk_confidence"] == "human"].copy()

provisional = df[df["bulk_confidence"] != "human"].copy()

# ---------------------------------------------------------
# Select examples across confidence levels
# ---------------------------------------------------------

high = provisional[
    provisional["bulk_confidence"].astype(str).str.lower() == "high"
].copy()

medium = provisional[
    provisional["bulk_confidence"].astype(str).str.lower() == "medium"
].copy()

low = provisional[
    provisional["bulk_confidence"].astype(str).str.lower() == "low"
].copy()

# ---------------------------------------------------------
# Prefer diversity across predicted intents
# ---------------------------------------------------------

def balanced_sample(data, n):
    if len(data) <= n:
        return data.copy()

    groups = []

    intents = data["bulk_intent"].dropna().unique()

    per_intent = max(1, n // len(intents))

    for intent in intents:
        subset = data[data["bulk_intent"] == intent]

        if len(subset) > 0:
            groups.append(
                subset.sample(
                    min(per_intent, len(subset)),
                    random_state=42
                )
            )

    result = pd.concat(groups).drop_duplicates("tweet_id")

    if len(result) < n:
        remaining = data[
            ~data["tweet_id"].isin(result["tweet_id"])
        ]

        extra = remaining.sample(
            min(n - len(result), len(remaining)),
            random_state=42
        )

        result = pd.concat([result, extra])

    return result.head(n)


# ---------------------------------------------------------
# Target distribution
#
# 10 high-confidence
# 15 medium-confidence
# 15 low-confidence
# 10 other/unclear or context-heavy
# ---------------------------------------------------------

high_sample = balanced_sample(high, 10)

medium_sample = balanced_sample(medium, 15)

low_sample = balanced_sample(low, 15)

# Explicitly include uncertain cases
uncertain_file = Path("golden_set/bulk_context_uncertain.csv")

if uncertain_file.exists():

    uncertain = pd.read_csv(uncertain_file)

    uncertain = uncertain[
        ~uncertain["tweet_id"].isin(human["tweet_id"])
    ]

    uncertain_sample = uncertain.sample(
        min(10, len(uncertain)),
        random_state=42
    )

else:
    uncertain_sample = low.sample(
        min(10, len(low)),
        random_state=42
    )


# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------

validation = pd.concat([
    high_sample,
    medium_sample,
    low_sample,
    uncertain_sample
], ignore_index=True)

# Remove duplicates
validation = validation.drop_duplicates("tweet_id")

# If more than 50, trim
if len(validation) > 50:
    validation = validation.sample(
        50,
        random_state=42
    )

# Add human-label columns
if "human_intent" not in validation.columns:
    validation["human_intent"] = ""

if "human_notes" not in validation.columns:
    validation["human_notes"] = ""

validation["human_intent"] = validation["human_intent"].fillna("")
validation["human_notes"] = validation["human_notes"].fillna("")

validation["validation_status"] = "needs_human_review"

# Shuffle final set
validation = validation.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
validation.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("HUMAN VALIDATION SET CREATED")
print("=" * 70)

print(f"Validation rows: {len(validation)}")

print()
print("Predicted intent distribution:")
print(validation["bulk_intent"].value_counts())

print()
print("Confidence distribution:")
print(validation["bulk_confidence"].value_counts())

print()
print(f"Output: {OUTPUT_FILE}")