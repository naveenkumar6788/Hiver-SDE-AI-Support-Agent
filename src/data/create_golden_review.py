import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

CANDIDATES_FILE = Path("golden_set/golden_candidates.csv")
BULK_REVIEW_FILE = Path("golden_set/bulk_review.csv")
OUTPUT_FILE = Path("golden_set/golden_review_200.csv")

TARGET_SIZE = 200
RANDOM_SEED = 42

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
    "other_unclear",
]

# ============================================================
# LOAD ORIGINAL 360 CANDIDATES
# ============================================================

print("Loading original candidates...")

candidates = pd.read_csv(CANDIDATES_FILE)

print(f"Candidates loaded: {len(candidates):,}")

# ============================================================
# LOAD BULK REVIEW
# ============================================================

print("Loading bulk review...")

bulk = pd.read_csv(BULK_REVIEW_FILE)

print(f"Bulk-review rows: {len(bulk):,}")

# ============================================================
# CLEAN
# ============================================================

for df in [candidates, bulk]:

    df["tweet_id"] = (
        df["tweet_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

# Remove duplicate tweets
candidates = candidates.drop_duplicates(
    subset="tweet_id"
).copy()

bulk = bulk.drop_duplicates(
    subset="tweet_id"
).copy()

# ============================================================
# IDENTIFY MISMATCHES
# ============================================================

mismatch_ids = set(
    bulk["tweet_id"].astype(str)
)

candidates["is_bulk_mismatch"] = (
    candidates["tweet_id"].isin(mismatch_ids)
)

print("\nCandidate classification:")
print(
    candidates["is_bulk_mismatch"]
    .value_counts()
    .rename({
        True: "Potential mismatch",
        False: "Potential match"
    })
)

# ============================================================
# SAMPLE MISMATCHES FIRST
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

mismatches = candidates[
    candidates["is_bulk_mismatch"]
].copy()

matches = candidates[
    ~candidates["is_bulk_mismatch"]
].copy()

print(f"\nPotential mismatches: {len(mismatches)}")
print(f"Potential matches:    {len(matches)}")

# We want most of the final golden set to contain
# difficult / ambiguous examples.

mismatch_target = min(
    len(mismatches),
    158
)

mismatch_indices = rng.choice(
    mismatches.index.to_numpy(),
    size=mismatch_target,
    replace=False
)

selected_mismatches = mismatches.loc[
    mismatch_indices
].copy()

remaining_needed = TARGET_SIZE - len(
    selected_mismatches
)

# ============================================================
# FILL FROM MATCHING EXAMPLES
# ============================================================

print(f"\nMismatch examples selected: {len(selected_mismatches)}")
print(f"Additional examples needed:  {remaining_needed}")

if remaining_needed > 0:

    # Try to balance the remaining examples
    # across candidate groups.

    selected_matches = []

    per_intent = max(
        1,
        remaining_needed // len(INTENTS)
    )

    for intent in INTENTS:

        subset = matches[
            matches["candidate_group"] == intent
        ].copy()

        if len(subset) == 0:
            continue

        n = min(
            per_intent,
            len(subset)
        )

        indices = rng.choice(
            subset.index.to_numpy(),
            size=n,
            replace=False
        )

        selected_matches.append(
            subset.loc[indices]
        )

    if selected_matches:

        selected_matches_df = pd.concat(
            selected_matches,
            ignore_index=False
        )

    else:

        selected_matches_df = pd.DataFrame(
            columns=matches.columns
        )

    # If still not enough, randomly fill.
    if len(selected_matches_df) < remaining_needed:

        already_selected = set(
            selected_matches_df["tweet_id"].astype(str)
        )

        remaining_matches = matches[
            ~matches["tweet_id"].astype(str).isin(
                already_selected
            )
        ].copy()

        extra_needed = (
            remaining_needed
            - len(selected_matches_df)
        )

        extra_needed = min(
            extra_needed,
            len(remaining_matches)
        )

        if extra_needed > 0:

            extra_indices = rng.choice(
                remaining_matches.index.to_numpy(),
                size=extra_needed,
                replace=False
            )

            extra = remaining_matches.loc[
                extra_indices
            ].copy()

            selected_matches_df = pd.concat(
                [
                    selected_matches_df,
                    extra
                ],
                ignore_index=False
            )

else:

    selected_matches_df = pd.DataFrame(
        columns=candidates.columns
    )

# ============================================================
# COMBINE
# ============================================================

review_df = pd.concat(
    [
        selected_mismatches,
        selected_matches_df
    ],
    ignore_index=True
)

# Remove duplicates
review_df = review_df.drop_duplicates(
    subset="tweet_id"
)

# If somehow below target, fill from all unused candidates
if len(review_df) < TARGET_SIZE:

    selected_ids = set(
        review_df["tweet_id"].astype(str)
    )

    unused = candidates[
        ~candidates["tweet_id"].astype(str).isin(
            selected_ids
        )
    ].copy()

    extra_needed = TARGET_SIZE - len(review_df)

    if len(unused) > 0:

        extra_needed = min(
            extra_needed,
            len(unused)
        )

        extra_indices = rng.choice(
            unused.index.to_numpy(),
            size=extra_needed,
            replace=False
        )

        extra = unused.loc[
            extra_indices
        ].copy()

        review_df = pd.concat(
            [
                review_df,
                extra
            ],
            ignore_index=True
        )

# Shuffle final set
review_df = review_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# ============================================================
# HUMAN LABEL COLUMNS
# ============================================================

review_df["human_intent"] = ""
review_df["human_notes"] = ""
review_df["label_status"] = "needs_human_review"

# ============================================================
# REMOVE INTERNAL COLUMN
# ============================================================

if "is_bulk_mismatch" in review_df.columns:
    review_df = review_df.drop(
        columns=["is_bulk_mismatch"]
    )

# ============================================================
# FINAL COLUMNS
# ============================================================

preferred_columns = [
    "tweet_id",
    "created_at",
    "conversation_id",
    "text",
    "candidate_group",
    "golden_intent",
    "label_notes",
    "human_intent",
    "human_notes",
    "label_status",
]

columns = [
    col
    for col in preferred_columns
    if col in review_df.columns
]

review_df = review_df[columns]

# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

review_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GOLDEN REVIEW SET CREATED")
print("=" * 60)

print(f"Final rows: {len(review_df)}")
print(f"Output:     {OUTPUT_FILE}")

print("\nCandidate-group distribution:")
print(
    review_df["candidate_group"]
    .value_counts()
    .to_string()
)

print("\nGolden-intent distribution:")
print(
    review_df["golden_intent"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nColumns:")
print(list(review_df.columns))

print("\nIMPORTANT:")
print("golden_intent = bulk suggestion only")
print("human_intent  = final human label")
print("human_notes   = reason for human decision")

print("=" * 60)