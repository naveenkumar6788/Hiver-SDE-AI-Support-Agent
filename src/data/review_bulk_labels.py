import pandas as pd
from pathlib import Path

CANDIDATE_FILE = Path("golden_set/golden_candidates.csv")
LABELED_FILE = Path("golden_set/golden_labeled.csv")
REVIEW_FILE = Path("golden_set/bulk_review.csv")


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def first_non_empty(row, columns):
    for column in columns:
        if column in row.index:
            value = clean_value(row[column])
            if value:
                return value
    return ""


def main():

    print("Loading files...")

    candidates = pd.read_csv(CANDIDATE_FILE)
    labeled = pd.read_csv(LABELED_FILE)

    print(f"Candidates loaded: {len(candidates):,}")
    print(f"Labeled loaded:   {len(labeled):,}")

    print("\nLabeled columns:")
    print(labeled.columns.tolist())

    # ---------------------------------------------------------
    # Recover the actual bulk label
    # ---------------------------------------------------------

    possible_intent_columns = [
        "golden_intent",
        "golden_intent_y",
        "golden_intent_x"
    ]

    possible_note_columns = [
        "label_notes",
        "label_notes_y",
        "label_notes_x"
    ]

    # Build clean dataframe using tweet_id
    labeled_clean = pd.DataFrame()

    labeled_clean["tweet_id"] = labeled["tweet_id"]

    labeled_clean["bulk_intent"] = labeled.apply(
        lambda row: first_non_empty(
            row,
            possible_intent_columns
        ),
        axis=1
    )

    labeled_clean["bulk_notes"] = labeled.apply(
        lambda row: first_non_empty(
            row,
            possible_note_columns
        ),
        axis=1
    )

    if "label_status" in labeled.columns:
        labeled_clean["label_status"] = (
            labeled["label_status"]
        )
    else:
        labeled_clean["label_status"] = "bulk_labeled"

    # Remove duplicate tweet IDs
    labeled_clean = labeled_clean.drop_duplicates(
        subset=["tweet_id"],
        keep="last"
    )

    # ---------------------------------------------------------
    # Merge with candidates
    # ---------------------------------------------------------

    merged = candidates.merge(
        labeled_clean,
        on="tweet_id",
        how="left"
    )

    # ---------------------------------------------------------
    # Show recovered labels
    # ---------------------------------------------------------

    print("\nRecovered bulk label sample:")

    print(
        merged[
            ["candidate_group", "bulk_intent"]
        ].head(10).to_string(index=False)
    )

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    merged["candidate_group_clean"] = (
        merged["candidate_group"]
        .apply(clean_value)
    )

    merged["bulk_intent_clean"] = (
        merged["bulk_intent"]
        .apply(clean_value)
    )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    merged["group_matches_label"] = (
        merged["candidate_group_clean"]
        == merged["bulk_intent_clean"]
    )

    # ---------------------------------------------------------
    # Find mismatches
    # ---------------------------------------------------------

    review = merged[
        ~merged["group_matches_label"]
    ].copy()

    # Include missing labels
    missing_labels = merged[
        merged["bulk_intent_clean"] == ""
    ].copy()

    review = pd.concat(
        [review, missing_labels],
        ignore_index=True
    )

    review = review.drop_duplicates(
        subset=["tweet_id"]
    )

    # ---------------------------------------------------------
    # Useful output columns
    # ---------------------------------------------------------

    columns = [
        "tweet_id",
        "text",
        "candidate_group",
        "bulk_intent",
        "bulk_notes",
        "label_status",
        "conversation_id",
        "created_at"
    ]

    columns = [
        col for col in columns
        if col in review.columns
    ]

    review = review[columns]

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    review.to_csv(
        REVIEW_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("BULK LABEL REVIEW")
    print("=" * 60)

    print(
        f"Total candidates:       {len(merged):,}"
    )

    print(
        f"Matching labels:        "
        f"{merged['group_matches_label'].sum():,}"
    )

    print(
        f"Potential mismatches:   {len(review):,}"
    )

    print("\nRecovered bulk label distribution:")

    print(
        merged["bulk_intent"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nCandidate group vs bulk label:")

    print(
        pd.crosstab(
            merged["candidate_group"],
            merged["bulk_intent"]
        ).to_string()
    )

    print()
    print("Review file saved:")
    print(REVIEW_FILE)


if __name__ == "__main__":
    main()