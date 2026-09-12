import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

INPUT_FILE = "golden_set/human_validation_50_reviewed.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only genuinely human-reviewed rows
df = df[
    df["human_intent"].notna()
    & (df["human_intent"].astype(str).str.strip() != "")
].copy()

human = df["human_intent"].astype(str).str.strip()
bulk = df["bulk_intent"].astype(str).str.strip()

print("=" * 70)
print("BULK LABELING vs HUMAN LABELING EVALUATION")
print("=" * 70)

print(f"\nHuman-reviewed examples: {len(df)}")

# --------------------------------------------------
# Accuracy
# --------------------------------------------------

accuracy = accuracy_score(human, bulk)

print("\n" + "=" * 70)
print("ACCURACY")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        human,
        bulk,
        zero_division=0
    )
)

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

labels = sorted(set(human) | set(bulk))

cm = confusion_matrix(
    human,
    bulk,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {x}" for x in labels],
    columns=[f"Predicted: {x}" for x in labels]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm_df.to_string())

# --------------------------------------------------
# Disagreements
# --------------------------------------------------

df["correct"] = human == bulk

disagreements = df[~df["correct"]].copy()

print("\n" + "=" * 70)
print("DISAGREEMENTS")
print("=" * 70)

print(f"\nCorrect: {df['correct'].sum()}")
print(f"Incorrect: {len(disagreements)}")

if len(disagreements) > 0:

    print("\nDetailed disagreements:\n")

    for _, row in disagreements.iterrows():

        print("-" * 70)

        print(f"Tweet ID: {row['tweet_id']}")

        print(f"\nCustomer:")
        print(row["text"])

        print(f"\nHuman label: {row['human_intent']}")
        print(f"Bulk label:  {row['bulk_intent']}")

        if "bulk_reason" in row:
            print(f"Reason:      {row['bulk_reason']}")

# --------------------------------------------------
# Confidence analysis
# --------------------------------------------------

if "bulk_confidence" in df.columns:

    print("\n" + "=" * 70)
    print("ACCURACY BY BULK CONFIDENCE")
    print("=" * 70)

    confidence_accuracy = (
        df.groupby("bulk_confidence")["correct"]
        .agg(["count", "sum", "mean"])
    )

    confidence_accuracy["accuracy_percent"] = (
        confidence_accuracy["mean"] * 100
    )

    print(confidence_accuracy)

# --------------------------------------------------
# Save disagreements
# --------------------------------------------------

if len(disagreements) > 0:

    output_file = "evaluation/results/bulk_label_disagreements.csv"

    disagreements[
        [
            "tweet_id",
            "text",
            "human_intent",
            "bulk_intent",
            "bulk_confidence",
            "bulk_reason"
        ]
    ].to_csv(output_file, index=False)

    print(
        f"\nDisagreements saved to: {output_file}"
    )

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)