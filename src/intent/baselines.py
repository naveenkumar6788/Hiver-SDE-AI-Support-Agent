import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


GOLDEN_PATH = "golden_set/golden_independently_verified.csv"


def main():

    print("=" * 80)
    print("PROPER TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 80)

    df = pd.read_csv(GOLDEN_PATH)

    # Clean text
    df["text"] = df["text"].fillna("").astype(str)
    df["verified_intent"] = df["verified_intent"].astype(str)
    df["conversation_id"] = (
    df["conversation_id"]
    .fillna("missing_conversation")
    .astype(str)
    .str.strip()
)

    print(f"\nTotal labeled examples: {len(df)}")
    print(f"Unique conversations: {df['conversation_id'].nunique()}")

    print("\nIntent distribution:")
    print(df["verified_intent"].value_counts())

    X = df["text"]
    y = df["verified_intent"]
    groups = df["conversation_id"]

    # 5-fold grouped + stratified CV
    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    all_true = []
    all_pred = []

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        overlap = train_groups.intersection(test_groups)

        print("\n" + "=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)

        print(f"Train examples: {len(train_idx)}")
        print(f"Test examples : {len(test_idx)}")
        print(f"Train conversations: {len(train_groups)}")
        print(f"Test conversations : {len(test_groups)}")
        print(f"Conversation overlap: {len(overlap)}")

        # ------------------------------------------------------------------
        # Majority class baseline
        # ------------------------------------------------------------------

        majority_class = y_train.value_counts().idxmax()

        majority_pred = np.full(
            len(y_test),
            majority_class
        )

        majority_accuracy = accuracy_score(
            y_test,
            majority_pred
        )

        majority_f1 = f1_score(
            y_test,
            majority_pred,
            average="macro",
            zero_division=0
        )

        print("\nMajority baseline:")
        print(f"Majority class: {majority_class}")
        print(f"Accuracy: {majority_accuracy:.4f}")
        print(f"Macro F1: {majority_f1:.4f}")

        # ------------------------------------------------------------------
        # TF-IDF + Logistic Regression
        # ------------------------------------------------------------------

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=50000
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ])

        print("\nTraining TF-IDF + Logistic Regression...")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        print("Training complete.")

        print("\nTF-IDF + Logistic Regression:")
        print(f"Accuracy:     {accuracy:.4f}")
        print(f"Macro F1:     {macro_f1:.4f}")
        print(f"Weighted F1:  {weighted_f1:.4f}")

        fold_results.append({
            "fold": fold,
            "majority_accuracy": majority_accuracy,
            "majority_macro_f1": majority_f1,
            "tfidf_accuracy": accuracy,
            "tfidf_macro_f1": macro_f1,
            "tfidf_weighted_f1": weighted_f1
        })

        all_true.extend(y_test)
        all_pred.extend(predictions)

    # ----------------------------------------------------------------------
    # Overall results
    # ----------------------------------------------------------------------

    results = pd.DataFrame(fold_results)

    print("\n\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)

    print("\nPer-fold results:")
    print(results.to_string(index=False))

    print("\nAverage results:")

    print(
        f"\nMajority Accuracy: "
        f"{results['majority_accuracy'].mean():.4f}"
    )

    print(
        f"Majority Macro F1: "
        f"{results['majority_macro_f1'].mean():.4f}"
    )

    print(
        f"\nTF-IDF Accuracy: "
        f"{results['tfidf_accuracy'].mean():.4f}"
    )

    print(
        f"TF-IDF Macro F1: "
        f"{results['tfidf_macro_f1'].mean():.4f}"
    )

    print(
        f"TF-IDF Weighted F1: "
        f"{results['tfidf_weighted_f1'].mean():.4f}"
    )

    # ----------------------------------------------------------------------
    # Overall out-of-fold classification report
    # ----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OUT-OF-FOLD CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            all_true,
            all_pred,
            zero_division=0
        )
    )

    # Save results
    results.to_csv(
        "evaluation/results/baseline_results.csv",
        index=False
    )

    print(
        "\nSaved results to:"
        " evaluation/results/baseline_results.csv"
    )


if __name__ == "__main__":
    main()