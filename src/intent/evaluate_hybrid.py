"""
src/intent/evaluate_hybrid.py

Rigorous conversation-grouped 3-fold cross-validation of intent classification models:
1. Majority Baseline
2. TF-IDF + Logistic Regression Baseline
3. Hybrid Intent Classifier (Rules + TF-IDF Logistic Regression fallback)

Evaluation integrity:
- Uses StratifiedGroupKFold on conversation_id to prevent intra-thread leakage between train and test.
- Reports out-of-fold metrics across held-out conversation folds.
- Records per-class precision, recall, F1, and confusion matrix.
"""

import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intent.classifier import IntentClassifier, clean_text

# Suppress minor scikit-learn warnings about classes with few samples in GroupKFold
warnings.filterwarnings("ignore", category=UserWarning)

GOLDEN_PATH = PROJECT_ROOT / "golden_set" / "golden_independently_verified.csv"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def normalize_id(value, fallback=""):
    """Normalize tweet and conversation IDs to strings without float artifacts."""
    if pd.isna(value) or str(value).strip() == "" or str(value).strip().lower() == "nan":
        return fallback
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_tfidf_baseline():
    """Construct the standard TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_features=50000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_PATH}")

    df = pd.read_csv(GOLDEN_PATH)

    # Validate required columns
    required_cols = ["tweet_id", "text", "human_intent"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in golden set: {col}")

    # Clean text and strings
    df["tweet_id"] = df["tweet_id"].apply(normalize_id)
    df["text"] = df["text"].fillna("").astype(str)
    df["human_intent"] = df["human_intent"].fillna("").astype(str)

    # Normalize conversation IDs; if missing, assign unique ID per tweet to avoid artificial clustering
    conv_col = "conversation_id" if "conversation_id" in df.columns else "tweet_id"
    df["normalized_conversation_id"] = [
        normalize_id(c, fallback=f"conv_missing_{t}")
        for c, t in zip(df[conv_col], df["tweet_id"])
    ]

    total_examples = len(df)
    unique_conversations = df["normalized_conversation_id"].nunique()
    intents_list = sorted(df["human_intent"].unique())
    num_intents = len(intents_list)

    print("=" * 60)
    print("HYBRID INTENT CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"\nDataset: {GOLDEN_PATH.name}")
    print(f"Total examples: {total_examples}")
    print(f"Unique conversations: {unique_conversations}")
    print(f"Number of intents: {num_intents}")

    X = df["text"]
    y = df["human_intent"]
    groups = df["normalized_conversation_id"]

    # 3-Fold Conversation-Grouped & Stratified Split
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    fold_metrics = []
    oof_predictions = []

    print("\nExecuting 3-Fold Conversation-Grouped Cross-Validation...")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups.intersection(test_groups)
        assert len(overlap) == 0, f"Fold {fold} has conversation leakage between train and test!"

        # -----------------------------------------------------------------
        # Model A: Majority Class Baseline
        # -----------------------------------------------------------------
        majority_class = train_df["human_intent"].value_counts().idxmax()
        maj_preds = [majority_class] * len(test_df)

        maj_acc = accuracy_score(test_df["human_intent"], maj_preds)
        maj_mf1 = f1_score(test_df["human_intent"], maj_preds, average="macro", zero_division=0)
        maj_wf1 = f1_score(test_df["human_intent"], maj_preds, average="weighted", zero_division=0)

        # -----------------------------------------------------------------
        # Model B: TF-IDF + Logistic Regression Baseline
        # -----------------------------------------------------------------
        tfidf_model = build_tfidf_baseline()
        tfidf_model.fit(train_df["text"], train_df["human_intent"])
        tfidf_preds = tfidf_model.predict(test_df["text"])

        tfidf_acc = accuracy_score(test_df["human_intent"], tfidf_preds)
        tfidf_mf1 = f1_score(test_df["human_intent"], tfidf_preds, average="macro", zero_division=0)
        tfidf_wf1 = f1_score(test_df["human_intent"], tfidf_preds, average="weighted", zero_division=0)

        # -----------------------------------------------------------------
        # Model C: Hybrid Classifier (Rules + TF-IDF fallback)
        # -----------------------------------------------------------------
        hybrid_model = IntentClassifier(data_path=train_df, verbose=False)

        hybrid_preds = []
        hybrid_methods = []
        hybrid_confs = []

        for text in test_df["text"]:
            pred_dict = hybrid_model.predict(text, use_rules=True)
            hybrid_preds.append(pred_dict["intent"])
            hybrid_methods.append(pred_dict.get("method", "unknown"))
            hybrid_confs.append(pred_dict.get("confidence", 0.0))

        hyb_acc = accuracy_score(test_df["human_intent"], hybrid_preds)
        hyb_mf1 = f1_score(test_df["human_intent"], hybrid_preds, average="macro", zero_division=0)
        hyb_wf1 = f1_score(test_df["human_intent"], hybrid_preds, average="weighted", zero_division=0)

        fold_metrics.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "majority_accuracy": maj_acc,
            "majority_macro_f1": maj_mf1,
            "majority_weighted_f1": maj_wf1,
            "tfidf_accuracy": tfidf_acc,
            "tfidf_macro_f1": tfidf_mf1,
            "tfidf_weighted_f1": tfidf_wf1,
            "hybrid_accuracy": hyb_acc,
            "hybrid_macro_f1": hyb_mf1,
            "hybrid_weighted_f1": hyb_wf1,
        })

        # Save out-of-fold sample predictions
        for i, idx in enumerate(test_idx):
            oof_predictions.append({
                "tweet_id": df.iloc[idx]["tweet_id"],
                "conversation_id": df.iloc[idx]["normalized_conversation_id"],
                "text": df.iloc[idx]["text"],
                "true_intent": df.iloc[idx]["human_intent"],
                "majority_pred": maj_preds[i],
                "tfidf_pred": tfidf_preds[i],
                "hybrid_pred": hybrid_preds[i],
                "hybrid_method": hybrid_methods[i],
                "hybrid_confidence": hybrid_confs[i],
                "fold": fold,
            })

    # Summary calculations
    metrics_df = pd.DataFrame(fold_metrics)
    oof_df = pd.DataFrame(oof_predictions)

    # Average metrics across folds
    maj_acc_mean = metrics_df["majority_accuracy"].mean()
    maj_mf1_mean = metrics_df["majority_macro_f1"].mean()
    maj_wf1_mean = metrics_df["majority_weighted_f1"].mean()

    tfidf_acc_mean = metrics_df["tfidf_accuracy"].mean()
    tfidf_mf1_mean = metrics_df["tfidf_macro_f1"].mean()
    tfidf_wf1_mean = metrics_df["tfidf_weighted_f1"].mean()

    hyb_acc_mean = metrics_df["hybrid_accuracy"].mean()
    hyb_mf1_mean = metrics_df["hybrid_macro_f1"].mean()
    hyb_wf1_mean = metrics_df["hybrid_weighted_f1"].mean()

    imp_acc = hyb_acc_mean - tfidf_acc_mean
    imp_mf1 = hyb_mf1_mean - tfidf_mf1_mean
    imp_wf1 = hyb_wf1_mean - tfidf_wf1_mean

    # -----------------------------------------------------------------
    # Print Required Terminal Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Conversation-Grouped 3-Fold CV)")
    print("=" * 60)

    print("\nMajority Baseline")
    print(f"Accuracy:    {maj_acc_mean:.4f}")
    print(f"Macro F1:    {maj_mf1_mean:.4f}")
    print(f"Weighted F1: {maj_wf1_mean:.4f}")

    print("\nTF-IDF + Logistic Regression")
    print(f"Accuracy:    {tfidf_acc_mean:.4f}")
    print(f"Macro F1:    {tfidf_mf1_mean:.4f}")
    print(f"Weighted F1: {tfidf_wf1_mean:.4f}")

    print("\nHybrid Classifier")
    print(f"Accuracy:    {hyb_acc_mean:.4f}")
    print(f"Macro F1:    {hyb_mf1_mean:.4f}")
    print(f"Weighted F1: {hyb_wf1_mean:.4f}")

    print("\nImprovement over TF-IDF:")
    print(f"Accuracy:    {'+' if imp_acc >= 0 else ''}{imp_acc:.4f} ({imp_acc * 100:+.2f}%)")
    print(f"Macro F1:    {'+' if imp_mf1 >= 0 else ''}{imp_mf1:.4f} ({imp_mf1 * 100:+.2f}%)")
    print(f"Weighted F1: {'+' if imp_wf1 >= 0 else ''}{imp_wf1:.4f} ({imp_wf1 * 100:+.2f}%)")

    # -----------------------------------------------------------------
    # Hybrid Breakdown: Rule vs ML Decisions
    # -----------------------------------------------------------------
    method_counts = oof_df["hybrid_method"].value_counts()
    print("\n" + "-" * 60)
    print("HYBRID DECISION METHOD BREAKDOWN (Out-of-Fold)")
    print("-" * 60)
    for method, count in method_counts.items():
        acc = accuracy_score(
            oof_df[oof_df["hybrid_method"] == method]["true_intent"],
            oof_df[oof_df["hybrid_method"] == method]["hybrid_pred"],
        )
        print(f"  {method:<18}: {count:>3} cases ({count / total_examples * 100:>5.1f}%) | Subset Accuracy: {acc:.4f}")

    # -----------------------------------------------------------------
    # Out-of-Fold Classification Report for Hybrid Classifier
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OUT-OF-FOLD CLASSIFICATION REPORT: HYBRID CLASSIFIER")
    print("=" * 60)
    report_text = classification_report(
        oof_df["true_intent"],
        oof_df["hybrid_pred"],
        labels=intents_list,
        zero_division=0,
    )
    print(report_text)

    # -----------------------------------------------------------------
    # Save Results
    # -----------------------------------------------------------------
    # 1. Summary comparison table
    summary_comparison = pd.DataFrame([
        {
            "model": "Majority Baseline",
            "accuracy": round(maj_acc_mean, 4),
            "macro_f1": round(maj_mf1_mean, 4),
            "weighted_f1": round(maj_wf1_mean, 4),
        },
        {
            "model": "TF-IDF + Logistic Regression",
            "accuracy": round(tfidf_acc_mean, 4),
            "macro_f1": round(tfidf_mf1_mean, 4),
            "weighted_f1": round(tfidf_wf1_mean, 4),
        },
        {
            "model": "Hybrid Classifier",
            "accuracy": round(hyb_acc_mean, 4),
            "macro_f1": round(hyb_mf1_mean, 4),
            "weighted_f1": round(hyb_wf1_mean, 4),
        },
        {
            "model": "Improvement (Hybrid - TF-IDF)",
            "accuracy": round(imp_acc, 4),
            "macro_f1": round(imp_mf1, 4),
            "weighted_f1": round(imp_wf1, 4),
        },
    ])

    results_file = RESULTS_DIR / "hybrid_results.csv"
    summary_comparison.to_csv(results_file, index=False)
    print(f"Saved model comparison to: {results_file}")

    # 2. Detailed fold metrics table
    fold_file = RESULTS_DIR / "hybrid_fold_metrics.csv"
    metrics_df.to_csv(fold_file, index=False)
    print(f"Saved fold metrics to: {fold_file}")

    # 3. Class-by-class report dictionary -> CSV
    report_dict = classification_report(
        oof_df["true_intent"],
        oof_df["hybrid_pred"],
        labels=intents_list,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "intent"})

    class_report_file = RESULTS_DIR / "hybrid_classification_report.csv"
    report_df.to_csv(class_report_file, index=False)
    print(f"Saved classification report to: {class_report_file}")

    # 4. Confusion matrix -> CSV
    cm = confusion_matrix(
        oof_df["true_intent"],
        oof_df["hybrid_pred"],
        labels=intents_list,
    )
    cm_df = pd.DataFrame(cm, index=intents_list, columns=intents_list)
    cm_file = RESULTS_DIR / "hybrid_confusion_matrix.csv"
    cm_df.to_csv(cm_file)
    print(f"Saved confusion matrix to: {cm_file}")

    # 5. Out-of-fold sample predictions -> CSV
    oof_file = RESULTS_DIR / "hybrid_oof_predictions.csv"
    oof_df.to_csv(oof_file, index=False)
    print(f"Saved out-of-fold predictions to: {oof_file}")

    # Evaluation Integrity Notice
    print("\n" + "=" * 60)
    print("EVALUATION INTEGRITY NOTE")
    print("=" * 60)
    print(
        "1. Conversation-grouped 3-fold cross-validation prevents data leakage across threads.\n"
        "2. All reported numbers reflect out-of-fold predictions on held-out conversations.\n"
        "3. Limitation: Rule patterns were engineered using domain knowledge and development\n"
        "   error observations on this corpus; while out-of-fold CV confirms generalization\n"
        "   across conversations, fully unbiased validation would require a completely new,\n"
        "   unseen test split."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
