"""
src/evaluation/human_annotation_template.py

Interactive / Command-Line Tool for Human Evaluation of Customer Support Replies.
Allows human evaluators to independently rate the 50 blind evaluation cases across 6 dimensions.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def prompt_score(dim_name: str, description: str) -> int:
    while True:
        val = input(f"  {dim_name} (1-5, {description}): ").strip()
        if val in {"1", "2", "3", "4", "5"}:
            return int(val)
        if val.lower() in {"q", "quit", "exit"}:
            print("Exiting interactive annotation.")
            sys.exit(0)
        print("    [Error] Please enter an integer from 1 to 5 (or 'q' to quit).")


def run_interactive_annotation(
    blind_file_path: str = "golden_set/human_judge_50_blind.csv",
    full_file_path: str = "golden_set/human_judge_50.csv",
    start_from_unrated: bool = True,
):
    blind_file = PROJECT_ROOT / blind_file_path
    full_file = PROJECT_ROOT / full_file_path

    if not blind_file.exists():
        raise FileNotFoundError(f"Blind evaluation file not found: {blind_file}")

    df = pd.read_csv(blind_file)
    dims = [
        ("human_correctness", "Correctness", "1=Harmful/Wrong, 5=Accurate/Safe"),
        ("human_groundedness", "Groundedness", "1=Hallucinated, 5=Fully Grounded/Safe Refusal"),
        ("human_relevance", "Relevance", "1=Irrelevant/Wrong Issue, 5=Directly Relevant"),
        ("human_completeness", "Completeness", "1=Misses Core Issue, 5=Complete/Proper Diagnostics"),
        ("human_tone", "Tone", "1=Unprofessional, 5=Empathetic/Polite"),
        ("human_unsupported_claims", "Unsupported Claims", "1=Substantial Hallucinations, 5=Zero Unsupported Claims"),
    ]

    rated_count = df["human_correctness"].dropna().count()
    total_count = len(df)
    print("=" * 80)
    print(f"HUMAN EVALUATION WORKFLOW ({rated_count}/{total_count} cases rated)")
    print("=" * 80)
    print("Press Ctrl+C or type 'q' at any prompt to save and quit.")
    print("-" * 80)

    for idx, row in df.iterrows():
        eval_id = row["evaluation_id"]
        is_rated = pd.notna(row["human_correctness"])

        if start_from_unrated and is_rated:
            continue

        print(f"\n[{idx + 1}/{total_count}] CASE ID: {eval_id} (Tweet ID: {row['tweet_id']})")
        print(f"Customer Message : {row['customer_message']}")
        print(f"Expected Intent  : {row['expected_intent']}")
        print(f"Evidence Used    : {row['evidence_used']}")
        if pd.notna(row['evidence_text']) and str(row['evidence_text']).strip():
            print(f"Evidence Text    : {row['evidence_text']}")
        print(f"Generated Reply  : {row['generated_reply']}")
        print(f"Escalation       : {row['escalation_decision']}")
        print("-" * 40)

        for col, label, desc in dims:
            score = prompt_score(label, desc)
            df.at[idx, col] = score

        # Save immediately to preserve progress
        df.to_csv(blind_file, index=False)

        # Sync to full_file if it exists
        if full_file.exists():
            full_df = pd.read_csv(full_file)
            for col, _, _ in dims:
                full_df.at[idx, col] = df.at[idx, col]
            full_df.to_csv(full_file, index=False)

        print(f"Saved {eval_id}. Progress: {idx + 1}/{total_count}")

    print("\nAnnotation completed for all cases!")


def validate_human_ratings(file_path: str = "golden_set/human_judge_50_blind.csv"):
    path = PROJECT_ROOT / file_path
    if not path.exists():
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    dims = [
        "human_correctness",
        "human_groundedness",
        "human_relevance",
        "human_completeness",
        "human_tone",
        "human_unsupported_claims",
    ]

    total = len(df)
    rated_per_dim = {d: df[d].dropna().count() for d in dims}
    min_rated = min(rated_per_dim.values())

    print("=" * 60)
    print(f"HUMAN RATING VALIDATION: {path.name}")
    print("=" * 60)
    print(f"Total Cases: {total}")
    for d, c in rated_per_dim.items():
        print(f"  {d:<28}: {c}/{total}")

    if min_rated == 0:
        print("\nStatus: NO HUMAN RATINGS RECORDED (0/50)")
    elif min_rated < total:
        print(f"\nStatus: PARTIAL RATINGS ({min_rated}/{total} complete)")
    else:
        print(f"\nStatus: COMPLETE (All {total} cases rated)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human evaluation rating template & validator.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive CLI rating session")
    parser.add_argument("--validate", action="store_true", help="Check status of existing ratings")
    parser.add_argument("--blind_file", default="golden_set/human_judge_50_blind.csv", help="Path to blind CSV")
    parser.add_argument("--full_file", default="golden_set/human_judge_50.csv", help="Path to full CSV")

    args = parser.parse_args()

    if args.interactive:
        run_interactive_annotation(args.blind_file, args.full_file)
    else:
        validate_human_ratings(args.blind_file)
