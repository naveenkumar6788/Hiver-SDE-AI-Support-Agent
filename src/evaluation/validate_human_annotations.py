"""
src/evaluation/validate_human_annotations.py

Strict validator for genuine human response-quality annotations.
Checks golden_set/human_annotation_form.csv.

Rules:
- Requires exactly 50 unique evaluation IDs (eval_01 through eval_50).
- Verifies all six human rating columns contain integers 1-5.
- Verifies annotation_source column is present and set to 'human' for all rows.
  (model_assisted or missing annotation_source → NOT_YET_VERIFIED)
- Detects missing values and duplicate evaluation IDs.
- Reports exactly which rows are incomplete or invalid.
- Never generates, infers, or modifies any ratings.

Outputs:
- NOT_YET_VERIFIED if any human rating is missing, invalid, or annotation_source != 'human'.
- VERIFIED only when all 50 cases are fully, validly, and genuinely human-rated.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RATING_DIMENSIONS = [
    "human_correctness",
    "human_groundedness",
    "human_relevance",
    "human_completeness",
    "human_tone",
    "human_unsupported_claims",
]


def validate_annotation_file(
    file_path: str = "golden_set/human_annotation_form.csv",
) -> Tuple[str, Dict[str, any]]:
    path = PROJECT_ROOT / file_path
    print("=" * 80)
    print("GENUINE HUMAN ANNOTATION VALIDATOR")
    print(f"Target File: {path}")
    print("=" * 80)

    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        print("\nResult: NOT_YET_VERIFIED")
        return "NOT_YET_VERIFIED", {"error": "file_not_found"}

    df = pd.read_csv(path)
    total_rows = len(df)
    print(f"Total Rows Found: {total_rows}")

    issues: List[str] = []
    incomplete_rows: List[Tuple[str, List[str]]] = []
    invalid_rows: List[Tuple[str, str, any]] = []

    # 1. Check evaluation_id column
    if "evaluation_id" not in df.columns:
        issues.append("Missing required 'evaluation_id' column.")
        print("\nResult: NOT_YET_VERIFIED")
        return "NOT_YET_VERIFIED", {"issues": issues}

    # 2. Check for exactly 50 rows
    if total_rows != 50:
        issues.append(f"Expected exactly 50 rows, but found {total_rows}.")

    # 3. Check for duplicates
    dup_ids = df[df["evaluation_id"].duplicated()]["evaluation_id"].tolist()
    if dup_ids:
        issues.append(f"Duplicate evaluation_ids found: {dup_ids}")

    # 4. Check presence of eval_01 through eval_50
    expected_ids = [f"eval_{i:02d}" for i in range(1, 51)]
    actual_ids = df["evaluation_id"].tolist()
    missing_ids = [eid for eid in expected_ids if eid not in actual_ids]
    if missing_ids:
        issues.append(f"Missing expected evaluation IDs: {missing_ids}")

    # 5. Check all 6 rating dimensions
    for dim in RATING_DIMENSIONS:
        if dim not in df.columns:
            issues.append(f"Missing required dimension column: {dim}")

    if issues:
        print("\n[STRUCTURAL ISSUES DETECTED]")
        for iss in issues:
            print(f"  - {iss}")

    # 5b. Check annotation_source — must be 'human' for all rows
    # This is the SCIENTIFIC INTEGRITY check.
    # structural validity (valid 1-5 integers) does NOT prove human authorship.
    if "annotation_source" not in df.columns:
        issues.append(
            "Missing 'annotation_source' column. "
            "Cannot verify whether ratings are genuinely human. "
            "Set annotation_source='human' for each row that has been independently rated by a human."
        )
        print("\n[SCIENTIFIC INTEGRITY] annotation_source column is missing.")
        print("  Ratings cannot be verified as human-authored.")
        print("  NOT_YET_VERIFIED — add annotation_source='human' for genuine human ratings.")
        print("-" * 80)
        print("VALIDATION RESULT: NOT_YET_VERIFIED")
        print("annotation_source column missing — cannot confirm human authorship.")
        print("-" * 80)
        return "NOT_YET_VERIFIED", {"issues": issues}
    else:
        non_human = df[df["annotation_source"] != "human"]
        if len(non_human) > 0:
            source_vals = non_human["annotation_source"].unique().tolist()
            issues.append(
                f"{len(non_human)}/50 rows have annotation_source != 'human' "
                f"(found: {source_vals}). "
                "These are not genuine independent human ratings."
            )
            print("\n[SCIENTIFIC INTEGRITY] annotation_source check FAILED.")
            print(f"  {len(non_human)}/50 rows annotated by: {source_vals}")
            print("  Structural CSV validity does NOT prove human authorship.")
            print("  To mark as genuinely human: set annotation_source='human' per row.")
            print("-" * 80)
            print("VALIDATION RESULT: NOT_YET_VERIFIED")
            print(f"Reason: annotation_source is '{source_vals[0] if source_vals else 'unknown'}', not 'human'.")
            print("-" * 80)
            return "NOT_YET_VERIFIED", {"issues": issues}

    complete_count = 0
    for idx, row in df.iterrows():
        eid = str(row.get("evaluation_id", f"row_{idx}"))
        missing_dims = []
        row_invalid = False

        for dim in RATING_DIMENSIONS:
            if dim not in df.columns:
                missing_dims.append(dim)
                continue

            val = row[dim]
            if pd.isna(val) or str(val).strip() == "":
                missing_dims.append(dim)
            else:
                try:
                    int_val = int(val)
                    if float(val) != int_val or int_val not in {1, 2, 3, 4, 5}:
                        invalid_rows.append((eid, dim, val))
                        row_invalid = True
                except (ValueError, TypeError):
                    invalid_rows.append((eid, dim, val))
                    row_invalid = True

        if missing_dims:
            incomplete_rows.append((eid, missing_dims))
        elif not row_invalid:
            complete_count += 1

    # 7. Summary reporting
    print(f"\nCompletion Progress : {complete_count}/50 cases fully rated")
    print(f"Incomplete Cases    : {len(incomplete_rows)}/50")
    print(f"Invalid Score Cases : {len(invalid_rows)}")

    if incomplete_rows:
        print("\nIncomplete Rows (Sample up to 10):")
        for eid, mdims in incomplete_rows[:10]:
            print(f"  - {eid}: Missing {mdims}")
        if len(incomplete_rows) > 10:
            print(f"  ... and {len(incomplete_rows) - 10} more incomplete rows.")

    if invalid_rows:
        print("\nInvalid Rows (Non-integer or out of 1-5 bounds):")
        for eid, dim, val in invalid_rows[:10]:
            print(f"  - {eid}: {dim} = '{val}' (must be integer 1-5)")

    print("-" * 80)
    if not issues and complete_count == 50 and not incomplete_rows and not invalid_rows:
        print("VALIDATION RESULT: VERIFIED")
        print("All 50 cases contain valid, genuine human annotations.")
        print("-" * 80)
        return "VERIFIED", {
            "complete_count": complete_count,
            "total": total_rows,
            "issues": issues,
            "incomplete_rows": incomplete_rows,
            "invalid_rows": invalid_rows,
        }
    else:
        print("VALIDATION RESULT: NOT_YET_VERIFIED")
        print("Human ratings are incomplete or missing. Calculations remain withheld.")
        print("-" * 80)
        return "NOT_YET_VERIFIED", {
            "complete_count": complete_count,
            "total": total_rows,
            "issues": issues,
            "incomplete_rows": incomplete_rows,
            "invalid_rows": invalid_rows,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate human evaluation CSV form.")
    parser.add_argument(
        "--file",
        default="golden_set/human_annotation_form.csv",
        help="Path to human annotation CSV file",
    )
    args = parser.parse_args()

    status, _ = validate_annotation_file(args.file)
    sys.exit(0 if status == "VERIFIED" else 1)
