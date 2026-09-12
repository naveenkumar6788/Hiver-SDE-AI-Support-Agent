"""
src/evaluation/validate_human_annotations.py

Strict validator for independent human response-quality annotations.
Default target: golden_set/human_evaluation_template.csv

Validation Invariants (Part 5):
1. Exactly 50 evaluation cases.
2. No duplicate case_id values.
3. No missing human_intent; intent must belong to the 12 allowed intents.
4. intent_correct must be 'YES' or 'NO'.
5. Every rating dimension (1-5) must be an integer between 1 and 5.
6. evidence_supported must be 'YES' or 'NO'.
7. overall_comment must be present and non-empty.
8. No unexpected or missing required columns.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALLOWED_INTENTS = {
    "app_problems",
    "app_store_downloads",
    "apple_id_icloud",
    "apple_music_itunes",
    "audio_speaker",
    "battery_charging",
    "calls_cellular",
    "device_hardware",
    "ios_update",
    "other_unclear",
    "screen_display",
    "wifi_connectivity",
}

REQUIRED_COLUMNS = [
    "case_id",
    "customer_message",
    "system_intent",
    "historical_customer_message",
    "historical_response",
    "system_reply",
    "system_decision",
    "system_escalation_reason",
    "human_intent",
    "intent_correct",
    "correctness_1_5",
    "groundedness_1_5",
    "relevance_1_5",
    "completeness_1_5",
    "tone_1_5",
    "unsupported_claims_1_5",
    "escalation_appropriateness_1_5",
    "evidence_supported",
    "overall_comment",
]

RATING_DIMENSIONS_NEW = [
    "correctness_1_5",
    "groundedness_1_5",
    "relevance_1_5",
    "completeness_1_5",
    "tone_1_5",
    "unsupported_claims_1_5",
    "escalation_appropriateness_1_5",
]

LEGACY_RATING_DIMENSIONS = [
    "human_correctness",
    "human_groundedness",
    "human_relevance",
    "human_completeness",
    "human_tone",
    "human_unsupported_claims",
]


def validate_human_annotations(
    file_path: str = "golden_set/human_evaluation_template.csv",
) -> Tuple[bool, List[str]]:
    """
    Validates independent human annotations against strict quality invariants (Part 5).
    Returns (is_valid, list_of_errors).
    """
    path = PROJECT_ROOT / file_path if not Path(file_path).is_absolute() else Path(file_path)
    errors: List[str] = []

    if not path.exists():
        errors.append(f"File not found: {path}")
        return False, errors

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Failed to read CSV: {exc}")
        return False, errors

    # Check columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra_cols:
        errors.append(f"Unexpected columns detected: {extra_cols}")
        return False, errors

    # Check row count
    if len(df) != 50:
        errors.append(f"Expected exactly 50 cases, but found {len(df)}")

    # Check case_id duplicates
    if df["case_id"].duplicated().any():
        dups = df[df["case_id"].duplicated()]["case_id"].tolist()
        errors.append(f"Duplicate case_id found: {dups}")

    if df["case_id"].isna().any():
        errors.append("Found missing case_id values")

    # Check each row
    for idx, row in df.iterrows():
        case_id = row.get("case_id", f"row_{idx+1}")

        # 1. human_intent
        h_intent = str(row.get("human_intent", "")).strip()
        if not h_intent or pd.isna(row.get("human_intent")):
            errors.append(f"{case_id}: Missing human_intent")
        elif h_intent not in ALLOWED_INTENTS:
            errors.append(
                f"{case_id}: Invalid human_intent '{h_intent}'. Must be one of: {sorted(ALLOWED_INTENTS)}"
            )

        # 2. intent_correct
        intent_corr = str(row.get("intent_correct", "")).strip().upper()
        if intent_corr not in {"YES", "NO"}:
            errors.append(
                f"{case_id}: Invalid intent_correct '{row.get('intent_correct')}'. Must be 'YES' or 'NO'"
            )

        # 3. Ratings (1-5 integer)
        for dim in RATING_DIMENSIONS_NEW:
            val = row.get(dim)
            if pd.isna(val) or str(val).strip() == "":
                errors.append(f"{case_id}: Missing rating for {dim}")
            else:
                try:
                    f_val = float(val)
                    if not f_val.is_integer() or not (1 <= int(f_val) <= 5):
                        errors.append(
                            f"{case_id}: Invalid rating {val} for {dim}. Must be integer 1 to 5"
                        )
                except (ValueError, TypeError):
                    errors.append(
                        f"{case_id}: Non-numeric rating '{val}' for {dim}. Must be integer 1 to 5"
                    )

        # 4. evidence_supported
        ev_supp = str(row.get("evidence_supported", "")).strip().upper()
        if ev_supp not in {"YES", "NO"}:
            errors.append(
                f"{case_id}: Invalid evidence_supported '{row.get('evidence_supported')}'. Must be 'YES' or 'NO'"
            )

        # 5. overall_comment
        comment = str(row.get("overall_comment", "")).strip()
        if not comment or pd.isna(row.get("overall_comment")):
            errors.append(f"{case_id}: Missing overall_comment")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_annotation_file(
    file_path: str = "golden_set/human_annotation_form.csv",
) -> Tuple[str, Dict[str, Any]]:
    """
    Validation helper preserving exact interface and return schema for existing test suites.
    """
    path = PROJECT_ROOT / file_path if not Path(file_path).is_absolute() else Path(file_path)
    if not path.exists():
        return "NOT_YET_VERIFIED", {"error": "file_not_found"}

    df = pd.read_csv(path)
    total_rows = len(df)
    issues: List[str] = []
    incomplete_rows: List[Tuple[str, List[str]]] = []
    invalid_rows: List[Tuple[str, str, Any]] = []

    # If new template schema
    if "case_id" in df.columns:
        valid, errs = validate_human_annotations(str(path))
        if valid:
            return "VERIFIED", {
                "complete_count": len(df),
                "total": total_rows,
                "issues": [],
                "incomplete_rows": [],
                "invalid_rows": [],
            }
        else:
            return "NOT_YET_VERIFIED", {
                "complete_count": 0,
                "total": total_rows,
                "issues": errs,
                "incomplete_rows": [],
                "invalid_rows": errs,
            }

    # Legacy schema
    if "evaluation_id" not in df.columns:
        issues.append("Missing required 'evaluation_id' column.")
        return "NOT_YET_VERIFIED", {"issues": issues}

    if total_rows != 50:
        issues.append(f"Expected exactly 50 rows, but found {total_rows}.")

    dup_ids = df[df["evaluation_id"].duplicated()]["evaluation_id"].tolist()
    if dup_ids:
        issues.append(f"Duplicate evaluation_ids found: {dup_ids}")

    expected_ids = [f"eval_{i:02d}" for i in range(1, 51)]
    actual_ids = df["evaluation_id"].tolist()
    missing_ids = [eid for eid in expected_ids if eid not in actual_ids]
    if missing_ids:
        issues.append(f"Missing expected evaluation IDs: {missing_ids}")

    for dim in LEGACY_RATING_DIMENSIONS:
        if dim not in df.columns:
            issues.append(f"Missing required dimension column: {dim}")

    if "annotation_source" not in df.columns:
        issues.append("Missing 'annotation_source' column.")
        return "NOT_YET_VERIFIED", {"issues": issues}
    else:
        non_human = df[df["annotation_source"] != "human"]
        if len(non_human) > 0:
            source_vals = non_human["annotation_source"].unique().tolist()
            issues.append(f"{len(non_human)}/50 rows have annotation_source != 'human'")
            return "NOT_YET_VERIFIED", {"issues": issues}

    complete_count = 0
    for idx, row in df.iterrows():
        eid = str(row.get("evaluation_id", f"row_{idx}"))
        missing_dims = []
        row_invalid = False

        for dim in LEGACY_RATING_DIMENSIONS:
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

    if not issues and complete_count == 50 and not incomplete_rows and not invalid_rows:
        return "VERIFIED", {
            "complete_count": complete_count,
            "total": total_rows,
            "issues": issues,
            "incomplete_rows": incomplete_rows,
            "invalid_rows": invalid_rows,
        }
    else:
        return "NOT_YET_VERIFIED", {
            "complete_count": complete_count,
            "total": total_rows,
            "issues": issues,
            "incomplete_rows": incomplete_rows,
            "invalid_rows": invalid_rows,
        }


def main():
    parser = argparse.ArgumentParser(description="Validate human annotation file.")
    parser.add_argument(
        "--file",
        default="golden_set/human_evaluation_template.csv",
        help="Path to annotation CSV file to validate",
    )
    args = parser.parse_args()

    is_valid, errors = validate_human_annotations(args.file)

    if is_valid:
        print("HUMAN ANNOTATION VALIDATION PASSED")
        print("Cases: 50")
        sys.exit(0)
    else:
        print("HUMAN ANNOTATION VALIDATION FAILED")
        print(f"Total Errors Found: {len(errors)}")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
