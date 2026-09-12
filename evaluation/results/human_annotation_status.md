# Human Annotation & Agreement Status

## Status: HUMAN AGREEMENT: NOT YET VERIFIED

### Official Statement
**Human–LLM agreement is currently NOT YET VERIFIED because genuine, independent human response-quality annotations have not been completed for the 50-case evaluation sample.**

In strict adherence to scientific integrity principles:
1. **Model-assisted ratings must NOT be used as human annotations.** The ratings in `golden_set/human_annotation_form.csv` have `annotation_source = model_assisted` — they were produced by an AI reviewer and are NOT genuine human evaluations.
2. **Structural CSV validity does NOT prove human authorship.** The validator checks `annotation_source` and only returns VERIFIED when `annotation_source = human` for all rows.
3. **No human ratings have been fabricated or simulated.** Agreement statistics (Spearman correlation, Cohen's kappa, exact agreement percentage) are withheld until authentic human ratings are provided.

---

## Evaluation Set Readiness

- **Evaluation Dataset**: 50 representative customer support cases sampled reproducibly (`random_state=42`) across 11 intent categories.
- **Blind Review File**: [`golden_set/human_judge_50_blind.csv`](file:///d:/Hiver-SDE-AI-Support-Agent/golden_set/human_judge_50_blind.csv) is prepared with all LLM judge scores omitted to prevent evaluator confirmation bias.
- **Annotation Guidelines**: Standardized integer 1–5 scoring rubrics across Correctness, Groundedness, Relevance, Completeness, Tone, and Unsupported Claims are documented in [`golden_set/human_judge_guidelines.md`](file:///d:/Hiver-SDE-AI-Support-Agent/golden_set/human_judge_guidelines.md).
- **Interactive Annotation Tooling**: Available via `python src/evaluation/human_annotation_template.py --interactive`.

---

## How to Complete Human Evaluation

1. Open `golden_set/human_judge_50_blind.csv` (LLM scores hidden).
2. For each of 50 rows, rate: correctness, groundedness, relevance, completeness, tone, unsupported_claims (1–5 integers).
3. Set `annotation_source = human` for each completed row.
4. Save as `golden_set/human_annotation_form.csv`.
5. Run: `python src/evaluation/validate_human_annotations.py` → should return VERIFIED.
6. Run: `python src/evaluation/human_agreement.py` → will compute agreement metrics.

---

## Verification Lifecycle

The verification pipeline ([`src/evaluation/human_agreement.py`](file:///d:/Hiver-SDE-AI-Support-Agent/src/evaluation/human_agreement.py)) enforces three distinct operational states:
1. **NOT YET VERIFIED**: Active state. `annotation_source != human` or no ratings. Calculations withheld.
2. **INCOMPLETE (1–49/50 cases rated)**: Progress tracked. Formal agreement withheld until all 50 cases complete.
3. **VERIFIED (50/50 rated, annotation_source=human)**: Computes exact agreement, within ±1, MAD, Spearman ρ, Pearson r, and Cohen's κ.
