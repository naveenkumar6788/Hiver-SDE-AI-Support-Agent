# Model-Simulated Human-Style Evaluation Status

## Notice of Scientific Integrity

> [!IMPORTANT]
> **These ratings are model-simulated human-style evaluations and are NOT genuine human annotations. They must not be used to claim human–LLM agreement.**

---

## 1. Purpose & Scope

This evaluation simulates the critical scrutiny of an independent human customer-support quality reviewer on the 50-case evaluation subset (`eval_01` through `eval_50`). It was produced to understand qualitative failure modes and strict quality criteria without violating scientific integrity.

The resulting dataset is saved strictly in:
- [`evaluation/results/model_simulated_human_ratings_50.csv`](file:///d:/Hiver-SDE-AI-Support-Agent/evaluation/results/model_simulated_human_ratings_50.csv)
- [`evaluation/results/model_simulated_human_summary.csv`](file:///d:/Hiver-SDE-AI-Support-Agent/evaluation/results/model_simulated_human_summary.csv)

The official human annotation files remain completely unrated:
- [`golden_set/human_annotation_form.csv`](file:///d:/Hiver-SDE-AI-Support-Agent/golden_set/human_annotation_form.csv) (all 50 cases remain blank)
- [`golden_set/human_judge_50.csv`](file:///d:/Hiver-SDE-AI-Support-Agent/golden_set/human_judge_50.csv) (all 50 cases remain blank)

---

## 2. Review Methodology & Behavioral Criteria

The simulation strictly enforced the following review standards:
1. **No Polite Bias**: A polite or empathetic tone was not rewarded with high correctness if the guidance failed to solve the customer's actual technical issue.
2. **Specific Problem Relevance**: Heavy penalties were applied when responses matched the broad topic (e.g., audio, cellular, battery) but completely missed the specific problem (e.g., suggesting software force-restart for a physically broken headphone jack, or troubleshooting network settings for App Store password configurations).
3. **Evidence Mismatch Awareness**: If the retrieved historical evidence was mismatched to the user's issue, groundedness was not awarded merely because the generated reply echoed the erroneous evidence.
4. **Link Truncation Penalties**: Responses ending abruptly with truncated links (e.g. "For more info, check out:") were scored 1 or 2 on completeness regardless of topic relevance.
5. **Legitimate Clarification Credit**: Diagnostic questions received credit only when clarification was genuinely warranted by an underspecified customer inquiry.

---

## 3. Results Summary (50 Cases)

| Criterion | Mean Score (1–5) | Distribution (1 / 2 / 3 / 4 / 5) | Reviewer Interpretation |
| :--- | :---: | :---: | :--- |
| **Correctness** | **2.52 / 5.0** | 15 / 12 / 9 / 10 / 4 | Significant rate of mismatched technical solutions and truncated links. |
| **Groundedness** | **3.88 / 5.0** | 3 / 7 / 7 / 9 / 24 | Most responses rely on retrieved snippets or safe diagnostic fallbacks. |
| **Relevance** | **2.74 / 5.0** | 13 / 10 / 9 / 13 / 5 | 23 of 50 cases failed to directly address the customer's specific problem. |
| **Completeness** | **2.16 / 5.0** | 18 / 12 / 16 / 2 / 2 | Clarifications and missing URLs leave many replies without actionable fixes. |
| **Tone** | **3.86 / 5.0** | 0 / 0 / 10 / 37 / 3 | Consistently polite, professional, and courteous customer-service tone. |
| **Unsupported Claims** | **4.46 / 5.0** | 0 / 0 / 7 / 13 / 30 | High absence of hallucinated facts; models stay within evidence or safe templates. |
| **Overall Mean** | **3.27 / 5.0** | — | Critical human-perspective quality baseline. |

- **Cases with Major Quality Problems**: **27 / 50 (54.0%)**  
  *(Defined as Correctness $\le 2$, Relevance $\le 2$, or Completeness $\le 1$)*.

---

## 4. Operational Invariants Preserved

1. **`python src/evaluation/validate_human_annotations.py`** continues to return **`NOT_YET_VERIFIED`**.
2. **`python src/evaluation/human_agreement.py`** continues to return **`HUMAN AGREEMENT: NOT YET VERIFIED`**.
3. Zero agreement metrics (Spearman, Pearson, Cohen's kappa, exact agreement percentage) are claimed or calculated between human annotations and the LLM judge.
