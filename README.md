# Hiver SDE AI Support Agent

Automated customer support agent for enterprise social channels, specialized for **@AppleSupport** Twitter inquiries. The system classifies customer intent, retrieves relevant historical support evidence, applies strict evidence safety and divergence gating, generates grounded responses, and enforces deterministic escalation policies for account security, billing, and hardware safety hazards.

---

## Submission Philosophy

This project intentionally prioritizes:
- **Strict Groundedness**: Every generated claim must be directly anchored in verified evidence or safe diagnostic clarification.
- **Defensible Safety**: Action divergence gating and deterministic sensitive pattern detection prevent misleading guidance and device safety hazards.
- **Reproducibility**: Clean deterministic evaluation harnesses with zero data leakage across conversation threads.
- **Rigorous Evaluation & Metric Literacy**: Transparent multidimensional evaluation (Accuracy, Macro F1, Weighted F1, Per-Class reports) rather than misleading single-metric headlines.
- **Transparent Limitations**: Clear disclosure of incomplete LLM judge provider blocks and pending human verification, prioritizing engineering integrity over ungrounded claims.

---

## Problem & Domain Scope

Public social customer support on `@AppleSupport` presents unique engineering challenges:
1. **Terse, Unstructured Inquiries**: Queries average 10–25 words, often filled with colloquial slang, missing device specifications, or emotional distress.
2. **High Cost of Technical Hallucinations**: Prescribing irrelevant or physically impossible instructions (e.g. software reboots for broken SIM trays, or network resets for billing locks) degrades user trust and can trigger battery/hardware risks.
3. **Sensitive Safety & Security Routing**: Issues involving compromised Apple IDs, fraudulent credit card transactions, or physical battery hazards must reliably escalate to human specialists rather than receiving automated advice.

The agent operates across 12 domain categories: `app_problems`, `app_store_downloads`, `apple_id_icloud`, `apple_music_itunes`, `audio_speaker`, `battery_charging`, `calls_cellular`, `device_hardware`, `ios_update`, `other_unclear`, `screen_display`, and `wifi_connectivity`.

---

## Architecture

The end-to-end processing pipeline operates synchronously through 5 modular stages:

```
Customer Message
       │
       ▼
[1. Intent Classification]       -->  src/intent/ (classifier.py, baselines.py)
       │
       ▼
[2. Historical Retrieval]        -->  src/retrieval/ (retriever.py, evidence_relevance.py)
       │
       ▼
[3. Evidence Safety Gate]        -->  src/agent/reply_generator.py (Action Divergence & Specific Problem Match)
       │
       ▼
[4. Grounded Reply Generation]   -->  src/agent/reply_generator.py (Grounded advice or safe clarification)
       │
       ▼
[5. Escalation Decision]         -->  src/agent/escalation.py (AUTO_HANDLE vs. ESCALATE)
```

### Module Organization
* `src/data/`: Conversation reconstruction and thread-level dataset curation.
* `src/intent/`: 12-class intent classification models, feature extractors, and heuristic baselines.
* `src/retrieval/`: Candidate retrieval index, query construction, and relevance scoring.
* `src/agent/`: Evidence safety gates, specific problem match layer, grounded reply generation, and escalation engine.
* `src/evaluation/`: End-to-end deterministic evaluation, LLM-as-a-judge pipeline, and human agreement validation.

---

## Verified Deterministic Results

Deterministic evaluation executed across all **160 evaluation examples, including 44 human-reviewed examples; the remaining examples were labeled through a structured audit process** (`src/evaluation/evaluate_reply_generator.py`):

| Metric / Dimension | Verified Result | Notes & Invariants |
| :--- | :---: | :--- |
| **Evaluation Set Size** | **160 cases** | 44 human-reviewed + structured audit labeling |
| **Intent Classification Accuracy** | **91.25% (146 / 160)** | +2.50% over baseline (0 regressions) |
| **Intent Macro F1** | **0.9075** | Equal weight across all 12 classes |
| **Intent Weighted F1** | **0.9133** | Support-weighted multi-class F1 |
| **Evidence Used & Safe Rate** | **84.38% (135 / 160)** | Gating safely rejected 4 unsafe action-divergent cases |
| **Deterministic Grounding Pass Rate** | **100.0% (160 / 160)** | All 160 evaluated responses passed deterministic grounding checks |
| **Unsupported Claims Detected** | **0 / 160 (0.0%)** | Zero unsupported claims flagged by verifier |
| **Problem Conflicts Detected** | **0 / 160 (0.0%)** | Zero problem-domain conflicts among accepted evidence |
| **Generation Errors / Exceptions** | **0 / 160 (0.0%)** | Zero runtime exceptions across all evaluation passes |
| **Specific Problem Match Rate** | **4.38% (7 / 160)** | Strict semantic problem compatibility active |
| **Routing Decisions** | **157 AUTO_HANDLE / 3 ESCALATE** | Sensitive security/hardware threats route to specialists |
| **Automated Regression Suite** | **39 / 39 PASS** | `pytest tests/ -q` (100% synchronous pass) |
| **Known Failure Regression Suite** | **24 / 24 PASS** | `pytest tests/test_known_failure_cases.py -q` |

### Per-Class Intent Breakdown (160 Cases)
```text
                     precision    recall  f1-score   support

        app_problems      1.00      0.75      0.86        12
 app_store_downloads      1.00      1.00      1.00         2
     apple_id_icloud      1.00      1.00      1.00        16
  apple_music_itunes      0.73      1.00      0.84         8
       audio_speaker      0.91      0.91      0.91        11
    battery_charging      0.73      1.00      0.85        11
      calls_cellular      1.00      1.00      1.00         9
     device_hardware      0.90      0.82      0.86        11
          ios_update      0.94      0.86      0.90        36
       other_unclear      1.00      0.96      0.98        28
      screen_display      0.88      0.78      0.82         9
   wifi_connectivity      0.78      1.00      0.88         7

            accuracy                          0.91       160
           macro avg      0.91      0.92      0.91       160
        weighted avg      0.92      0.91      0.91       160
```

---

## LLM Judge Evaluation

### Current LLM Judge Run: INCOMPLETE (Partial)
* **Status**: **`INCOMPLETE` (PARTIAL EVALUATION)**
* **Target / Requested**: 50 cases
* **Successful Judgments**: 12 cases
* **Failed / Unavailable**: 38 cases (1 rate-limited attempt + 37 unattempted)
* **Root Cause**: The Groq API endpoint returned a **Cloudflare HTTP 403 Forbidden (error code 1010: access blocked based on client signature)** across all attempted fallback models (`openai/gpt-oss-20b`, `llama-3.1-8b-instant`, `llama3-8b-8192`, `mixtral-8x7b-32768`).
* **Methodological Notice**: The 12-case partial results (Overall: 2.4583, Correctness: 1.5833, Groundedness: 2.0833) are **NOT** presented as a final benchmark score.

### Historical LLM Judge Baseline (Separately Preserved)
Preserved in [`evaluation/results/historical_llm_judge_summary.csv`](evaluation/results/historical_llm_judge_summary.csv) as an independent historical benchmark (50 / 50 cases evaluated):
* **Historical Overall Score**: **2.5272 / 5.0**
* **Historical Correctness**: **1.3800** | **Historical Groundedness**: **2.2800**
* **Historical Relevance**: **1.8600** | **Historical Completeness**: **1.2800**
* **Historical Tone**: **4.0800** | **Historical Unsupported Claims**: **4.2800**
* **Historical Escalation Appropriateness**: **2.4000**
* **Score Distribution**: 1-star: 4 | 2-star: 28 | 3-star: 11 | 4-star: 6 | 5-star: 1

---

## Human Evaluation Status

### Status: HUMAN AGREEMENT NOT YET VERIFIED
* The evaluation set contains **44 human-reviewed examples**, while the remaining examples were labeled through a structured audit process.
* The 50 annotation rows in `golden_set/human_annotation_form.csv` have `annotation_source = model_assisted` (produced by an automated model reviewer). In strict adherence to scientific integrity, **model-assisted ratings are NOT claimed as human ratings**, and inter-rater agreement statistics (Spearman's $\rho$, Cohen's $\kappa$) are withheld until authentic, independent human evaluations are recorded.
* The double-blind evaluation dataset (`golden_set/human_judge_50_blind.csv`) and verification harness (`src/evaluation/validate_human_annotations.py`) are fully prepared.

---

## Dataset & Thread Splitting

* **Full Dataset**: The raw Kaggle Customer Service on Twitter dataset (~500 MB) is intentionally **excluded from Git tracking** via `.gitignore` to respect GitHub storage constraints and prevent repository bloat.
* **Dialogue-Thread-Level Split**: Conversations are partitioned at the root thread level, ensuring all turns of an interaction remain exclusively in either the retrieval corpus or the evaluation set, eliminating conversation data leakage.

---

## Environment Setup

### Windows PowerShell Setup
```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (UTF-8 encoded)
python -m pip install -r requirements.txt

# 3. Configure environment variables (optional for offline deterministic evaluation)
Copy-Item .env.example .env
# Edit .env to supply LLM_API_KEY if running LLM judge
```

---

## Running Tests

Execute the automated test suite synchronously:

```powershell
# Run the complete regression test suite (39 tests)
pytest tests/ -q
# Expected: 39 passed

# Run known failure regression cases specifically (24 tests)
pytest tests/test_known_failure_cases.py -q
# Expected: 24 passed
```

---

## Reproducing Evaluation

Run the deterministic evaluation pipeline using the actual repository scripts:

```powershell
# 1. Run the primary 160-case deterministic reply generator evaluation
python src/evaluation/evaluate_reply_generator.py
# Outputs metrics to evaluation/results/reply_generator_summary.csv

# 2. Run baseline intent classification evaluation
python src/intent/baselines.py

# 3. Validate human annotation integrity status
python src/evaluation/validate_human_annotations.py
# Expected output: NOT_YET_VERIFIED (enforces model_assisted guard)

# 4. Check human agreement status
python src/evaluation/human_agreement.py
# Expected output: HUMAN AGREEMENT: NOT YET VERIFIED
```

---

## System Limitations

1. **Incomplete Current LLM Judge**: Cloudflare HTTP 403 blocks at the Groq provider edge prevented automated completion of the full 50-case evaluation run.
2. **Pending Human Ground Truth**: Response quality annotations are model-assisted; verified human–LLM agreement is pending authentic human study completion.
3. **Historical Evidence Generality**: Historical Twitter customer support frequently relied on generic canned phrases ("Please DM us"), requiring the safety gate to fall back to diagnostic clarification rather than providing complete instant troubleshooting.
4. **Imperfect Retrieval Specificity**: Lexical retrieval occasionally prioritizes keyword frequency over semantic problem nuances without deep semantic re-ranking.
5. **Deterministic Grounding Scope**: Deterministic grounding verifies that all stated guidance is strictly supported by evidence or structured templates, but does not prove semantic perfection.

---

## Key Documentation Links

* [Comprehensive System Evaluation Report](report/report.md) — Complete 16-section technical evaluation report.
* [System Walkthrough & Invariants](walkthrough.md) — Detailed implementation walkthrough and verification record.
* [Decision Log](decision_log.md) — 13 key architectural decisions, rationale, and engineering trade-offs.
* [Intent Classification Guidelines](golden_set/intent_guidelines.md) — Official 12-class taxonomy definitions.
* [Human Evaluation Guidelines](golden_set/human_judge_guidelines.md) — Standardized 1–5 scoring rubrics for human reviewers.
