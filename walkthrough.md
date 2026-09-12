# Hiver SDE AI Support Agent — System Walkthrough & Verification

## 1. System Evaluation Framework

The project rigorously separates evaluation into four distinct stages:
- **Stage A: 160-Case Intent, Retrieval, and Safety Baseline**: Empirical evaluation across 160 evaluation examples, including 44 human-reviewed examples; the remaining examples were labeled through a structured audit process.
- **Stage B: Deterministic Grounding & Safety Verification**: Verifies zero unsupported claims and zero problem-domain conflicts across all 160 evaluation cases.
- **Stage C: LLM Judge Evaluation**:
  - **Historical Baseline**: 50/50 complete run (Overall: 2.5272 / 5.0, Correctness: 1.3800, Groundedness: 2.2800, Relevance: 1.8600, Completeness: 1.2800, Tone: 4.0800, Unsupported Claims: 4.2800, Escalation Appropriateness: 2.4000).
  - **Current System Run**: **`INCOMPLETE` (PARTIAL)** — 12 / 50 successful, 38 failed/unavailable due to Groq Cloudflare HTTP 403 error 1010. Partial 12-case metrics are non-representative and must not be presented as a final score.
- **Stage D: Genuine Human Evaluation**: Blinded annotation protocol and validation tooling established; human agreement status strictly enforced as `HUMAN AGREEMENT NOT YET VERIFIED` because current annotations are model-assisted.

---

## 2. Final Verified Deterministic Metrics

Deterministic evaluation executed across all **160 evaluation examples, including 44 human-reviewed examples; the remaining examples were labeled through a structured audit process** (`src/evaluation/evaluate_reply_generator.py`):

| Metric / Dimension | Baseline Metric | Post-Optimization Metric | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Intent Classification Accuracy** | 88.75% (142 / 160) | **91.25% (146 / 160)** | **+2.50% (+4 cases corrected, 0 regressions)** |
| **Intent Macro F1** | 0.8663 | **0.9075** | **+0.0412** |
| **Intent Weighted F1** | 0.8888 | **0.9133** | **+0.0245** |
| **Deterministic Grounded Rate** | 100.0% (160 / 160) | **100.0% (160 / 160)** | **Preserved Invariant** |
| **Unsupported Claims** | 0 / 160 (0.0%) | **0 / 160 (0.0%)** | **Preserved Invariant** |
| **Problem Conflicts Detected** | 0 / 160 (0.0%) | **0 / 160 (0.0%)** | **Preserved Invariant** |
| **Generation Errors / Exceptions** | 0 / 160 (0.0%) | **0 / 160 (0.0%)** | **Zero runtime failures** |
| **Evidence Used & Safe Rate** | 86.88% (139 / 160) | **84.38% (135 / 160)** | **Safely gated 4 unsafe action-divergent cases** |
| **Specific Problem Match Rate** | 0.00% (0 / 160) | **4.38% (7 / 160)** | **Semantic problem alignment active** |
| **Average Evidence Relevance** | — | **0.5681** | Empirically measured evidence relevance |
| **Average Evidence Quality** | — | **0.7508** | Quality score across candidate evidence |
| **Average Problem Match** | — | **0.4889** | Semantic problem compatibility score |
| **Average Customer Problem Sim** | — | **0.5413** | Customer query vs. historical problem |
| **Average Response Problem Sim** | — | **0.4412** | Response vs. problem context similarity |
| **Average Final Evidence Score** | — | **0.4570** | Specificity-weighted combined score |
| **AUTO_HANDLE Decisions** | 159 / 160 | **157 / 160** | Automated handling with safe fallbacks |
| **ESCALATE Decisions** | 1 / 160 | **3 / 160** | Proactive routing for security/hardware risks |
| **Automated Test Suite** | — | **39 / 39 PASS** | **`pytest tests/ -q` (includes 24/24 failure tests)** |

### Per-Class Intent Classification Report (160 Examples)

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

## 3. Mandatory Metric Literacy: Why "91.25% Intent Accuracy" Is Incomplete

Reporting **"91.25% Intent Accuracy"** in isolation is incomplete because:
1. **Support Imbalance**: Dominant classes (`ios_update` with $N=36$, `other_unclear` with $N=28$) represent 40% of the evaluation set. Raw accuracy can mask poor performance on smaller, high-consequence classes.
2. **Macro F1 (0.9075) vs. Weighted F1 (0.9133)**: The Macro F1 treats all 12 classes equally. Lower recall on `app_problems` (0.75) and lower precision on `battery_charging` (0.73) or `apple_music_itunes` (0.73) are visible in Macro F1 but obscured by accuracy alone.
3. **Operational Consequence**: Customer harm from a misclassified account security or hardware issue is much higher than misclassifying a general inquiry. Intent accuracy must always be evaluated alongside Macro F1, Weighted F1, and per-class metrics.

---

## 4. LLM-as-a-Judge Evaluation Status

### Current Run: INCOMPLETE (12 / 50 Judged)
- **Evaluation Status**: **`INCOMPLETE`**
- **Requested**: 50 | **Successful**: 12 | **Failed / Unavailable**: 38
- **Provider Failure**: Groq API blocked subsequent calls via Cloudflare HTTP 403 (error code 1010) across all fallback models (`openai/gpt-oss-20b`, `llama-3.1-8b-instant`, `llama3-8b-8192`, `mixtral-8x7b-32768`).
- **Partial 12-Case Metrics (NON-REPRESENTATIVE — DO NOT USE AS FINAL SCORE)**:
  - Overall Score: **2.4583 / 5.0**
  - Correctness: **1.5833** | Groundedness: **2.0833** | Relevance: **1.9167**
  - Completeness: **1.5000** | Tone: **4.2500** | Unsupported Claims: **3.4167**
  - Escalation Appropriateness: **2.3333** | Unsupported Claims Found: 5 / 12

### Historical Baseline LLM Judge (Separate 50-Case Benchmark)
Preserved in [`evaluation/results/historical_llm_judge_summary.csv`](file:///D:/Hiver-SDE-AI-Support-Agent/evaluation/results/historical_llm_judge_summary.csv):
- **Historical Overall Score**: **2.5272 / 5.0**
- **Historical Correctness**: **1.3800** | **Groundedness**: **2.2800** | **Relevance**: **1.8600**
- **Historical Completeness**: **1.2800** | **Tone**: **4.0800** | **Unsupported Claims**: **4.2800**
- **Historical Escalation Appropriateness**: **2.4000**
- **Historical Score Distribution**: 1-star: 4 | 2-star: 28 | 3-star: 11 | 4-star: 6 | 5-star: 1

---

## 5. Failure Analysis (50 Historical Judge Cases)

Audited from [`evaluation/results/llm_judge_failure_analysis.csv`](file:///D:/Hiver-SDE-AI-Support-Agent/evaluation/results/llm_judge_failure_analysis.csv). Every case (`eval_01` to `eval_50`) is accounted for exactly once in a mutually exclusive 7-category taxonomy:

| Category | Description | Count | Percentage |
| :--- | :--- | :---: | :---: |
| **A** | Generic / template evidence selected (boilerplate historical reply) | **15** | 30.0% |
| **B** | Wrong intent classification upstream | **12** | 24.0% |
| **J** | Judge strictness / semantic limitation (judge penalized valid clarification) | **9** | 18.0% |
| **D** | Unsupported claim / domain divergence | **7** | 14.0% |
| **H** | Resolved customer message handling (user fixed issue; redundant advice) | **4** | 8.0% |
| **E** | Unnecessary / redundant clarification | **2** | 4.0% |
| **G** | Safety / escalation issue (hazard required human specialist) | **1** | 2.0% |
| **TOTAL** | **Mutually exclusive primary categories** | **50** | **100.0%** |

- **Verification of `eval_34` (tweet `1123670`)**: Customer reported high-pitched smoke-alarm sound from phone speaker. Classified as **`G` (Safety / escalation issue)** because hardware acoustic alarms indicate battery/hardware danger requiring human escalation.
- **Verification of Tweet `1338119` (`eval_43`)**: Customer query regarding SIM-free pre-order classified as **`D` (Unsupported claim / domain divergence)**.

---

## 6. Human Evaluation & Scientific Integrity Status

### Official Status: HUMAN AGREEMENT NOT YET VERIFIED
- The `annotation_source` column in [`golden_set/human_annotation_form.csv`](file:///D:/Hiver-SDE-AI-Support-Agent/golden_set/human_annotation_form.csv) is `model_assisted` for all 50 rows. These ratings were produced by an AI reviewer session and are **not** independent human ground truth.
- In accordance with scientific integrity rules, agreement metrics (Spearman's $\rho$, Cohen's $\kappa$) are explicitly withheld until genuine independent human annotations are recorded.
- Automated validation checks enforce this guard:
  ```bash
  python src/evaluation/validate_human_annotations.py  # Output: NOT_YET_VERIFIED
  python src/evaluation/human_agreement.py             # Output: HUMAN AGREEMENT: NOT YET VERIFIED
  ```

---

## 7. Technical Defensibility & Verified Invariants

- **Careful Grounding Invariant**: All 160 evaluated responses passed the deterministic grounding checks, with no unsupported-claim violations detected by the implemented verifier.
- **Safety Invariant**: No problem-domain conflicts were detected among accepted evidence in the 160-example evaluation.
- **Intent Classifier Performance**: 91.25% Intent Accuracy, 0.9075 Macro F1, and 0.9133 Weighted F1 across 12 support categories with zero regressions.
- **Evidence Safety Gate**: Action divergence and generic boilerplate filters actively reject misaligned guidance (pruning evidence usage to 84.38%), with 0 problem conflicts detected.
- **Safety Escalation**: Sensitive security threats, account takeovers, unauthorized billing, and hardware hazards route to official Apple Support.
- **Test Integrity**: All 39 regression tests pass (`pytest tests/ -q`), including all 24 known failure test cases.
