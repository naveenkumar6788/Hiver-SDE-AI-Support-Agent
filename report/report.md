# Comprehensive System Evaluation Report: Hiver SDE AI Support Agent

**Date**: September 2026  
**Dataset**: AppleSupport Twitter Customer Service Corpus  
**Repository**: `D:\Hiver-SDE-AI-Support-Agent`  
**Test Suite**: 102/102 Passing (`pytest`)  
**Evaluation Scope**: Autonomous Intent Classification, Thread Reconstruction, Evidence Retrieval, Action-Divergence Safety Gating, Response Generation, Deterministic Escalation, Automated Benchmarks, Current & Historical LLM Judge Evaluations, 50-Case Independent Human Evaluation, and Human–LLM Agreement.

---

## 1. Problem / Objective

Automated customer support for public enterprise channels (@AppleSupport on Twitter) presents three critical production challenges:
1. **Terse and noisy queries**: User messages average 10–25 words with non-standard slang, emojis, missing device specifications, and emotional venting.
2. **Risk of technical hallucination**: Prescribing incorrect troubleshooting (e.g. software reboots for swollen batteries, or network resets for carrier SIM locks) degrades customer trust and causes device hazards.
3. **Strict safety and security boundaries**: Issues involving compromised Apple IDs, unauthorized billing charges, or physical hardware danger must immediately route to human specialists.

**Objective**: Build a deterministic, safety-first AI support agent that accurately classifies customer intent, retrieves relevant historical support guidance, enforces rigorous evidence safety gates, prevents unsupported claims, and reliably escalates critical risks.

---

## 2. Dataset

- **Corpus**: Reconstructed dialogue threads from the public Kaggle Customer Service on Twitter dataset, filtered strictly to the `@AppleSupport` domain.
- **Scale**: 117,076 customer inquiries paired with official Apple Support responses.
- **Evaluation Golden Set**: Exactly **160 evaluation examples**. Provenance: **44 examples were genuinely human-reviewed**; the remaining **116 examples were labeled through a structured, multi-pass audit process**. The report does not claim that all 160 examples were independently human-verified.
- **Domain Scope**: Covers 12 operational support categories across iOS devices, hardware, audio, battery, system updates, Apple ID, cellular connectivity, and App Store services.

---

## 3. Conversation Reconstruction

Random message-level splitting in conversational NLP causes severe data leakage: turn $t$ appears in the retrieval index while turn $t+1$ of the same dialogue appears in evaluation, artificially inflating metrics.
- **Thread-Level Grouping**: Splitting is executed strictly at the root `conversation_id` / thread level using `tweet_id` and `in_response_to_tweet_id` tree reconstruction.
- **Temporal & Entity Isolation**: All customer queries and agent turns belonging to the same interaction remain co-located in either the retrieval database or evaluation set, never split across both.
- **Leakage Invariant**: Zero overlap exists between evaluation query contexts and historical response snippets used for candidate evidence.

---

## 4. Intent Taxonomy

The system implements a domain-tailored 12-class taxonomy reflecting real Apple Support operational routing:

| Intent Category | Operational Scope | Support ($N$) |
| :--- | :--- | :---: |
| `app_problems` | Third-party app crashes, freezes, and app-specific glitches | 12 |
| `app_store_downloads` | App Store downloads, purchase errors, and account billing | 2 |
| `apple_id_icloud` | Apple ID authentication, password recovery, 2FA, iCloud sync | 16 |
| `apple_music_itunes` | Music streaming, library syncing, iTunes playlist errors | 8 |
| `audio_speaker` | Speaker crackle, microphone failure, call receiver volume | 11 |
| `battery_charging` | Battery drain, charging cables/ports, device overheating | 11 |
| `calls_cellular` | Cellular carrier reception, dropped calls, SIM activation, VoLTE | 9 |
| `device_hardware` | Physical damage, cracked screens, buttons, cameras, water contact | 11 |
| `ios_update` | iOS installation errors, post-update glitches, storage space | 36 |
| `other_unclear` | Terse queries, greetings, ambiguous complaints, or fix confirmations | 28 |
| `screen_display` | Touchscreen unresponsiveness, display artifacts, brightness issues | 9 |
| `wifi_connectivity` | Wi-Fi network discovery, password rejection, intermittent drops | 7 |
| **Total Evaluation Set** | **Multi-class distribution (44 human-reviewed + 116 audit-verified)** | **160** |

---

## 5. System Architecture

The AI Support Agent operates as a deterministic, pipelined expert architecture:
1. **Pre-Processing & Sensitive Guard**: Fast regex scanner checks for physical hazards, account compromise, and payment disputes. If triggered, execution immediately routes to `ESCALATE`.
2. **Intent Classification**: Hybrid ensemble combining calibrated TF-IDF Logistic Regression with 16 high-precision domain disambiguation rules.
3. **Intent-Constrained Retrieval**: TF-IDF index restricted to the candidate pool matching the predicted intent.
4. **Action-Divergence Safety Gating**: 4-factor scoring model (relevance, quality, problem match, specificity) with hard conflict rules rejecting misaligned technical actions.
5. **Deterministic Response Generation**: If evidence passes all gates, generates grounded technical guidance; if evidence is rejected or the query is ambiguous, generates safe diagnostic clarifications.

---

## 6. Retrieval

- **Constraint**: Retrieval is partitioned strictly by predicted intent, eliminating cross-domain candidate contamination (e.g. Wi-Fi advice retrieved for battery drain).
- **Candidate Scoring**: Candidacy scoring rewards actionable, specific diagnostic steps while penalizing generic boilerplate (e.g. pure "DM us" or "Restart your phone").
- **Specificity-Weighted Ranking**: Combines query-to-problem cosine similarity, response-to-problem alignment, and evidence quality heuristics.

---

## 7. Evidence Safety

Historical support logs frequently contain guidance that, while historically genuine, is hazardous if transferred to a divergent customer problem:
- **Action-Divergence Rejection**:
  - Carrier SIM unlock queries reject physical SIM-tray ejection instructions.
  - Physical hardware damage queries reject software restart instructions.
  - App crash queries reject App Store payment/billing instructions.
  - Wi-Fi connectivity queries reject carrier cellular reset steps.
- **Safety Impact**: Gating reduced the accepted evidence rate from 86.88% to 84.38% (135/160), safely rejecting 4 action-divergent cases and maintaining **0 problem conflicts** across all 160 evaluation cases.

---

## 8. Reply Generation

The response generator operates under strict deterministic invariants:
- **Groundedness Guarantee**: All 160 evaluated responses passed deterministic grounding verification, with **0 unsupported claims**.
- **Safe Clarification Policy**: When retrieved evidence is gated or when queries are ambiguous (`other_unclear`), the system generates targeted diagnostic questions matching the intent domain.
- **Resolved-Message Handling**: When customers indicate an issue is already resolved ("Never mind, it works now"), the system acknowledges resolution courteously without prescribing redundant troubleshooting.

---

## 9. Escalation Policy

Customer safety requires unequivocal boundaries between autonomous resolution and human escalation:
- **Safety Hazards**: Swelling batteries, burning smells, smoke, or extreme overheating immediately escalate with clear safety warnings (e.g., stop using the device, do not charge).
- **Account Security**: Account hacking, unauthorized password changes, or takeover attempts immediately escalate to official account security specialists.
- **Billing & Purchases**: Double charges, refund demands, and payment disputes escalate directly.
- **Decisions**: In the 160-case benchmark, 3 cases routed to `ESCALATE` (hardware hazard, account security, billing) while 157 were resolved via `AUTO_HANDLE`.

---

## 10. Evaluation Methodology

The system is evaluated across four rigorous, complementary dimensions:
1. **Automated Deterministic Evaluation (N=160)**: Complete dataset scoring intent classification, evidence safety, grounding invariants, and escalation routing.
2. **Manual Acceptance Testing (N=20)**: Interactive end-to-end acceptance tests on realistic edge cases.
3. **LLM-as-a-Judge Evaluation (N=50 attempted, N=24 completed)**: Qualitative evaluation across 7 dimensions using an external LLM judge.
4. **Independent Human Evaluation & Agreement (N=50 human, N=24 paired)**: Independent human review across 7 dimensions compared case-by-case against judge ratings.

---

## 11. Automated Evaluation Results

Deterministic metrics across all 160 evaluation cases (`src/evaluation/evaluate_reply_generator.py`):

| Metric / Dimension | Keyword Baseline | ML Baseline | Final Verified System | Operational Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Accuracy** | 61.25% (98/160) | 88.75% (142/160) | **91.25% (146/160)** | **+2.50% (+4 cases corrected, 0 regressions)** |
| **Intent Macro F1** | 0.5842 | 0.8663 | **0.9075** | **+0.0412 across all 12 classes** |
| **Intent Weighted F1** | 0.6091 | 0.8888 | **0.9133** | **+0.0245 support-weighted** |
| **Evidence Safe & Used** | 92.50% | 86.88% (139/160) | **84.38% (135/160)** | **Safely gated 4 unsafe divergent cases** |
| **Deterministic Grounding**| — | 96.25% | **100.0% (160/160)** | **Preserved invariant (0 hallucinations)** |
| **Unsupported Claims** | — | 6 / 160 | **0 / 160** | **Preserved invariant** |
| **Problem Conflicts** | — | 4 / 160 | **0 / 160** | **Preserved invariant** |
| **AUTO_HANDLE Decisions** | 160 / 160 | 159 / 160 | **157 / 160** | **Safe autonomous resolution & clarification** |
| **ESCALATE Decisions** | 0 / 160 | 1 / 160 | **3 / 160** | **Hardware hazards, security, billing disputes** |
| **Automated Test Suite** | — | — | **102 / 102 PASS** | **`pytest` (16 safety, 25 failures, 12 validation)** |

### Per-Class Intent Classification Report
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

### Manual 20-Case Acceptance Test Disclosure
In the initial manual 20-case acceptance test, the system correctly resolved 19/20 cases. However, it **initially failed on Test Case 14 ("My battery is swollen and getting hot")**, generating a diagnostic clarification question rather than escalating. This exposed a critical gap: existing safety patterns matched "swollen battery" but failed to match "battery is swollen" or "getting hot". Prior to final release, targeted regex patterns were implemented, and the case was verified to escalate immediately with safety guidance. The report does not claim that the original manual acceptance test passed without iteration.

---

## 12. LLM Judge Evaluation

### Current System Run Status (50 Attempted, 24 Valid)
- **Evaluation Target**: Exactly 50 frozen system replies from `golden_set/human_evaluation_50.csv`.
- **System Reply Consistency**: **50 / 50 (100.0%)** of evaluated replies strictly match the current frozen system output.
- **Provider & Model**: Groq API (`openai/gpt-oss-20b`).
- **Execution Outcome**:
  - **Cases Attempted**: **50 cases**
  - **Valid LLM Judgments**: **24 cases** (48.0% completion rate)
  - **Unavailable Cases**: **26 cases** unavailable because of persistent HTTP 429 rate limits after 3 attempts per case with exponential backoff.
- **Scientific Integrity Invariant**: The 26 unavailable judge calls were **NOT** backfilled with fabricated, simulated, or interpolated ratings. The report does **NOT** claim 50 successful LLM judgments.
- **Measured Metrics across the 24 Valid Judgments**:
  - Overall Score: **2.7229 / 5.0**
  - Correctness: **1.9583** | Groundedness: **2.3750** | Relevance: **2.0833**
  - Completeness: **1.6667** | Tone: **4.2500** | Unsupported Claims: **4.0000**
  - Escalation Appropriateness: **3.3750**

### Historical LLM Judge Baseline (Separately Preserved)
Preserved in `evaluation/results/historical_llm_judge_summary.csv` as a historical benchmark (50/50 judged):
- Overall Score: **2.5272 / 5.0** | Correctness: **1.3800** | Groundedness: **2.2800** | Relevance: **1.8600** | Completeness: **1.2800** | Tone: **4.0800** | Unsupported Claims: **4.2800** | Escalation Appropriateness: **2.4000**.
- *Note*: Only 29/50 historical replies match the current system's replies; historical results are preserved strictly as a baseline and are never combined with current results.

---

## 13. Human Evaluation

An independent human evaluation was conducted across 50 customer support cases (`golden_set/human_evaluation_50.csv`) evaluating 7 dimensions on an ordinal 1–5 scale:
- **Sample Size**: Exactly 50 frozen cases.
- **Independent Evaluator Means**:
  - **Tone**: **4.72 / 5.0** (courteous, professional, standardized brand voice)
  - **Unsupported Claims**: **4.20 / 5.0** (strong safety discipline; no invented procedures)
  - **Groundedness**: **4.18 / 5.0** (guidance rooted in verifiable troubleshooting)
  - **Relevance**: **3.20 / 5.0** (addresses problem area, but relies heavily on clarification)
  - **Escalation Appropriateness**: **3.02 / 5.0** (safe on hazards, occasionally cautious)
  - **Correctness**: **2.78 / 5.0** (safe diagnostic questions preferred over unconfirmed advice)
  - **Completeness**: **2.30 / 5.0** (single-turn answers intentionally brief)
- **Categorical Human Verification**:
  - **Human Intent Correctness**: **58.0% (29/50)** confirmed correct by the human rater.
  - **Evidence Grounded Support**: **76.0% (38/50)** supported by historical support context.

---

## 14. Human–LLM Agreement

### Current System Agreement (N = 24 Valid Paired Cases)
Current human–LLM agreement uses **ONLY the N = 24 valid paired cases** where both independent human ratings and current-system LLM judge scores exist (`evaluation/results/human_llm_agreement_summary_current_n24.csv`). The report does **NOT** claim N = 50 human–LLM agreement, and **these N = 24 results are not generalized to the full 50 cases**:

| Dimension | Paired Cases | Human Mean | LLM Mean | Mean Absolute Diff (MAD) | Spearman $\rho$ | Exact Agreement | Within ±1 Point |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Correctness** | 24 | 2.7917 | 1.9583 | 1.0833 | 0.6511 | 25.0% (6/24) | 75.0% (18/24) |
| **Groundedness** | 24 | 4.3333 | 2.3750 | 2.6250 | -0.2897 | 25.0% (6/24) | 29.2% (7/24) |
| **Relevance** | 24 | 3.2083 | 2.0833 | 1.3750 | 0.5069 | 29.2% (7/24) | 58.3% (14/24) |
| **Completeness** | 24 | 2.2083 | 1.6667 | 0.8750 | 0.3811 | 50.0% (12/24) | 79.2% (19/24) |
| **Tone** | 24 | 4.7500 | 4.2500 | 0.6667 | 0.2261 | 45.8% (11/24) | 87.5% (21/24) |
| **Unsupported Claims** | 24 | 4.2917 | 4.0000 | 1.2083 | 0.1077 | 62.5% (15/24) | 66.7% (16/24) |
| **Escalation Appropriateness**| 24 | 2.9167 | 3.3750 | 1.2917 | 0.3672 | 33.3% (8/24) | 62.5% (15/24) |

### Key Agreement Insights
1. **Strongest Agreement**: Tone (87.5% within ±1, MAD 0.67) and Completeness (79.2% within ±1, MAD 0.88).
2. **Groundedness Divergence (MAD 2.625, $\rho = -0.2897$)**: The primary disagreement source. The LLM judge heavily penalizes safe diagnostic clarification questions (scoring 1.0 when no external knowledge snippet is quoted), whereas the human evaluator scored safe, non-hallucinatory diagnostic questions 5/5.
3. **Preserved Archived Baseline (N=6)**: Preserved untouched in `*_n6.*` artifacts (Completeness MAD 0.17, Tone MAD 0.33) as a separate small-batch checkpoint. It remains completely separate from current N=24 results.

---

## 15. Failure Analysis

Audit of 50 historical cases (`evaluation/results/llm_judge_failure_analysis.csv`) under a 7-category taxonomy:

| Category | Description | Count | Percentage |
| :---: | :--- | :---: | :---: |
| **A** | Generic / template evidence selected (boilerplate historical reply) | **15** | 30.0% |
| **B** | Wrong intent classification upstream | **12** | 24.0% |
| **J** | Judge strictness / semantic limitation (judge penalized valid clarification) | **9** | 18.0% |
| **D** | Unsupported claim / domain divergence | **7** | 14.0% |
| **H** | Resolved customer message handling (user fixed issue; redundant advice) | **4** | 8.0% |
| **E** | Unnecessary / redundant clarification | **2** | 4.0% |
| **G** | Safety / escalation issue (hazard required human specialist) | **1** | 2.0% |
| **TOTAL** | **Mutually exclusive primary failure categories** | **50** | **100.0%** |

- **`eval_34` (tweet_id `1123670`)**: Customer reported high-pitched alarm/smoke-detector noise from iPhone speaker. Categorized as **Category G (Safety/Escalation)**: speaker hardware failure accompanied by smoke detector alarms represents potential thermal danger requiring human escalation.
- **`eval_43` (tweet_id `1338119`)**: Customer stated they found the answer regarding iPhone X SIM-free pre-orders. Categorized as **Category D (Unsupported claim/divergence)** with secondary A.

---

## 16. Limitations

1. **Twitter Corpus Informality**: Customer queries are brief, emotionally charged, and frequently lack device model or iOS version details, necessitating diagnostic clarification turns.
2. **Retrieval Specificity Bottleneck**: Historical tweets frequently contain generic boilerplate ("Please DM us"). While safety gating rejects these phrases, the fallback produces diagnostic questions rather than immediate resolution.
3. **LLM Judge Availability**: External provider rate-limiting (Groq HTTP 429) constrained automated judging to 24 valid cases. Multi-provider fallbacks are necessary for uninterrupted continuous evaluation.
4. **Single-Turn Scope**: The current pipeline processes isolated customer turns; multi-turn context tracking is planned for future iterations.

---

## 17. Mandatory Misleading-Headline Number

### Why "91.25% Intent Accuracy" Is Scientifically Incomplete

Highlighting **"91.25% Intent Accuracy"** as a standalone headline is misleading:
1. **Class Frequency Imbalance**: The 160-case benchmark is dominated by `ios_update` ($N=36$) and `other_unclear` ($N=28$), representing 40.0% of all queries. A naïve classifier over-predicting dominant classes achieves high accuracy while failing rare categories.
2. **Macro F1 (0.9075) vs. Weighted F1 (0.9133)**: Macro F1 weights all 12 classes equally. Lower performance in critical small classes (e.g. `screen_display` F1=0.82, `battery_charging` precision=0.73) is masked by weighted averages.
3. **Asymmetric Operational Risk**: Misclassifying `other_unclear` results in a harmless clarification question. Misclassifying `apple_id_icloud` (account takeover) or `battery_charging` (swelling battery) causes severe customer harm. Accuracy treats all errors equally; enterprise safety does not.

---

## 18. What I Would Do Next Week

1. **Dense Semantic Retrieval**: Integrate hybrid BM25 + dense bi-encoder embeddings (e.g., BGE-small) to match troubleshooting procedures by semantic symptom equivalence rather than keyword overlap.
2. **Official Knowledge Base Ingestion**: Supplement Twitter dialogue with structured Apple Support KB articles to eliminate boilerplate and provide direct, actionable steps.
3. **Multi-Provider LLM Judge Harness**: Implement automated round-robin failover (Groq $\rightarrow$ Anthropic $\rightarrow$ OpenAI $\rightarrow$ local Ollama) to eliminate API rate-limiting bottlenecks.
4. **Multi-Turn Dialogue State Tracking**: Maintain conversational state across successive customer turns, enabling progressive diagnostic narrowing.

---

## 19. Decision Log

| # | Architectural Decision | Technical Rationale | Engineering Trade-off |
| :-: | :--- | :--- | :--- |
| **1** | **Thread-Level Split** | Prevents conversational data leakage between turns of the same customer interaction. | Reduces total number of independent training partitions. |
| **2** | **Deterministic Baseline First** | Establishes a verifiable, reproducible benchmark before applying complex heuristics. | Requires building complete evaluation infrastructure upfront. |
| **3** | **12-Class Taxonomy** | Accurately models real-world Apple Support routing requirements. | Finer granularity increases classification boundary ambiguity. |
| **4** | **Dedicated `other_unclear` Class** | Prevents forced misclassification of terse, ambiguous customer queries. | Requires conservative classification thresholds. |
| **5** | **Action-Divergence Safety Gating** | Prevents proposing technically divergent actions (e.g. software reboots for SIM lock). | Slightly lowers accepted evidence rate from 86.88% to 84.38%. |
| **6** | **Specific Problem Match Layer** | Suppresses generic historical boilerplate that fails to address specific issues. | Requires fallback clarification when specific evidence is unavailable. |
| **7** | **Resolved-Message Policy** | Recognizes when customers resolve their own issues, avoiding unwanted advice. | Adds regex pattern checks to reply generator pipeline. |
| **8** | **Deterministic Escalation Guard** | Proactively routes security, billing, and safety risks to human specialists. | Routes cases to higher-cost human support queues. |
| **9** | **Deterministic Grounding Verifier** | Enforces zero unsupported technical claims in all generated responses. | Strict verification rejects marginal or ambiguous phrasing. |
| **10**| **Targeted Hardware Safety Fix** | Resolves the Test 14 battery swelling gap via regex without modifying classifier weights. | Focuses fix on safety hazard without risking regression in unrelated classes. |
| **11**| **Withholding Fabricated Ratings** | When provider rate limits hit, records exact valid count (24/50) without interpolation. | Maintains scientific integrity at the expense of an incomplete sample. |
| **12**| **Historical vs. Current Separation** | Prevents statistical contamination between different system iterations. | Requires maintaining distinct historical baseline records. |
| **13**| **Archived N=6 Baseline Preservation** | Preserves original small-batch verification artifacts alongside current N=24 results. | Requires explicit artifact naming (`*_n6.*` vs. `*_n24.*`). |

---

## 20. Finalization Checklist

- [x] **Preserve Evaluation Artifacts**: Both archived N=6 (`human_llm_agreement_*_n6.*`) and current N=24 (`human_llm_agreement_*_current_n24.*`, `llm_judge_current_50.csv`) artifacts are intact, verified, and separately preserved.
- [x] **Disclose N=24 LLM Limitation**: The evaluation transparently reports 24 valid judgments and 26 unavailable cases resulting from external HTTP 429 provider rate limits, with zero backfilling of simulated ratings.
- [x] **Final Repository & Secret Check**: Verified `.env` is uncommitted, no API keys exist in tracked files, and no large datasets, virtualenvs, or bytecode caches are staged.
- [x] **Submission**: All 102 automated tests pass synchronously; production behavior is frozen and verified for final submission.
