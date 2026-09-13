# Hiver SDE AI Support Agent

An automated customer support agent for social media inquiries, built and evaluated historical **@AppleSupport** customer-support conversations from Twitter. 

The system classifies customer intent, retrieves relevant historical support cases, validates that retrieved guidance actually fits the customer's problem (rejecting mismatched actions), generates grounded replies, and escalates safety hazards, billing disputes, and account takeovers to human specialists.

---

## 1. What is this?

This repository contains a deterministic, safety-focused customer support pipeline designed for high-volume, public customer support on Twitter/X.

Rather than letting an unconstrained language model generate open-ended answers that risk technical hallucination, this system uses an intent-driven pipeline:
1. It predicts what the user is asking about.
2. It retrieves similar historical support conversations.
3. It validates that the historical advice is safe and relevant to the customer's specific problem.
4. It generates either grounded troubleshooting steps or a safe diagnostic clarification question.
5. It routes dangerous or sensitive issues (e.g., swollen batteries or compromised accounts) directly to human support.

---

## 2. What problem does it solve?

Automating customer support on social platforms is challenging for three main reasons:

1. **Terse, noisy messages**: Customer tweets are typically 10–25 words, filled with slang, missing device models, and lacking iOS version numbers.
2. **High cost of technical hallucinations**: Giving the wrong technical advice damages user trust and can cause physical damage. For example, suggesting a software reset for a physical SIM tray failure, or suggesting a network reset for a billing problem, wastes user time and frustrates customers.
3. **Safety and security boundaries**: Issues involving compromised Apple IDs, fraudulent purchases, or swelling batteries should never receive automated canned responses; they must immediately route to human specialists.

This project solves these issues by enforcing strict evidence gating (rejecting action-divergent advice), generating safe clarifications when evidence is insufficient, and hard-coding escalation policies for critical risks.

---

## 3. Why AppleSupport?

The Kaggle Customer Support on Twitter dataset includes over 117,000 real interactions from the `@AppleSupport` account. This domain is an ideal benchmark because:
- It covers a broad set of technical domains (iOS updates, battery, audio, display, Apple ID, cellular, Wi-Fi, hardware).
- Real users frequently report urgent safety issues (overheating chargers, swollen batteries).
- Historical agent replies vary widely from specific troubleshooting steps to generic "send us a DM" boilerplate, requiring explicit evidence quality filtering.

---

## 4. How does it work?

The pipeline processes each incoming customer message through five stages:

```
Customer Message
       │
       ▼
[1. Escalation Guard]        --> Checks for safety hazards, account security, billing disputes
       │                          (If sensitive -> immediately routes to ESCALATE)
       ▼
[2. Intent Classification]   --> Predicts 1 of 12 categories (TF-IDF + disambiguation rules)
       │
       ▼
[3. Historical Retrieval]    --> Finds candidate support cases matching the predicted intent
       │
       ▼
[4. Evidence Safety Gate]    --> Checks domain match, specific problem match, and action divergence
       │                          (Rejects advice that contradicts or mismatches the query)
       ▼
[5. Grounded Reply]          --> If safe evidence: extracts verified troubleshooting guidance
                                  If rejected/unclear: generates safe diagnostic questions
                                  If customer resolved: sends courteous acknowledgement
```

---

## 5. Main Components

The codebase is organized into modular packages:

- `src/data/`: Reconstructs conversation threads from raw tweets, ensuring dialogue-level isolation.
- `src/intent/`: 12-class intent classifier combining TF-IDF logistic regression with domain disambiguation rules.
- `src/retrieval/`: Candidate retrieval index, query normalization, and evidence scoring.
- `src/agent/`: Evidence safety gate (`is_evidence_safe`), reply generator, and escalation logic.
- `src/evaluation/`: Evaluation harnesses for deterministic testing, LLM judge scoring, and human agreement analysis.
- `app.py`: Streamlit web interface for interactive query testing and pipeline inspection.
- `tests/`: 102 automated tests covering unit logic, known failure cases, adversarial safety, and evaluation validation.

---

## 6. What is intentionally not automated?

The system explicitly refuses to auto-handle the following sensitive categories, immediately routing them to `ESCALATE`:

1. **Hardware Safety Hazards**: Reports of swollen batteries, burning smells, smoking devices, or extreme heat. The agent tells the user to stop using the device and routes them to official support.
2. **Account Security & Compromise**: Reports of hacked Apple IDs, unauthorized password changes, or stolen credentials. The agent directs the user to official account recovery channels.
3. **Billing & Financial Transactions**: Disputes about unexpected charges, double billings, or refund demands. These require access to private account records and are routed to human specialists.

---

## 7. How was it evaluated?

The system was evaluated across four complementary evaluation tracks:

### A. Deterministic Benchmark (160 Cases)
Evaluated across **160 evaluation examples** (44 human-reviewed + 116 labeled through a structured audit process):
- **Intent Accuracy**: **91.25%** (146 / 160)
- **Intent Macro F1**: **0.9075** (equal weight across all 12 classes)
- **Intent Weighted F1**: **0.9133** (support-weighted)
- **Evidence Used / Safe Rate**: **84.38%** (135 / 160) — gating safely rejected 4 action-divergent cases
- **Deterministic Grounding Pass Rate**: **100.0%** (160 / 160)
- **Unsupported Claims**: **0.0%** (0 / 160)
- **Problem Conflicts Detected**: **0** (0 / 160)
- **Routing Decisions**: **157 AUTO_HANDLE / 3 ESCALATE**
- **Generation Errors**: **0**

### B. Independent Human Evaluation (50 Cases)
An independent human rater evaluated 50 frozen system responses across 7 dimensions on a 1–5 scale:
- **Tone**: **4.72 / 5.0** (professional and polite)
- **Unsupported Claims**: **4.20 / 5.0** (strong safety adherence; no invented claims)
- **Groundedness**: **4.18 / 5.0** (guidance backed by verifiable steps)
- **Relevance**: **3.20 / 5.0** (addresses problem area, though often via clarification)
- **Escalation Appropriateness**: **3.02 / 5.0** (safe on hazards, cautious on edge cases)
- **Correctness**: **2.78 / 5.0** (safe diagnostic questions preferred over unconfirmed guesses)
- **Completeness**: **2.30 / 5.0** (single-turn responses are intentionally concise)
- **Categorical Human Ratings**: **58.0%** Human Intent Correctness, **76.0%** Evidence Supported.

### C. LLM-as-a-Judge Evaluation (50 Attempted, 24 Completed)
- Evaluated via Groq API (`openai/gpt-oss-20b`) on the exact 50 frozen system responses.
- **24 cases completed successfully**; **26 cases were unavailable due to persistent HTTP 429 provider rate limits**.
- In accordance with scientific integrity, unavailable cases were **not** backfilled with simulated or fabricated scores.
- Across the 24 valid judgments: Overall: 2.72 / 5.0, Correctness: 1.96, Groundedness: 2.38, Relevance: 2.08, Completeness: 1.67, Tone: 4.25, Unsupported Claims: 4.00, Escalation Appropriateness: 3.38.

### D. Human vs. LLM Agreement (N = 24 Paired Cases)
- Computed strictly on the 24 paired cases with both valid human and LLM ratings.
- **Strongest Agreement**: Tone (87.5% within ±1, MAD 0.67) and Completeness (79.2% within ±1, MAD 0.88).
- **Primary Divergence**: Groundedness (MAD 2.625, $\rho = -0.29$). The LLM judge heavily penalized safe diagnostic questions (scoring 1.0 when no troubleshooting procedure was provided), whereas the human evaluator rated safe, non-hallucinatory clarifications 5/5.
- The archived N=6 exploratory agreement checkpoint is preserved separately in `evaluation/results/` as historical context.

---

## 8. Important Limitations

1. **Twitter Message Brevity**: Customer inquiries on Twitter are very short. Without the device model or OS version, the system often has to ask for clarification instead of providing an immediate fix.
2. **Historical Boilerplate**: Many real Apple Support tweets say "Please DM us" or "Restart your phone". Our evidence gate filters out this boilerplate, which reduces the number of immediate answers and increases clarification fallbacks.
3. **LLM Judge Availability**: External API rate limits prevented evaluating all 50 cases with the LLM judge. Future work should use multi-provider failover.
4. **Single-Turn Architecture**: The current implementation handles isolated customer turns. Multi-turn dialogue state tracking would allow progressive diagnostic narrowing.

---

## 9. How to Run

### Setup & Reproduction

```bash
# 1. Clone repository (use main branch)
git clone -b main https://github.com/naveenkumar6788/Hiver-SDE-AI-Support-Agent.git
cd Hiver-SDE-AI-Support-Agent

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Tests
```bash
# Run the complete test suite (102 tests)
pytest -v
```

### Run the Interactive App
```bash
# Launch the Streamlit web demo
streamlit run app.py
```

### Run Evaluation Scripts
```bash
# Run the 160-case deterministic evaluation benchmark
python -m src.evaluation.evaluate_reply_generator

# Run the human-LLM agreement analysis on valid paired cases
python -m src.evaluation.human_agreement
```

### Dataset & Offline Pipeline Reproduction (Optional)
Pre-processed evaluation golden sets and benchmark summaries are committed directly in `golden_set/` and `evaluation/results/`, allowing the full test suite and evaluations to run out of the box.

The large raw Twitter dataset (~500 MB) and processed dialogue threads (`data/processed/`, ~114 MB) are intentionally excluded from Git tracking to prevent repository bloat. To rebuild the historical retrieval index from raw data:
1. Place raw `twcs.csv` into `data/raw/twcs/`.
2. Reconstruct dialogue threads: `python -m src.data.reconstruct_conversations`.
3. Validate and build threads: `python -m src.data.validate_conversations`.

---

## 10. Key Files & Directory Structure

```text
Hiver-SDE-AI-Support-Agent/
├── app.py                                # Streamlit live demonstration interface
├── requirements.txt                      # Frozen Python dependencies
├── README.md                             # Project overview and run instructions
├── walkthrough.md                        # Developer implementation walkthrough
├── report/
│   └── report.md                         # Detailed technical evaluation report
├── src/
│   ├── data/
│   │   └── reconstruct_conversations.py  # Thread-level dialogue reconstruction
│   ├── intent/
│   │   ├── classifier.py                 # 12-class intent classifier & rules
│   │   └── baselines.py                  # Baseline keyword classifiers
│   ├── retrieval/
│   │   ├── retriever.py                  # TF-IDF candidate retriever
│   │   └── evidence_relevance.py         # Evidence scoring & conflict checks
│   ├── agent/
│   │   ├── reply_generator.py            # Evidence safety gate & reply builder
│   │   └── escalation.py                 # Sensitive topic escalation guard
│   └── evaluation/
│       ├── evaluate_reply_generator.py   # 160-case deterministic benchmark
│       ├── run_current_llm_judge_50.py   # LLM judge execution script
│       ├── human_agreement.py            # Human vs. LLM agreement calculator
│       └── validate_human_annotations.py # Human annotation validation harness
├── golden_set/
│   ├── golden_human_160.csv              # Frozen 160-case evaluation set
│   └── human_evaluation_50.csv           # 50-case independent human annotations
├── evaluation/results/                   # Evaluation CSV summaries and reports
└── tests/
    ├── test_adversarial_safety.py        # Safety and adversarial query tests
    ├── test_known_failure_cases.py       # Regression tests for known failure modes
    └── test_human_evaluation_validation.py # Data integrity and validation tests
```
