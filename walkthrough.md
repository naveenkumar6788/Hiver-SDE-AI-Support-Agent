# Hiver AI Support Agent — Engineering Walkthrough

This document walks through the architecture, design choices, implementation details, evaluation results, and failure modes of the customer support agent built for the `@AppleSupport` Twitter domain.

---

## 1. What I Built

I built a deterministic, safety-focused support agent designed to handle inbound Twitter/X support inquiries. Rather than relying on an unconstrained large language model that might generate plausible-sounding but incorrect technical steps, I built a pipeline that:

1. Checks whether the inquiry is a sensitive risk (swollen battery, hacked account, payment dispute) and immediately routes it to a human specialist.
2. Classifies the query into one of 12 operational support intents.
3. Retrieves relevant historical support interactions from a corpus of 117,000+ real Apple Support tweets.
4. Checks whether the retrieved advice actually matches the customer's specific problem (rejecting mismatched actions).
5. Generates grounded troubleshooting guidance when safe evidence exists, or falls back to targeted diagnostic clarification questions when information is missing or ambiguous.
6. Acknowledges already-resolved issues without offering redundant troubleshooting advice.

---

## 2. Dataset Preparation & Conversation Reconstruction

### Selecting AppleSupport Data
I used the public Kaggle Customer Support on Twitter dataset. The full dataset is roughly 500 MB and covers multiple brands. I filtered it down specifically to the `@AppleSupport` domain, resulting in 117,076 customer inquiries paired with official Apple Support responses.

### Thread-Level Reconstruction
In conversational customer service, a common mistake is splitting individual messages randomly into train and test sets. When turn $t$ is in the training index and turn $t+1$ of the same dialogue is in the test set, the model looks artificially good because of conversational data leakage.

To avoid this, I reconstructed conversation trees using `tweet_id` and `in_response_to_tweet_id`. All tweets belonging to the same root `conversation_id` were kept together. When creating the evaluation set, complete conversation threads were held out so that no turn from an evaluated conversation existed in the retrieval database.

---

## 3. Defining the 12 Support Intents

Customer inquiries on Twitter are noisy and varied. I defined 12 distinct operational intents:

1. `app_problems`: Third-party app crashes, freezes, and bugs.
2. `app_store_downloads`: App Store download failures, purchase errors, app updates.
3. `apple_id_icloud`: Apple ID logins, two-factor authentication, passwords, iCloud sync.
4. `apple_music_itunes`: Music playback, library sync, iTunes purchases.
5. `audio_speaker`: Crackling speakers, microphone failure, call volume.
6. `battery_charging`: Rapid drain, slow charging, overheating cables/ports.
7. `calls_cellular`: Dropped calls, carrier reception, cellular data, SIM activation.
8. `device_hardware`: Physical damage, broken screens, stuck buttons, water damage.
9. `ios_update`: iOS installation errors, post-update bugs, storage issues.
10. `screen_display`: Unresponsive touchscreens, display lines, brightness bugs.
11. `wifi_connectivity`: Wi-Fi disconnection, network password issues, router drops.
12. `other_unclear`: Extremely short messages, greetings, vague complaints, or thank-yous.

I explicitly included `other_unclear` so the model would not force ambiguous messages into technical categories.

The classifier uses a hybrid approach:
- A calibrated TF-IDF Logistic Regression model trained on domain examples.
- 16 high-precision disambiguation rules for edge cases where surface words are misleading (e.g. "SIM tray won't open" has the word "SIM" but is a physical hardware issue, not a cellular carrier issue).

On the 160-case evaluation set, this achieved **91.25% accuracy**, **0.9075 Macro F1**, and **0.9133 Weighted F1**.

---

## 4. How Retrieval Works

Once the intent is predicted, retrieval is restricted to historical support cases within that same intent. This intent-constrained retrieval immediately eliminates cross-domain errors (such as retrieving Wi-Fi troubleshooting for battery problems).

Within the candidate pool, the retriever scores historical pairs using:
- TF-IDF cosine similarity between the incoming query and historical customer messages.
- Specificity scoring that rewards actionable diagnostic steps while penalizing generic boilerplate (such as pure "Please DM us" responses).

---

## 5. Evidence Safety Gate (Action Divergence Checking)

During early testing, I noticed a major issue: **retrieved tweets often looked relevant at a keyword level, but the recommended action was completely wrong for the customer's actual problem.**

For example:
- A customer asking how to **carrier-unlock** a phone retrieved historical advice about **ejecting the physical SIM tray**.
- A customer reporting an **app crash** retrieved advice about **checking App Store payment methods**.
- A customer reporting a **Wi-Fi drop** retrieved advice on **resetting carrier network settings**.

To fix this, I implemented an evidence safety gate (`is_evidence_safe` in `src/agent/reply_generator.py`):
1. **Domain Compatibility**: Ensures the query and retrieved guidance belong to compatible problem domains.
2. **Specific Problem Match**: Checks whether specific sub-problems (e.g., auto-play music vs. library sync) match.
3. **Action Divergence Rules**: Hard rules that reject incompatible technical actions.
4. **Boilerplate Detection**: Flags and suppresses replies that only contain redirection phrases ("Please DM us").

In the 160-case benchmark, this safety gate accepted 135 cases (84.38%) and rejected 25 cases, safely pruning 4 action-divergent cases and maintaining **0 problem conflicts**.

---

## 6. Reply Generation & Deterministic Grounding

The reply generator (`ReplyGenerator`) constructs responses under strict grounding rules:

1. **If safe evidence exists**: It extracts the verified troubleshooting procedure from the historical Apple Support response.
2. **If evidence is rejected or the query is unclear**: It does not invent advice. Instead, it generates a targeted diagnostic question specific to the predicted domain (e.g., asking for the iOS version and error message for update issues).
3. **If the customer says the issue is resolved** (e.g., "Fixed it, thanks!"): It acknowledges the resolution courteously without prescribing unnecessary troubleshooting steps.

Every generated reply is passed through a deterministic grounding verifier (`perform_deterministic_grounding_check`). In our 160-case evaluation benchmark, **100% of responses passed grounding verification with 0 unsupported claims**.

---

## 7. Escalation Policy

Some issues should never be automated. I built a deterministic regex-based escalation guard (`check_sensitive_topic` in `src/agent/escalation.py`) that runs before any retrieval or reply generation:

- **Hardware Safety Hazards**: Battery swelling, burning smell, smoking device, or dangerous overheating. The agent tells the user to stop using the device and charge it no further, routing them immediately to official Apple Support.
- **Account Security**: Hacked accounts, unauthorized password changes, stolen credentials.
- **Billing Disputes**: Fraudulent charges, double billings, refund requests.

In the 160-case benchmark, 3 cases routed to `ESCALATE` and 157 were safely handled autonomously.

---

## 8. What I Learned from Failures

Building this system involved several iterations where edge cases exposed flaws in earlier logic:

### Failure 1: The Battery Swelling Edge Case (Manual Acceptance Test 14)
During an interactive 20-case acceptance test, the system handled 19/20 cases correctly. However, **Test Case 14 ("My battery is swollen and getting hot") failed to escalate**. Instead, the agent generated a diagnostic clarification question asking about charging habits.

When I investigated the failure, I found the existing regex matched `"swollen battery"` as a compound phrase, but did not match `"battery is swollen"` or `"getting hot"`. Because battery expansion is a physical safety hazard, this was a critical flaw. I expanded the safety patterns to include split phrasings and thermal hazard keywords. I then added regression tests in `tests/test_adversarial_safety.py` and `tests/test_known_failure_cases.py` to ensure physical hazards always escalate.

### Failure 2: Carrier Unlock vs. SIM Tray Ejection
Before adding the action divergence checks, a query about carrier unlocking retrieved a support tweet explaining how to insert a paperclip into the SIM tray. While both dealt with "SIM cards", the action was physically irrelevant to carrier network provisioning. Adding action-divergence conflict rules resolved this across all SIM-related test cases.

### Failure 3: The LLM Judge Groundedness Divergence
When evaluating 50 system replies with an independent human rater and comparing them against the LLM judge (Groq `openai/gpt-oss-20b` across 24 completed cases), I observed a significant divergence on **Groundedness** (Human mean: 4.33 vs. LLM mean: 2.38; Spearman $\rho = -0.29$):
- When a customer's query was too vague for immediate troubleshooting, the system safely generated a diagnostic clarification question.
- The **human evaluator** rated these safe, non-hallucinatory clarifications 5/5 for Groundedness, recognizing that asking for clarification is the correct engineering choice.
- The **LLM judge** gave these cases 1/5, assuming that any response without a concrete troubleshooting step lacked grounded evidence.

This demonstrated why human evaluation remains necessary alongside automated LLM judging.

---

## 9. Evaluation Summary

| Dimension | Scope | Key Result | Notes |
| :--- | :---: | :---: | :--- |
| **Deterministic Benchmark** | 160 cases | **91.25% Intent Accuracy** | 0.9075 Macro F1, 100% Grounded, 0 Unsupported claims |
| **Human Evaluation** | 50 cases | **4.72 Tone, 4.18 Groundedness** | 58% Intent correctness, 76% evidence supported |
| **LLM Judge** | 50 attempted | **24 valid / 26 unavailable** | Unavailable cases caused by Groq HTTP 429 rate limits |
| **Human–LLM Agreement** | 24 paired cases | **87.5% Tone (±1), 79.2% Completeness (±1)** | Groundedness divergence due to clarification questions |
| **Automated Test Suite** | 102 tests | **102 / 102 Passed** | 16 adversarial safety, 25 regression, 12 validation |

---

## 10. How to Run the Project

### 1. Run the Test Suite
```bash
pytest -v
```
All 102 tests should pass in approximately 10 seconds.

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
This opens an interactive interface where you can test queries, view predicted intents, inspect retrieved evidence, examine safety checks, and verify escalation routing.

### 3. Run Deterministic Evaluation
```bash
python -m src.evaluation.evaluate_reply_generator
```
This evaluates the 160 golden cases and outputs metrics to `evaluation/results/`.

### 4. Run Human vs. LLM Agreement Analysis
```bash
python -m src.evaluation.human_agreement
```
This runs the statistical agreement analysis comparing valid human ratings against LLM judge ratings.
