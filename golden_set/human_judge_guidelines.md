# Human Annotation Guidelines for Customer Support Response Evaluation

This document defines the official, independent human annotation protocol for evaluating AI-generated customer-support responses on the AppleSupport dataset.

---

## 1. Core Principles & Rules

> [!IMPORTANT]
> **Rate independently from the LLM judge. Do not attempt to match the model's score.**  
> Your ratings must represent your genuine, unassisted human evaluation of the support response.

> [!NOTE]
> **If you are uncertain, record your best independent judgment and explain the reason in `human_notes`.**

### Essential Ground Rules
1. **Blind Evaluation**: Evaluate responses using [`golden_set/human_annotation_form.csv`](file:///d:/Hiver-SDE-AI-Support-Agent/golden_set/human_annotation_form.csv) or [`golden_set/human_judge_50_blind.csv`](file:///d:/Hiver-SDE-AI-Support-Agent/golden_set/human_judge_50_blind.csv). Do not look at or consider automated judge scores.
2. **Context-Driven**: Read the `customer_message`, understand the `expected_intent`, inspect the `evidence_text` (if provided), and evaluate the `generated_reply` and `escalation_decision`.
3. **Safe Refusals & Clarifications**: Do not penalize a safe diagnostic clarification or safe escalation merely because it does not provide an immediate technical fix when customer information is underspecified or historical evidence is missing.
4. **Substance Over Fluency**: Do not award high scores to answers that sound confident or polished if they provide incorrect technical guidance, address the wrong problem, or make unsubstantiated claims.

---

## 2. Rating Scale (1 to 5)

All dimensions use a standardized 1–5 integer scale:

- **1 = Very Poor**
- **2 = Poor**
- **3 = Acceptable / Mixed**
- **4 = Good**
- **5 = Excellent**

---

## 3. The Six Evaluation Dimensions

### 1. Correctness
**Core Question**: *Does the response correctly address the customer's actual problem?*
- **1 (Very Poor)**: Completely incorrect, dangerous, or harmful advice that would worsen the problem or cause data loss.
- **2 (Poor)**: Substantially incorrect or provides troubleshooting for an incompatible issue/operating system.
- **3 (Acceptable/Mixed)**: Partially correct guidance, but contains minor inaccuracies or slight confusion.
- **4 (Good)**: Mostly correct guidance with only negligible ambiguities.
- **5 (Excellent)**: Completely accurate, safe, and appropriate resolution or safe refusal/clarification.

### 2. Groundedness
**Core Question**: *Is the response supported by the supplied historical evidence?*
- **1 (Very Poor)**: Completely unsupported or contradicts the supplied historical evidence.
- **2 (Poor)**: Weakly supported; relies largely on ungrounded claims.
- **3 (Acceptable/Mixed)**: Partially supported by evidence, but introduces key unsupported details.
- **4 (Good)**: Mostly supported by evidence, with only standard conversational framing added.
- **5 (Excellent)**: Fully supported by the provided evidence (or standard safe diagnostic clarification fallback).

### 3. Relevance (Specific Problem Match)
**Core Question**: *Does it directly address the customer's specific issue rather than a nearby/general issue?*
- **1 (Very Poor)**: Unrelated to the customer's actual issue (e.g. general audio settings for a physically broken headphone jack).
- **2 (Poor)**: Mostly unrelated; belongs to the broad topic but targets a completely different failure mode.
- **3 (Acceptable/Mixed)**: Partially relevant; touches the issue but misses key nuances or specific symptoms.
- **4 (Good)**: Mostly relevant; directly addresses the specific technical problem.
- **5 (Excellent)**: Directly and precisely targets the exact issue and symptoms described by the customer.

### 4. Completeness
**Core Question**: *Does it provide enough useful information to resolve or appropriately handle the request?*
- **1 (Very Poor)**: Fails to provide useful help; misses the core issue entirely (e.g. empty link placeholder).
- **2 (Poor)**: Major information or crucial steps missing.
- **3 (Acceptable/Mixed)**: Partially useful; provides basic guidance or general questions, but leaves noticeable gaps.
- **4 (Good)**: Sufficiently complete; provides actionable troubleshooting or clear diagnostic inquiries.
- **5 (Excellent)**: Complete for the information available; offers thorough guidance or targeted diagnostic questions.

### 5. Tone
**Core Question**: *Is it professional, clear, empathetic, and appropriate for customer support?*
- **1 (Very Poor)**: Inappropriate, rude, dismissive, or robotic tone.
- **2 (Poor)**: Poor tone; curt, cold, or confusingly phrased.
- **3 (Acceptable/Mixed)**: Acceptable; neutral and functional support tone.
- **4 (Good)**: Professional, polite, clear, and helpful.
- **5 (Excellent)**: Very professional, empathetic, warm, and highly supportive.

### 6. Unsupported Claims
**Core Question**: *Does the response avoid claims that are not supported by the evidence or customer context?*
- **1 (Very Poor)**: Many unsupported claims; substantial fabrication of facts, links, or procedures.
- **2 (Poor)**: Several unsupported claims not found in the evidence.
- **3 (Acceptable/Mixed)**: Some unsupported claims or unverified assertions.
- **4 (Good)**: Almost completely supported; only minor harmless phrasing without strict evidence backing.
- **5 (Excellent)**: Fully supported / no unsupported claims (100% grounded or safe template refusal).

---

## 4. Human Annotation Workflow

1. Open [`golden_set/human_annotation_form.csv`](file:///d:/Hiver-SDE-AI-Support-Agent/golden_set/human_annotation_form.csv).
2. For each row from `eval_01` to `eval_50`:
   - Read `customer_message` and note `expected_intent`.
   - Inspect `evidence_text` (if evidence was used).
   - Review `generated_reply` and `escalation_decision`.
   - Enter your six independent 1–5 integer scores in:
     - `human_correctness`
     - `human_groundedness`
     - `human_relevance`
     - `human_completeness`
     - `human_tone`
     - `human_unsupported_claims`
   - If uncertain about any case, record your best independent judgment and document the rationale in `human_notes`.
3. Save the file.
4. Validate your annotations:
   ```bash
   python src/evaluation/validate_human_annotations.py
   ```
5. Run the agreement analysis:
   ```bash
   python src/evaluation/human_agreement.py
   ```
