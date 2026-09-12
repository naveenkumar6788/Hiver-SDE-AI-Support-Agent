# Human Review Instructions for AI Support Responses

This document provides standardized instructions and rubrics for evaluating generated customer-support replies for `@AppleSupport` Twitter queries.

---

## 1. Overview and Core Principles

As a human reviewer, your objective is to evaluate the quality, groundedness, safety, and escalation appropriateness of the AI-generated customer support reply.

### Critical Grounding Rules:
1. **Judge ONLY the provided historical evidence.** Do NOT use your own technical knowledge to assume an instruction is grounded. A response may be 100% technically correct in the real world, but if the supplied historical evidence does not explicitly support that claim or instruction, it **fails groundedness**.
2. **Distinguish Factual Correctness from Evidence Support.**
   - Factual correctness: Does the advice make sense for the customer's problem?
   - Groundedness: Is the advice derived from the supplied historical evidence?
3. **Mark Unsupported Claims Explicitly.** If the reply includes specific actions (e.g., "reset network settings", "go to Settings > General", "restore phone") that are NOT in the evidence, mark the unsupported claims score down and list them in your notes.
4. **Evaluate Escalation Separately.** Escalation appropriateness is judged independently and is NOT factored into the overall score.
5. **Do NOT rewrite or alter the AI response.** Evaluate what the system produced as-is.

---

## 2. The 1–5 Scoring Rubric

All dimensions use a **1–5 scale** where **higher is always better**:
- **1** = Very Poor
- **2** = Poor
- **3** = Acceptable / Borderline
- **4** = Good
- **5** = Excellent

---

### Dimension 1: Correctness (`human_correctness`)
*Does the reply correctly address the customer's problem?*
- **5**: Accurately addresses the customer's specific issue or appropriately asks necessary diagnostic questions.
- **4**: Generally addresses the problem with minor ambiguity.
- **3**: Partially addresses the problem or gives overly generic assistance.
- **2**: Misses key aspects of the problem or misinterprets the user's issue.
- **1**: Completely incorrect, dangerous advice, or hallucinated resolution.

---

### Dimension 2: Groundedness (`human_groundedness`)
*Are the claims, directives, and troubleshooting instructions supported by the supplied historical evidence?*
- **5**: Every instruction and claim is directly derived from the supplied evidence, OR if no evidence was used, the reply safely requests clarifying info without hallucinating steps.
- **4**: Mostly grounded with minor stylistic paraphrasing that doesn't add unverified claims.
- **3**: Some advice is plausible but not clearly present in the evidence.
- **2**: Reply gives specific technical instructions not found in the evidence.
- **1**: Pure hallucination; claims contradict or bear zero relation to the evidence.

---

### Dimension 3: Relevance (`human_relevance`)
*Does the response directly address the customer's issue without unnecessary or off-topic content?*
- **5**: Directly targeted at the customer's stated issue with zero fluff.
- **4**: Relevant with slight verbosity.
- **3**: Somewhat relevant, but includes tangential information.
- **2**: Mostly off-topic or replies to an issue the customer didn't mention.
- **1**: Completely irrelevant.

---

### Dimension 4: Completeness (`human_completeness`)
*Does it provide enough useful guidance or ask necessary diagnostic questions?*
- **5**: Provides a complete first-contact response (either actionable steps or asks for device/OS version).
- **4**: Gives sufficient guidance, though another clarifying question would help.
- **3**: Minimal guidance provided.
- **2**: Incomplete response that leaves the customer stranded without next steps.
- **1**: Empty, fragmented, or unhelpful.

---

### Dimension 5: Tone (`human_tone`)
*Is it professional, polite, concise, and appropriate for official Apple customer support on Twitter?*
- **5**: Empathetic, polite, concise, highly professional ("We'd be glad to help...").
- **4**: Professional and polite, slightly robotic or standard.
- **3**: Neutral, abrupt, or mildly verbose.
- **2**: Impolite, overly blunt, or unprofessional.
- **1**: Rude, offensive, or wildly inappropriate.

---

### Dimension 6: Unsupported Claims (`human_unsupported_claims`)
*Does the reply contain claims, promises, or steps not backed by evidence?*
- **5 = Fully Supported**: Zero unsupported claims or directives.
- **4 = Mostly Supported**: Minor unverified assertion with zero technical risk.
- **3 = Some Questionable Claims**: Mentions a feature or setting without clear evidence backing.
- **2 = Several Unsupported Claims**: Directs the user to perform troubleshooting steps not in the evidence.
- **1 = Many Unsupported Claims**: Extensive hallucinated technical instructions, guarantees, or refund promises.

---

### Dimension 7: Escalation Appropriateness (`human_escalation_appropriateness`)
*Is the decision (AUTO_HANDLE vs ESCALATE) appropriate given the customer's sentiment, severity, and complexity?*
- **5**: Perfectly matches (e.g., highly frustrated/legal/billing escalated; standard troubleshooting auto-handled).
- **4**: Reasonable decision with minor borderline traits.
- **3**: Acceptable but either slightly cautious or slightly lenient.
- **2**: Poor decision (e.g., high-risk hardware issue auto-handled, or trivial greeting escalated).
- **1**: Severe mismatch (e.g., auto-handling account takeover or battery swelling).

---

### Overall Score (`human_overall_score`)
Compute the **mean of dimensions 1 through 6**:
$$\text{overall\_score} = \frac{\text{correctness} + \text{groundedness} + \text{relevance} + \text{completeness} + \text{tone} + \text{unsupported\_claims}}{6}$$
*(Do NOT include escalation appropriateness in the overall score calculation.)*

---

## 3. Illustrative Examples

### Example A: Grounded & Safe Auto-Handle
- **Customer**: "My battery is draining quickly on iOS 11."
- **Evidence**: "Check Settings > Battery to see which apps are consuming battery."
- **AI Reply**: "We'd like to help. Please check Settings > Battery to see which apps are using the most battery. Let us know what you find!"
- **Decision**: AUTO_HANDLE
- **Scores**: Correctness=5, Groundedness=5, Relevance=5, Completeness=4, Tone=5, Unsupported Claims=5. Escalation Appropriateness=5. Overall = 4.83.

### Example B: Technically Correct but UNGROUNDED (Hallucination)
- **Customer**: "My SIM isn't recognized."
- **Evidence**: "We understand you have SIM issues. Let us know your carrier in DM."
- **AI Reply**: "Please remove your SIM tray with a paperclip, inspect the golden contacts, and clean it with alcohol."
- **Decision**: AUTO_HANDLE
- **Scores**: Correctness=3, Groundedness=1 (cleaning with alcohol was NOT in evidence!), Relevance=4, Completeness=4, Tone=4, Unsupported Claims=1 (unsupported directives: remove tray, clean with alcohol). Escalation Appropriateness=3. Overall = 2.83.

### Example C: Safe Fallback with Insufficient Evidence
- **Customer**: "My device is acting weird."
- **Evidence**: NONE (rejected or below threshold)
- **AI Reply**: "I'd like to make sure we point you in the right direction. Could you tell us what exactly is happening, your device model, and your current iOS version?"
- **Decision**: ESCALATE
- **Scores**: Correctness=5, Groundedness=5 (no evidence, safely asked for info without hallucinating), Relevance=5, Completeness=4, Tone=5, Unsupported Claims=5. Escalation Appropriateness=5. Overall = 4.83.
