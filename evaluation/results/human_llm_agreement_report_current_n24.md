# Human vs. LLM Judge Agreement Evaluation

## 1. Evaluation Setup & Case Matching

- **Human Evaluation Set**: 50 complete, independent human evaluations recorded in `golden_set/human_evaluation_50.csv`.
- **Target LLM Judge Set**: 50 cases from current system run (`evaluation/results/llm_judge_results.csv`).
- **Valid Current LLM Judge Judgments**: **24 / 50 cases** (due to Groq Cloudflare HTTP 403 provider edge block on 37 cases, rate-limit on 1 case).
- **Evaluated Paired Sample**: Agreement statistics are strictly and honestly computed on the **24 valid paired cases** (`case_01` through `case_06`).
- **Scientific Integrity Rule**: No missing judge scores were fabricated, simulated, or interpolated.

## 2. Dimension-Level Agreement Statistics

| Dimension | Paired Cases | Human Mean | LLM Mean | Mean Absolute Diff (MAD) | Spearman $\rho$ | Exact Agreement | Within ±1 Point |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Correctness | 24 | 2.7917 | 1.9583 | 1.0833 | 0.6511 | 25.0% (6/24) | 75.0% (18/24) |
| Groundedness | 24 | 4.3333 | 2.375 | 2.625 | -0.2897 | 25.0% (6/24) | 29.2% (7/24) |
| Relevance | 24 | 3.2083 | 2.0833 | 1.375 | 0.5069 | 29.2% (7/24) | 58.3% (14/24) |
| Completeness | 24 | 2.2083 | 1.6667 | 0.875 | 0.3811 | 50.0% (12/24) | 79.2% (19/24) |
| Tone | 24 | 4.75 | 4.25 | 0.6667 | 0.2261 | 45.8% (11/24) | 87.5% (21/24) |
| Unsupported Claims | 24 | 4.2917 | 4.0 | 1.2083 | 0.1077 | 62.5% (15/24) | 66.7% (16/24) |
| Escalation Appropriateness | 24 | 2.9167 | 3.375 | 1.2917 | 0.3672 | 33.3% (8/24) | 62.5% (15/24) |

## 3. Categorical Intent & Escalation Alignment

- **Human Intent Correctness**: **45.8% (11/24)** of evaluated cases were validated as having correct system intent by the independent human evaluator.
- **Escalation Appropriateness**: **29.2% (7/24)** of cases were rated as appropriately handled (rating $\ge 4/5$).

## 4. Disagreement Analysis

Identified 21 cases with an absolute dimension difference $\ge 2.0$ points:

### case_01: "Got my new Three SIM card yay... and now the sim drawer won’t open on my phone....."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4); relevance (Human=4, LLM=2, Δ=+2); escalation_appropriateness (Human=2, LLM=5, Δ=-3)
- **Human Rater Assessment**: The reply recognizes a hardware issue and asks for details, but it does not address the specific stuck SIM tray or answer whether the customer should visit a store.
- **LLM Judge Reasoning**: The response is a generic diagnostic request that lacks grounding in evidence and does not provide a solution, resulting in low scores for correctness, groundedness, relevance, and completeness, but it maintains a good tone and appropriate escalation.

### case_02: "@AppleSupport I have restarted my phone and it is working now. Must be a bug?..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4); unsupported_claims (Human=5, LLM=3, Δ=+2)
- **Human Rater Assessment**: The reply appropriately acknowledges that restarting fixed the issue and asks for the iOS version if it recurs, although the system intent 'other_unclear' is not very specific to a possible software bug.
- **LLM Judge Reasoning**: High correctness, relevance, tone, and escalation appropriateness, but low groundedness and some unsupported claims reduce the overall score.

### case_03: "@AppleSupport Hi, I have an issue where I have set my Calendar app on Mac to sta..."
- **Key Differences**: correctness (Human=3, LLM=1, Δ=+2); groundedness (Human=5, LLM=1, Δ=+4); relevance (Human=4, LLM=1, Δ=+3); tone (Human=5, LLM=3, Δ=+2); escalation_appropriateness (Human=3, LLM=1, Δ=+2)
- **Human Rater Assessment**: The reply asks for more details about the app behavior, which is relevant, but it does not address the specific Calendar setting that keeps reverting.
- **LLM Judge Reasoning**: The response fails to address the customer's issue, lacks grounding, relevance, and completeness, making it largely ineffective.

### case_04: "@AppleSupport my son is being exposed to inappropriate ads in ad supported kid's..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4)
- **Human Rater Assessment**: The reply treats the problem as downloading or installing an app, while the customer is asking about inappropriate in-game advertising and how to prevent it.
- **LLM Judge Reasoning**: The reply is largely ineffective and unsupported, resulting in a low overall evaluation.

### case_05: "I wish @AppleSupport would let me download the previous iOS so that my phone can..."
- **Key Differences**: groundedness (Human=5, LLM=2, Δ=+3); relevance (Human=4, LLM=2, Δ=+2)
- **Human Rater Assessment**: The reply recognizes an iOS-update context but asks about downloading or installing the update instead of addressing the request to downgrade to a previous iOS version.
- **LLM Judge Reasoning**: The response is polite but fails to address the customer’s request and lacks grounding in the evidence, resulting in a moderate overall score.

### case_06: "@AppleSupport How do I allow no password for already purchased apps but require ..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4); tone (Human=5, LLM=3, Δ=+2); unsupported_claims (Human=5, LLM=1, Δ=+4)
- **Human Rater Assessment**: The response assumes the customer cannot download apps and asks about an error, but the actual request is about App Store password settings.
- **LLM Judge Reasoning**: The response fails to address the customer’s intent, lacks grounding, relevance, and completeness, and contains unsupported claims, resulting in a low overall score.

### case_09: "@AppleSupport unable to download music/install apps on iPad for months. Asks for..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4); relevance (Human=4, LLM=2, Δ=+2); escalation_appropriateness (Human=3, LLM=5, Δ=-2)
- **Human Rater Assessment**: The reply asks about signing in, passwords, or iCloud, which is somewhat related to verification, but it does not directly address the App Store download problem or failed verification code.
- **LLM Judge Reasoning**: The response is polite but lacks groundedness, relevance, and completeness, resulting in an overall moderate score.

### case_11: "Is anyone else’s iPhone making a high pitched alarm noise? It sounds like a smok..."
- **Key Differences**: groundedness (Human=5, LLM=2, Δ=+3); completeness (Human=4, LLM=2, Δ=+2)
- **Human Rater Assessment**: The reply correctly identifies the relevant audio components and asks which part is affected; because the customer already specifies the speaker, the question is slightly broader than necessary.
- **LLM Judge Reasoning**: The reply is partially correct and relevant but lacks grounding and completeness, resulting in an overall moderate score.

### case_12: "@AppleSupport I have not touched my phone to summon Siri nor anybody spoke hey S..."
- **Key Differences**: correctness (Human=2, LLM=4, Δ=-2); relevance (Human=2, LLM=5, Δ=-3); completeness (Human=1, LLM=3, Δ=-2); escalation_appropriateness (Human=2, LLM=5, Δ=-3)
- **Human Rater Assessment**: The reply discusses how to use 'Hey Siri' rather than investigating why Siri activated unexpectedly, and its reference to an article does not support the specific symptom.
- **LLM Judge Reasoning**: The response is generally correct and relevant, but lacks completeness and contains one unsupported claim, resulting in an overall score of 4.0.

### case_13: "@116333 @AppleSupport @144811 Every time I️ connect my phone via USB this song s..."
- **Key Differences**: tone (Human=4, LLM=2, Δ=+2)
- **Human Rater Assessment**: The system-status message does not address automatic music playback over USB, so it is neither grounded in nor relevant to the customer's problem.
- **LLM Judge Reasoning**: The reply fails to address the issue, lacks grounding, relevance, and completeness, and contains unsupported claims, resulting in a low overall score.

### case_14: "@AppleSupport It pops up every time I look at my phone after I come out of the s..."
- **Key Differences**: groundedness (Human=2, LLM=4, Δ=-2); unsupported_claims (Human=2, LLM=5, Δ=-3); escalation_appropriateness (Human=2, LLM=5, Δ=-3)
- **Human Rater Assessment**: The reply only points to an unspecified article and does not identify or troubleshoot the recurring pop-up described by the customer.
- **LLM Judge Reasoning**: The reply is partially grounded but lacks relevance, completeness, and specific guidance, resulting in an overall moderate score.

### case_15: "So @115858 when are yall gunna fix this update, because if my phone freezes for ..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4); unsupported_claims (Human=5, LLM=1, Δ=+4); escalation_appropriateness (Human=2, LLM=5, Δ=-3)
- **Human Rater Assessment**: The reply asks about a hardware issue even though the customer describes freezing after an update, so the response is poorly matched to the actual software/performance complaint.
- **LLM Judge Reasoning**: The response is largely unhelpful and unsupported by evidence, resulting in a low overall score.

### case_16: "@AppleSupport secured #iPhoneX at launch but ONLY w/a select carrier. Yet was to..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4)
- **Human Rater Assessment**: The response treats the message as a cellular service problem involving calls, data, or signal, but the customer is asking about purchasing a SIM-free phone.
- **LLM Judge Reasoning**: The response is largely irrelevant and unsupported, with only the tone being acceptable; overall performance is poor.

### case_17: "@115858 when are you going to fix this bug because am getting really annoyed. Ab..."
- **Key Differences**: groundedness (Human=2, LLM=4, Δ=-2); unsupported_claims (Human=2, LLM=5, Δ=-3); escalation_appropriateness (Human=2, LLM=5, Δ=-3)
- **Human Rater Assessment**: The reply assumes the bug concerns notifications even though the customer gives no such detail, so both the intent and troubleshooting direction are unsupported.
- **LLM Judge Reasoning**: The reply is grounded in evidence but fails to address the customer’s specific request for a fix timeline, resulting in low correctness, relevance, and completeness. The tone is appropriate and no unsupported claims are present.

### case_18: "@AppleSupport I think I found the answer: I can't pre-order iPhone X SIM-free at..."
- **Key Differences**: correctness (Human=5, LLM=2, Δ=+3); groundedness (Human=5, LLM=1, Δ=+4); relevance (Human=5, LLM=2, Δ=+3); completeness (Human=5, LLM=1, Δ=+4); escalation_appropriateness (Human=5, LLM=2, Δ=+3)
- **Human Rater Assessment**: The reply is an appropriate polite acknowledgment of the customer's thanks, but the predicted cellular intent does not match the customer's purpose in this message.
- **LLM Judge Reasoning**: The response is polite but lacks relevance, completeness, and grounding, resulting in a low overall effectiveness.

### case_19: "@167325 @AppleSupport It’s the worst....wish I could downgrade 😂..."
- **Key Differences**: correctness (Human=3, LLM=1, Δ=+2); groundedness (Human=5, LLM=1, Δ=+4); relevance (Human=4, LLM=1, Δ=+3); unsupported_claims (Human=5, LLM=1, Δ=+4)
- **Human Rater Assessment**: The reply recognizes the iOS-update context but asks about downloading or installing an update rather than the customer's stated desire to downgrade.
- **LLM Judge Reasoning**: The reply fails to address the customer's intent, lacks grounding, relevance, and completeness, though it maintains a polite tone. The escalation decision is also inappropriate.

### case_20: "@AppleSupport hi so I got an email my account was locked. But I’m pretty sure it..."
- **Key Differences**: correctness (Human=5, LLM=2, Δ=+3); groundedness (Human=5, LLM=1, Δ=+4); relevance (Human=5, LLM=1, Δ=+4); completeness (Human=5, LLM=1, Δ=+4)
- **Human Rater Assessment**: The reply appropriately treats a suspected account-security/phishing issue cautiously and directs the customer to official support without claiming the email is genuine or fake.
- **LLM Judge Reasoning**: The response is partially correct but lacks evidence-based guidance, relevance, and completeness, resulting in a moderate overall score.

### case_21: "Is anybody looking into the mail and messages apps syncing issues with @133941 a..."
- **Key Differences**: correctness (Human=4, LLM=2, Δ=+2); relevance (Human=5, LLM=3, Δ=+2); completeness (Human=4, LLM=2, Δ=+2)
- **Human Rater Assessment**: The response asks which app is affected and what happens, which is a sensible first troubleshooting step for the reported syncing problem.
- **LLM Judge Reasoning**: The reply is polite and grounded but lacks direct relevance and completeness, resulting in an overall moderate score.

### case_26: "@AppleSupport why do you keep taking my money!!!! Apple hold keeps taking $1 eve..."
- **Key Differences**: groundedness (Human=1, LLM=5, Δ=-4); unsupported_claims (Human=1, LLM=5, Δ=-4)
- **Human Rater Assessment**: The reply gives a generic software workaround without addressing billing, recurring $1 authorization charges, or any evidence that a software update will fix the issue.
- **LLM Judge Reasoning**: The response is largely unhelpful, lacking actionable guidance, and does not resolve the customer’s issue, resulting in a low overall score.

### case_35: "Everytime i refresh my tl my music stops and its really making me mad. @115858 f..."
- **Key Differences**: relevance (Human=3, LLM=1, Δ=+2)
- **Human Rater Assessment**: The response stays on the music topic but asks how often music is deleted and whether the customer subscribes, neither of which addresses playback stopping during timeline refresh.
- **LLM Judge Reasoning**: The response is fully grounded in evidence but fails to address the customer’s issue, providing minimal relevance and completeness, resulting in a low overall score.

### case_45: "FIX SPOTIFY IN THE MUSIC PLAYER #mygod @115858..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4); unsupported_claims (Human=5, LLM=1, Δ=+4); escalation_appropriateness (Human=3, LLM=1, Δ=+2)
- **Human Rater Assessment**: The response asks about Apple Music or iTunes even though the customer explicitly identifies Spotify, so it does not correctly target the reported service.
- **LLM Judge Reasoning**: The response fails to address the customer’s Spotify issue, lacks grounding, is irrelevant, incomplete, and contains unsupported claims, making it unsuitable for auto-handling.

## 5. Methodological Comparison: Current System vs. Historical Benchmark

- **Current System Run (N=6 valid)**: Evaluates current replies under the frozen response generator.
- **Historical Run (N=50 benchmark)**: In the historical benchmark (`evaluation/results/historical_llm_judge_summary.csv`), only 29/50 historical replies match the current system's outputs. Therefore, historical results cannot serve as a direct proxy for current system agreement.

## 6. Interpretation & Limitations

1. **Provider Edge Availability**: Cloudflare HTTP 403 on the Groq endpoint prevented completing all 50 judge cases. When the provider becomes available, running `python -m src.evaluation.human_agreement` will update all metrics seamlessly.
2. **Ordinal Scale Nature**: Human support ratings are ordinal (1–5 scale). Spearman rank correlation ($ho$) is preferred over Pearson $r$ because it evaluates monotonic alignment rather than linear spacing.
3. **Judge Strictness Bias**: The LLM judge tends to penalize standard clarification questions heavily on Groundedness (scoring 1.0 when no external document was cited), whereas human evaluators rated safe diagnostic questions 5/5.
4. **No Claim of Production Readiness**: High or moderate agreement validates the evaluation harness; it does not prove the model is ready for unassisted customer deployment without human escalation safeguards.
