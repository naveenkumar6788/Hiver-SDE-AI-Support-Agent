# Human vs. LLM Judge Agreement Evaluation

## 1. Evaluation Setup & Case Matching

- **Human Evaluation Set**: 50 complete, independent human evaluations recorded in `golden_set/human_evaluation_50.csv`.
- **Target LLM Judge Set**: 50 cases from current system run (`evaluation/results/llm_judge_results.csv`).
- **Valid Current LLM Judge Judgments**: **6 / 50 cases** (due to Groq Cloudflare HTTP 403 provider edge block on 37 cases, rate-limit on 1 case).
- **Evaluated Paired Sample**: Agreement statistics are strictly and honestly computed on the **6 valid paired cases** (`case_01` through `case_06`).
- **Scientific Integrity Rule**: No missing judge scores were fabricated, simulated, or interpolated.

## 2. Dimension-Level Agreement Statistics

| Dimension | Paired Cases | Human Mean | LLM Mean | Mean Absolute Diff (MAD) | Spearman $\rho$ | Exact Agreement | Within ±1 Point |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Correctness | 6 | 2.8333 | 1.8333 | 1.3333 | 0.7303 | 0.0% (0/6) | 66.7% (4/6) |
| Groundedness | 6 | 5.0 | 3.0 | 2.0 | N/A (insufficient variance or N < 3) | 50.0% (3/6) | 50.0% (3/6) |
| Relevance | 6 | 3.5 | 2.1667 | 1.3333 | 0.8198 | 16.7% (1/6) | 66.7% (4/6) |
| Completeness | 6 | 2.0 | 1.8333 | 0.1667 | 0.8333 | 83.3% (5/6) | 100.0% (6/6) |
| Tone | 6 | 5.0 | 4.6667 | 0.3333 | N/A (insufficient variance or N < 3) | 66.7% (4/6) | 100.0% (6/6) |
| Unsupported Claims | 6 | 5.0 | 4.3333 | 0.6667 | N/A (insufficient variance or N < 3) | 83.3% (5/6) | 83.3% (5/6) |
| Escalation Appropriateness | 6 | 2.8333 | 3.5 | 1.3333 | 0.9000 | 16.7% (1/6) | 50.0% (3/6) |

## 3. Categorical Intent & Escalation Alignment

- **Human Intent Correctness**: **50.0% (3/6)** of evaluated cases were validated as having correct system intent by the independent human evaluator.
- **Escalation Appropriateness**: **16.7% (1/6)** of cases were rated as appropriately handled (rating $\ge 4/5$).

## 4. Disagreement Analysis

Identified 5 cases with an absolute dimension difference $\ge 2.0$ points:

### case_01: "Got my new Three SIM card yay... and now the sim drawer won’t open on my phone....."
- **Key Differences**: escalation_appropriateness (Human=2, LLM=4, Δ=-2)
- **Human Rater Assessment**: The reply recognizes a hardware issue and asks for details, but it does not address the specific stuck SIM tray or answer whether the customer should visit a store.
- **LLM Judge Reasoning**: The reply is polite and grounded but fails to provide a solution or clear next steps, resulting in a moderate overall evaluation.

### case_03: "@AppleSupport Hi, I have an issue where I have set my Calendar app on Mac to sta..."
- **Key Differences**: correctness (Human=3, LLM=1, Δ=+2); groundedness (Human=5, LLM=1, Δ=+4); relevance (Human=4, LLM=1, Δ=+3); escalation_appropriateness (Human=3, LLM=5, Δ=-2)
- **Human Rater Assessment**: The reply asks for more details about the app behavior, which is relevant, but it does not address the specific Calendar setting that keeps reverting.
- **LLM Judge Reasoning**: The response lacks any solution or relevant guidance, resulting in low correctness, groundedness, relevance, and completeness scores, while tone and escalation appropriateness remain high.

### case_04: "@AppleSupport my son is being exposed to inappropriate ads in ad supported kid's..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4)
- **Human Rater Assessment**: The reply treats the problem as downloading or installing an app, while the customer is asking about inappropriate in-game advertising and how to prevent it.
- **LLM Judge Reasoning**: The response fails to address the customer’s problem, lacks grounding, and is incomplete, though it is polite. Escalation decision is inappropriate.

### case_05: "I wish @AppleSupport would let me download the previous iOS so that my phone can..."
- **Key Differences**: correctness (Human=3, LLM=1, Δ=+2); relevance (Human=4, LLM=2, Δ=+2); escalation_appropriateness (Human=3, LLM=5, Δ=-2)
- **Human Rater Assessment**: The reply recognizes an iOS-update context but asks about downloading or installing the update instead of addressing the request to downgrade to a previous iOS version.
- **LLM Judge Reasoning**: The reply is polite and grounded but fails to address the customer's request, resulting in a moderate overall score.

### case_06: "@AppleSupport How do I allow no password for already purchased apps but require ..."
- **Key Differences**: groundedness (Human=5, LLM=1, Δ=+4); unsupported_claims (Human=5, LLM=1, Δ=+4)
- **Human Rater Assessment**: The response assumes the customer cannot download apps and asks about an error, but the actual request is about App Store password settings.
- **LLM Judge Reasoning**: The response fails to address the customer’s request, lacks evidence, and is not helpful.

## 5. Methodological Comparison: Current System vs. Historical Benchmark

- **Current System Run (N=6 valid)**: Evaluates current replies under the frozen response generator.
- **Historical Run (N=50 benchmark)**: In the historical benchmark (`evaluation/results/historical_llm_judge_summary.csv`), only 29/50 historical replies match the current system's outputs. Therefore, historical results cannot serve as a direct proxy for current system agreement.

## 6. Interpretation & Limitations

1. **Provider Edge Availability**: Cloudflare HTTP 403 on the Groq endpoint prevented completing all 50 judge cases. When the provider becomes available, running `python -m src.evaluation.human_agreement` will update all metrics seamlessly.
2. **Ordinal Scale Nature**: Human support ratings are ordinal (1–5 scale). Spearman rank correlation ($ho$) is preferred over Pearson $r$ because it evaluates monotonic alignment rather than linear spacing.
3. **Judge Strictness Bias**: The LLM judge tends to penalize standard clarification questions heavily on Groundedness (scoring 1.0 when no external document was cited), whereas human evaluators rated safe diagnostic questions 5/5.
4. **No Claim of Production Readiness**: High or moderate agreement validates the evaluation harness; it does not prove the model is ready for unassisted customer deployment without human escalation safeguards.
