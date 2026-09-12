# Architecture & Engineering Decision Log

**Project**: Hiver SDE AI Support Agent (`@AppleSupport`)  
**Scope**: 13 Core Architectural & Methodological Decisions

This document records the key technical decisions made during the design, implementation, evaluation, and quality-control phases of the project, including the explicit technical rationale and engineering trade-offs.

---

### Decision 1: Thread-Level Dataset Splitting
* **Decision**: Split conversation datasets strictly at the root dialogue-thread (`conversation_id`) level rather than by random tweet sampling.
* **Reason**: Random tweet splitting leaks conversational context between training/retrieval corpora and the evaluation set when multiple turns of the same customer interaction are separated across splits.
* **Trade-off**: Reduces the total number of independent training splits and requires conversational graph reconstruction.

---

### Decision 2: Deterministic Baseline First
* **Decision**: Implement and thoroughly evaluate heuristic and deterministic baselines before adding complex retrieval scoring or LLM-based components.
* **Reason**: Establishes a verifiable, reproducible benchmark that proves subsequent model improvements are statistically meaningful and non-regressive.
* **Trade-off**: Requires substantial upfront engineering investment in evaluation infrastructure before generating dynamic responses.

---

### Decision 3: Domain-Tailored 12-Class Intent Taxonomy
* **Decision**: Define a specific 12-class intent taxonomy (`app_problems`, `app_store_downloads`, `apple_id_icloud`, `apple_music_itunes`, `audio_speaker`, `battery_charging`, `calls_cellular`, `device_hardware`, `ios_update`, `other_unclear`, `screen_display`, `wifi_connectivity`) matching actual Apple Support operational workflows.
* **Reason**: Generic coarse-grained taxonomies fail to distinguish critical customer differences (e.g. cellular SIM connectivity vs. local Wi-Fi, or software freeze vs. hardware screen crack).
* **Trade-off**: Finer granularity increases classification boundary ambiguity between related technical symptoms.

---

### Decision 4: Dedicated `other_unclear` Class
* **Decision**: Include an explicit `other_unclear` intent category for terse, ambiguous, greeting, or unclassifiable user statements.
* **Reason**: Prevents forcing ambiguous messages (e.g. "what is this?", "help please") into specific technical categories, which would trigger hallucinated or irrelevant troubleshooting.
* **Trade-off**: Requires conservative confidence thresholds to avoid over-predicting the catch-all class for terse technical queries.

---

### Decision 5: Action-Divergence Safety Gating
* **Decision**: Implement explicit action-divergence heuristic checks that reject retrieved troubleshooting evidence prescribing actions incompatible with the customer problem state (e.g. software reboot for SIM tray damage; Wi-Fi resets for carrier SIM locks).
* **Reason**: Protects customer safety and device integrity; prevents recommending irrelevant or counterproductive actions.
* **Trade-off**: Lowers the raw evidence utilization rate from 86.88% to 84.38%, forcing fallback to diagnostic clarification for rejected candidates.

---

### Decision 6: Specific Problem Match Layer
* **Decision**: Score semantic compatibility between customer query tokens and candidate evidence problem contexts to suppress generic historical boilerplate.
* **Reason**: Historical customer support responses frequently contain generic phrases ("Please DM us", "Restart your phone") that do not resolve specific technical issues.
* **Trade-off**: When historical evidence lacks specific troubleshooting steps, the system must trigger safe diagnostic clarification rather than providing instant answers.

---

### Decision 7: Resolved-Message Policy
* **Decision**: Detect customer messages confirming issue resolution ("never mind, it works now", "fixed it thanks") and acknowledge them courteously without prescribing troubleshooting steps.
* **Reason**: Avoids annoying customers with redundant troubleshooting guidance after an issue is already resolved.
* **Trade-off**: Adds regex and intent guard checks to the reply generation pipeline.

---

### Decision 8: Deterministic Escalation Guard
* **Decision**: Proactively route account security threats (hacked Apple ID), financial disputes (unauthorized charges), and physical hardware hazards (battery swelling, acoustic alarm sounds) to human specialists (`ESCALATE`).
* **Reason**: High-risk safety, financial, and security concerns must never be handled autonomously without human oversight.
* **Trade-off**: Diverts cases away from autonomous handling into higher-cost human customer support queues.

---

### Decision 9: Deterministic Grounding Verifier
* **Decision**: Enforce an automated deterministic grounding verifier requiring all factual claims in generated replies to be strictly anchored in accepted evidence or certified clarification templates.
* **Reason**: Guarantees zero unverified technical claims pass through the generator unchecked.
* **Trade-off**: Strict verification stringency rejects potentially helpful but marginally supported conversational phrasing.

---

### Decision 10: 50-Case Sample for Diagnostic LLM Judge
* **Decision**: Use a representative 50-case stratified sample for deep multidimensional LLM judge diagnostic evaluations.
* **Reason**: Balances API quota limits, execution time, and expense while providing granular diagnostic scores across 6 evaluation dimensions.
* **Trade-off**: Smaller sample size than the full 160-case deterministic evaluation suite.

---

### Decision 11: Withholding Human Agreement Pending Genuine Annotations
* **Decision**: Keep human evaluation status as `HUMAN AGREEMENT NOT YET VERIFIED` because the 50 current annotation rows have `annotation_source = model_assisted`.
* **Reason**: Strict scientific integrity: model-assisted ratings must never be claimed or reported as independent human ground truth.
* **Trade-off**: Statistical inter-rater agreement metrics (Spearman's $\rho$, Cohen's $\kappa$) remain withheld until authentic human evaluations are recorded.

---

### Decision 12: Separation of Historical vs. Current Judge Artifacts
* **Decision**: Maintain historical 50/50 complete LLM judge results in [`evaluation/results/historical_llm_judge_summary.csv`](evaluation/results/historical_llm_judge_summary.csv) completely separate from current evaluation outputs.
* **Reason**: Prevents blending stale evaluation results with current system outputs or creating misleading aggregate averages.
* **Trade-off**: Requires maintaining and documenting distinct historical artifact files.

---

### Decision 13: Halting Evaluation on Cloudflare Edge Block
* **Decision**: Halt automated LLM judge execution immediately upon receiving Cloudflare HTTP 403 (error code 1010) blocks rather than retrying indefinitely or fabricating missing scores.
* **Reason**: Prevents polluting evaluation datasets with repeated network failure artifacts and avoids risking permanent IP/account bans.
* **Trade-off**: Results in an incomplete 12/50 run that must be transparently reported as partial.
