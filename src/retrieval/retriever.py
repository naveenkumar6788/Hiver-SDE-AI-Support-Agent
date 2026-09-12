"""
src/retrieval/retriever.py

Historical AppleSupport case retriever using TF-IDF and hybrid intent alignment.
Retrieves relevant historical customer questions and official AppleSupport resolutions
to ground automated support replies.
"""

import os
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intent.classifier import IntentClassifier

THREADS_PATH = PROJECT_ROOT / "data" / "processed" / "applesupport_threads.csv"


# ============================================================
# Utility Functions
# ============================================================

def normalize_id(value):
    """Normalize tweet and conversation IDs to strings without float artifacts."""
    if pd.isna(value) or str(value).strip() == "" or str(value).strip().lower() == "nan":
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def clean_text(text):
    """
    Clean and canonicalize customer support text for retrieval.
    Preserves and standardizes domain-critical technical terms.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter user mentions (@AppleSupport, @123456, etc.)
    text = re.sub(r"@\w+", " ", text)

    # Decode common HTML / Twitter noise
    text = re.sub(r"&amp;", " and ", text)
    text = re.sub(r"&lt;", " < ", text)
    text = re.sub(r"&gt;", " > ", text)

    # Normalize compound technical terms so they match as unified semantic tokens
    text = re.sub(r"\bapp\s+store\b", "app_store", text)
    text = re.sub(r"\bapple\s+music\b", "apple_music", text)
    text = re.sub(r"\bapple\s+id\b", "apple_id", text)
    text = re.sub(r"\bi\s*cloud\b", "icloud", text)
    text = re.sub(r"\bwi\s*-?\s*fi\b", "wifi", text)
    text = re.sub(r"\btouch\s+id\b", "touch_id", text)
    text = re.sub(r"\bface\s+id\b", "face_id", text)
    text = re.sub(r"\bhome\s+button\b", "home_button", text)
    text = re.sub(r"\bpower\s+button\b", "power_button", text)
    text = re.sub(r"\bblue\s*tooth\b", "bluetooth", text)
    text = re.sub(r"\bsoftware\s+update\b", "software_update", text)
    text = re.sub(r"\bblack\s+screen\b", "black_screen", text)

    # Contraction expansions
    text = re.sub(r"\bwon['’]t\b", "will not", text)
    text = re.sub(r"\bcan['’]t\b", "cannot", text)
    text = re.sub(r"\b(doesn|don|didn|isn|aren|wasn|weren|haven|hasn|hadn)['’]t\b", r"\1 not", text)

    # Keep letters, numbers, underscores (for compound terms), apostrophes, and spaces
    text = re.sub(r"[^a-z0-9_'\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_informative_message(cleaned_text):
    """
    Check if a historical customer message contains sufficient technical substance.
    Filters out 1-3 word queries, acknowledgments, and empty greetings.
    """
    words = cleaned_text.split()
    if len(words) < 4:
        return False

    # Filter messages containing purely generic acknowledgments/fillers
    generic_words = {
        "thanks", "thank", "you", "apple", "applesupport", "help",
        "please", "yes", "no", "hi", "hello", "dm", "ok", "okay",
        "didnt", "work", "fixed", "yay", "thx", "pls"
    }
    non_generic = [w for w in words if w not in generic_words]
    if len(non_generic) < 2:
        return False

    return True


# ============================================================
# Evidence Quality Evaluation
# ============================================================

GENERIC_DM_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(we('?| )?ve|we have) received your dm\b",
        r"\b(we('?| )?ve|we have) responded to your dm\b",
        r"\bcheck (your|the) dm\b",
        r"\blook for us there\b",
        r"\bcontinue with you there\b",
        r"\bcontinue from there\b",
        r"\bwe('?| )?ll continue from there\b",
        r"\bwe will continue from there\b",
        r"\bjoin us in dm\b",
        r"\breach out in dm\b",
        r"\bsend us a dm and we('?| )?ll (start|continue)\b",
        r"\bsend us a dm so we can\b",
        r"\bwe('?| )?re glad to hear that worked\b",
        r"\bglad to hear that worked\b",
        r"\bhave a great (day|rest of your day)\b",
    ]
]

ACTIONABLE_STEP_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(try|follow|complete|check out|take a look at)\b.*?\b(steps|article|resource|guide|link|page)\b",
        r"\b(restart|restarting|force restart)\b",
        r"\b(reset|restore|backup|back up|sign out|sign in)\b",
        r"\b(update|updating|install|reinstall)\b",
        r"\b(toggle|turn off|turn on|switch off|switch on)\b",
        r"\b(settings|wi-?fi|bluetooth|cellular|cache)\b",
        r"\b(remove|unplug|clean|insert)\b.*?\b(case|protector|sim|cable|port)\b",
        r"\bmake sure (you|it|that)\b",
        r"\bactivation instructions\b",
        r"\bfree up space\b",
        r"\bcheck the box\b",
    ]
]

DIAGNOSTIC_QUESTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwhat (ios )?version\b",
        r"\bwhich (ios )?version\b",
        r"\bversion of ios\b",
        r"\bwhich model\b",
        r"\bwhat model\b",
        r"\bwhen did this (start|begin)\b",
        r"\bis this happening with (all|both|any)\b",
        r"\bare you seeing (an? |the )?(error|message|alert|code|screen|no service)\b",
        r"\bhave you (tried|tested|attempted)\b",
        r"\bdoes this (also )?happen (when|on|with)\b",
        r"\bare there any (app )?updates\b",
        r"\bare you downloading (updates|new)\b",
        r"\bcan you test if\b",
        r"\bare you able to\b",
        r"\bwas the iphone ordered\b",
        r"\bis it no longer syncing\b",
        r"\bare you able to remove\b",
        r"\bwhat happens when you\b",
    ]
]


def evaluate_evidence_quality(resp):
    """
    Deterministically evaluates the evidence quality of an historical AppleSupport response.

    Components:
    - Base quality (length & substantive words)
    - Actionable troubleshooting indicators (+0.25)
    - Diagnostic isolation questions (+0.20)
    - Useful support article/link bonus (+0.15)
    - Generic boilerplate / DM redirection penalty (-0.45 if pure boilerplate, -0.15 if accompanied by steps)

    Returns:
        dict with:
            - evidence_quality_score (float, 0.05 to 1.0)
            - evidence_quality_reason (str)
            - actionable (bool)
            - generic_response (bool)
    """
    if not resp or not isinstance(resp, str):
        return {
            "evidence_quality_score": 0.05,
            "evidence_quality_reason": "empty response",
            "actionable": False,
            "generic_response": True
        }

    clean = re.sub(r"@\w+", " ", resp).lower()
    clean = re.sub(r"\s+", " ", clean).strip()
    words = [w for w in clean.split() if not w.startswith("http")]
    word_count = len(words)

    is_generic = any(p.search(clean) for p in GENERIC_DM_PATTERNS)
    has_action = any(p.search(clean) for p in ACTIONABLE_STEP_PATTERNS)
    has_diag = any(p.search(clean) for p in DIAGNOSTIC_QUESTION_PATTERNS)
    has_link = bool(re.search(r"https?://\S+", resp))
    has_link_context = has_link and bool(
        re.search(r"\b(steps|article|resource|guide|visit|check|help|link|page)\b", clean)
    )

    score = 0.50
    reasons = []

    # Substantive length
    if word_count < 6:
        score -= 0.20
    elif word_count >= 12:
        score += 0.10

    # Actionable guidance
    if has_action:
        score += 0.25
        reasons.append("troubleshooting guidance")

    # Diagnostic inquiries
    if has_diag:
        score += 0.20
        reasons.append("diagnostic questions")

    # Knowledge link
    if has_link_context:
        score += 0.15
        reasons.append("support resource link")
    elif has_link:
        score += 0.08
        reasons.append("link provided")

    # Generic penalty
    if is_generic:
        if not has_action and not has_diag:
            score -= 0.45
            reasons.append("pure generic/DM handoff")
        else:
            score -= 0.15
            reasons.append("DM redirection")

    score = max(0.05, min(1.0, score))

    actionable = (score >= 0.60) and (has_action or (has_diag and word_count >= 10))
    generic_response = (score < 0.45) or (is_generic and not has_action and not has_diag)

    if reasons:
        reason_str = " + ".join(reasons)
    else:
        reason_str = "substantive response" if not generic_response else "low substance"

    return {
        "evidence_quality_score": round(score, 3),
        "evidence_quality_reason": reason_str,
        "actionable": actionable,
        "generic_response": generic_response
    }


def is_valid_resolution_response(resp):
    """
    Verify that an AppleSupport response is not entirely devoid of text.
    """
    if not resp or not isinstance(resp, str):
        return False
    clean_resp = re.sub(r"@\w+", " ", resp)
    clean_resp = re.sub(r"https?://\S+", " ", clean_resp).strip()
    return len(clean_resp) >= 8


from src.retrieval.evidence_relevance import (
    calculate_evidence_relevance,
    MIN_EVIDENCE_RELEVANCE,
    MIN_EVIDENCE_QUALITY,
    RELEVANCE_WEIGHTS,
    extract_domain_entities
)


# ============================================================
# AppleSupport Retriever Class
# ============================================================

class AppleSupportRetriever:

    def __init__(self, threads_path=THREADS_PATH):
        threads_path = Path(threads_path)
        print("Loading historical conversations...")

        if not threads_path.exists():
            raise FileNotFoundError(f"Conversation file not found: {threads_path}")

        # ------------------------------------------------------
        # 1. Load reconstructed conversations
        # ------------------------------------------------------
        df = pd.read_csv(threads_path)

        # Normalize IDs
        df["tweet_id"] = df["tweet_id"].apply(normalize_id)
        df["conversation_id"] = df["conversation_id"].apply(normalize_id)

        customer_df = df[df["author_id"] != "AppleSupport"].copy()
        agent_df = df[df["author_id"] == "AppleSupport"].copy()

        # ------------------------------------------------------
        # 2. Map customer messages to AppleSupport responses
        # ------------------------------------------------------
        agent_map = agent_df.groupby("conversation_id")["text"].first().to_dict()
        customer_df["response"] = customer_df["conversation_id"].map(agent_map)

        # Drop customer messages without an AppleSupport response
        customer_df = customer_df.dropna(subset=["response"]).copy()
        customer_df["response"] = customer_df["response"].fillna("").astype(str)

        # ------------------------------------------------------
        # 3. Filter out non-resolution responses & short noise
        # ------------------------------------------------------
        customer_df = customer_df[customer_df["response"].apply(is_valid_resolution_response)].copy()

        # ------------------------------------------------------
        # 4. Clean text & Filter low-information messages
        # ------------------------------------------------------
        customer_df["clean_text"] = customer_df["text"].apply(clean_text)
        customer_df = customer_df[customer_df["clean_text"].apply(is_informative_message)].copy()

        # Deduplicate exact cleaned customer messages
        customer_df = customer_df.drop_duplicates(subset=["clean_text"], keep="first")
        self.df = customer_df.reset_index(drop=True)

        print(f"Customer messages with usable AppleSupport responses: {len(self.df)}")

        # ------------------------------------------------------
        # 5. Evaluate Evidence Quality across Historical Responses
        # ------------------------------------------------------
        print("Evaluating historical response evidence quality...")
        evidence_results = [evaluate_evidence_quality(r) for r in self.df["response"]]
        self.df["evidence_quality_score"] = [e["evidence_quality_score"] for e in evidence_results]
        self.df["evidence_quality_reason"] = [e["evidence_quality_reason"] for e in evidence_results]
        self.df["actionable"] = [e["actionable"] for e in evidence_results]
        self.df["generic_response"] = [e["generic_response"] for e in evidence_results]

        # ------------------------------------------------------
        # 6. Initialize Hybrid Intent Classifier
        # ------------------------------------------------------
        print("Loading intent classifier...")
        self.intent_classifier = IntentClassifier(verbose=False)

        # ------------------------------------------------------
        # 7. Assign Retrieval Intents via Fast Batch Processing
        # ------------------------------------------------------
        print("Assigning retrieval intents...")
        # Step 1: Rule-based intent detection
        rules_intent = self.df["clean_text"].apply(self.intent_classifier.rule_based_intent)

        # Step 2: Batch ML inference for unmatched queries
        unmatched_mask = rules_intent.isna()
        if unmatched_mask.any():
            unmatched_texts = self.df.loc[unmatched_mask, "clean_text"]
            X_vec = self.intent_classifier.vectorizer.transform(unmatched_texts)
            probas = self.intent_classifier.model.predict_proba(X_vec)
            best_indices = probas.argmax(axis=1)
            best_classes = self.intent_classifier.model.classes_[best_indices]
            max_probas = probas.max(axis=1)
            ml_intents = np.where(max_probas < 0.12, "other_unclear", best_classes)
            rules_intent.loc[unmatched_mask] = ml_intents

        self.df["retrieval_intent"] = rules_intent.values

        # ------------------------------------------------------
        # 8. Build TF-IDF Retrieval Index
        # ------------------------------------------------------
        print("Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(self.df["clean_text"])
        print(f"TF-IDF index ready: {self.matrix.shape[0]} documents")

    def predict_intent(self, query):
        """Predict intent for a customer query using the hybrid classifier."""
        return self.intent_classifier.predict(query)

    def retrieve(
        self,
        query,
        top_k=5,
        intent=None,
        exclude_conversation_id=None
    ):
        """
        Retrieve top-k historical AppleSupport examples for a customer query with
        evidence relevance and quality scoring.

        Scoring strategy:
            Prioritizes:
            1. Evidence Relevance (topic, problem, and entity fit to current query)
            2. Intent Alignment (bonus for matching intent, penalty for conflict)
            3. Evidence Quality (actionable steps, official links vs generic DM penalty)
            4. Semantic Similarity (TF-IDF cosine similarity)
        """
        # Predict intent if not explicitly supplied
        intent_confidence = 0.0
        if intent is None:
            intent_result = self.predict_intent(query)
            if isinstance(intent_result, dict):
                intent = intent_result.get("intent", "other_unclear")
                intent_confidence = float(intent_result.get("confidence", 0.0))
            elif isinstance(intent_result, tuple):
                intent = str(intent_result[0])
                intent_confidence = float(intent_result[1]) if len(intent_result) > 1 else 0.0
            else:
                intent = str(intent_result)

        query_clean = clean_text(query)

        empty_cols = [
            "rank", "final_score", "retrieval_score", "semantic_similarity",
            "similarity", "tweet_id", "conversation_id", "text", "customer_text",
            "response", "historical_response", "predicted_intent",
            "retrieval_intent", "historical_intent", "intent_match",
            "evidence_quality_score", "evidence_quality_reason", "actionable",
            "generic_response", "evidence_relevance_score", "problem_match_score",
            "problem_match_reason", "entity_overlap", "entity_mismatch",
            "issue_overlap", "issue_mismatch", "relevance_accepted", "relevance_reason"
        ]
        if not query_clean:
            empty_df = pd.DataFrame(columns=empty_cols)
            empty_df.attrs["grounded"] = False
            return empty_df

        # ------------------------------------------------------
        # 1. Compute Cosine Similarities
        # ------------------------------------------------------
        query_vector = self.vectorizer.transform([query_clean])
        similarities = cosine_similarity(query_vector, self.matrix)[0]

        # ------------------------------------------------------
        # 2. Exclude current conversation (leakage prevention)
        # ------------------------------------------------------
        if exclude_conversation_id:
            exclude_id = normalize_id(exclude_conversation_id)
            conv_mask = (self.df["conversation_id"].values != exclude_id)
        else:
            conv_mask = np.ones(len(self.df), dtype=bool)

        valid_indices = np.where(conv_mask)[0]
        if len(valid_indices) == 0:
            empty_df = pd.DataFrame(columns=empty_cols)
            empty_df.attrs["grounded"] = False
            return empty_df

        valid_sims = similarities[valid_indices]

        # Candidate pre-ranking by similarity + intent match to select top candidate pool
        conf = max(0.0, min(1.0, intent_confidence))
        if intent and intent != "other_unclear":
            match_bonus = 0.08 + (0.06 * conf)
            mismatch_penalty = -(0.04 + (0.04 * conf))
            intent_mask = (self.df["retrieval_intent"].values[valid_indices] == intent)
            intent_adj_pre = np.where(intent_mask, match_bonus, mismatch_penalty)
        else:
            intent_mask = (self.df["retrieval_intent"].values[valid_indices] == intent)
            intent_adj_pre = np.where(intent_mask, 0.04, 0.0)

        pre_score = valid_sims + intent_adj_pre

        # Select top 60 candidates for detailed evidence relevance evaluation
        pool_size = min(60, len(valid_indices))
        if len(valid_indices) > pool_size:
            top_pos = np.argpartition(pre_score, -pool_size)[-pool_size:]
            top_candidate_indices = valid_indices[top_pos]
        else:
            top_candidate_indices = valid_indices

        cand_df = self.df.iloc[top_candidate_indices].copy()
        cand_df["similarity"] = similarities[top_candidate_indices]
        cand_df["semantic_similarity"] = cand_df["similarity"]
        cand_df["intent_match"] = (cand_df["retrieval_intent"] == intent)
        cand_df["predicted_intent"] = intent

        # ------------------------------------------------------
        # 3. Calculate Deterministic Evidence Relevance
        # ------------------------------------------------------
        relevance_records = []
        for _, row in cand_df.iterrows():
            rel_info = calculate_evidence_relevance(
                query=query,
                historical_customer_message=row["clean_text"],
                historical_response=row["response"],
                current_intent=intent,
                historical_intent=row["retrieval_intent"],
                semantic_similarity=row["similarity"],
                evidence_quality_score=row["evidence_quality_score"],
                actionable=row["actionable"],
                generic_response=row["generic_response"]
            )
            relevance_records.append(rel_info)

        cand_df["evidence_relevance_score"] = [r["evidence_relevance_score"] for r in relevance_records]
        cand_df["relevance_reason"] = [r["relevance_reason"] for r in relevance_records]
        cand_df["problem_match_score"] = [r["problem_match_score"] for r in relevance_records]
        cand_df["problem_match_reason"] = [r["problem_match_reason"] for r in relevance_records]
        cand_df["entity_overlap"] = [r["entity_overlap"] for r in relevance_records]
        cand_df["entity_mismatch"] = [r["entity_mismatch"] for r in relevance_records]
        cand_df["issue_overlap"] = [r["issue_overlap"] for r in relevance_records]
        cand_df["issue_mismatch"] = [r["issue_mismatch"] for r in relevance_records]
        cand_df["relevance_accepted"] = [r["relevance_accepted"] for r in relevance_records]

        # Specific Problem Match Diagnostic Signals
        cand_df["customer_problem_similarity"] = [r.get("customer_problem_similarity", 0.0) for r in relevance_records]
        cand_df["response_problem_similarity"] = [r.get("response_problem_similarity", 0.0) for r in relevance_records]
        cand_df["domain_match"] = [r.get("domain_match", 0.0) for r in relevance_records]
        cand_df["action_match"] = [r.get("action_match", 0.0) for r in relevance_records]
        cand_df["conflict_detected"] = [r.get("conflict_detected", False) for r in relevance_records]
        cand_df["final_evidence_score"] = [r.get("final_evidence_score", r["evidence_relevance_score"]) for r in relevance_records]
        cand_df["evidence_rejection_reason"] = [r.get("evidence_rejection_reason", r["relevance_reason"]) for r in relevance_records]

        # ------------------------------------------------------
        # 4. Final Transparent Scoring & Ranking
        # ------------------------------------------------------
        # Weighted combination prioritizing verified relevance:
        # - Evidence Relevance : 45%
        # - Semantic Similarity: 30%
        # - Intent Alignment   : 15%
        # - Evidence Quality   : 10%
        word_counts = cand_df["clean_text"].str.split().str.len()
        informativeness = np.where(word_counts >= 8, 1.02, 0.95)

        cand_df["final_score"] = (
            0.45 * cand_df["evidence_relevance_score"] +
            0.30 * (cand_df["similarity"] * informativeness) +
            0.15 * np.where(cand_df["intent_match"], 0.10, -0.05) +
            0.10 * cand_df["evidence_quality_score"]
        )
        cand_df["retrieval_score"] = cand_df["final_score"]

        # Sort: Accepted grounded evidence first, then highest final score, then evidence relevance, then similarity
        cand_df = cand_df.sort_values(
            by=["relevance_accepted", "final_score", "evidence_relevance_score", "similarity"],
            ascending=[False, False, False, False]
        )

        # Filter noise floor: require minimal evidence relevance or similarity
        filtered_results = cand_df[
            (cand_df["similarity"] >= 0.10) |
            (cand_df["evidence_relevance_score"] >= MIN_EVIDENCE_RELEVANCE)
        ].copy()

        if len(filtered_results) == 0:
            filtered_results = cand_df.copy()

        filtered_results = filtered_results.head(top_k).copy()

        # Check grounding status
        has_grounded = bool(filtered_results["relevance_accepted"].any())
        filtered_results.attrs["grounded"] = has_grounded

        # Add 1-indexed rank
        filtered_results["rank"] = range(1, len(filtered_results) + 1)

        # Aliases for robust caller access
        filtered_results["customer_text"] = filtered_results["text"]
        filtered_results["historical_response"] = filtered_results["response"]
        filtered_results["historical_intent"] = filtered_results["retrieval_intent"]

        out_cols = [
            "rank",
            "final_score",
            "retrieval_score",
            "semantic_similarity",
            "similarity",
            "tweet_id",
            "conversation_id",
            "text",
            "customer_text",
            "response",
            "historical_response",
            "predicted_intent",
            "retrieval_intent",
            "historical_intent",
            "intent_match",
            "evidence_quality_score",
            "evidence_quality_reason",
            "actionable",
            "generic_response",
            "evidence_relevance_score",
            "problem_match_score",
            "problem_match_reason",
            "entity_overlap",
            "entity_mismatch",
            "issue_overlap",
            "issue_mismatch",
            "relevance_accepted",
            "relevance_reason",
            "customer_problem_similarity",
            "response_problem_similarity",
            "domain_match",
            "action_match",
            "conflict_detected",
            "final_evidence_score",
            "evidence_rejection_reason",
        ]

        return filtered_results[out_cols].reset_index(drop=True)


# ============================================================
# Standalone Test (15 Verification Cases)
# ============================================================

if __name__ == "__main__":
    retriever = AppleSupportRetriever()

    test_queries = [
        # Standard Core Intents (1 to 10)
        "My iPhone battery is draining very quickly",
        "I cannot connect to WiFi",
        "My iPhone won't make calls",
        "I cannot download an app from the App Store",
        "My iCloud account is not working",
        "My screen is completely black",
        "My iPhone speaker is not working",
        "My iPhone camera is not working",
        "My iPhone is stuck on the Apple logo",
        "The YouTube app keeps crashing on my iPhone",
        # Mismatch & Edge Case Tests (11 to 15)
        "My iCloud account is locked",
        "YouTube keeps crashing",
        "My battery is draining quickly",
        "My speaker has stopped working",
        "My SIM card is not working"
    ]

    print("\n" + "=" * 95)
    print("RETRIEVER VERIFICATION TEST WITH EVIDENCE RELEVANCE SCORING (15 QUERIES)")
    print("=" * 95)

    for idx, query in enumerate(test_queries, start=1):
        intent_res = retriever.predict_intent(query)
        pred_intent = intent_res.get("intent", "other_unclear") if isinstance(intent_res, dict) else str(intent_res)
        conf = float(intent_res.get("confidence", 0.0)) if isinstance(intent_res, dict) else 0.0

        print("\n" + "-" * 95)
        print(f"[{idx}] QUERY           : {query}")
        print(f"    PREDICTED INTENT: {pred_intent} (Confidence: {conf:.4f})")
        print("-" * 95)

        results = retriever.retrieve(query, top_k=2)

        if len(results) == 0:
            print("  No relevant results retrieved (grounded=False).")
            continue

        for i, row in results.iterrows():
            cust_disp = str(row["customer_text"])[:95].replace("\n", " ").encode("ascii", "replace").decode("ascii")
            resp_disp = str(row["historical_response"])[:105].replace("\n", " ").encode("ascii", "replace").decode("ascii")
            print(f"  [Rank {row['rank']}] FinalScore: {row['final_score']:.4f} | Sim: {row['similarity']:.4f} | Match: {row['intent_match']} | Intent: {row['retrieval_intent']}")
            print(f"    Problem Match     : Score={row['problem_match_score']:.4f} | Reason: {row['problem_match_reason']}")
            print(f"    Mismatches        : AppMismatch={row['entity_mismatch']} | IssueMismatch={row['issue_mismatch']} | IssueOverlap={row['issue_overlap']}")
            print(f"    Evidence Quality  : Score={row['evidence_quality_score']:.2f} | Act={row['actionable']} | Gen={row['generic_response']}")
            print(f"    Evidence Relevance: Score={row['evidence_relevance_score']:.4f} | Accepted={row['relevance_accepted']} | Overlap={row['entity_overlap']}")
            print(f"    Reason            : {row['relevance_reason']}")
            print(f"    Customer          : {cust_disp}...")
            print(f"    Response          : {resp_disp}...")