import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import AppleSupportRetriever


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_FILE = PROJECT_ROOT / "golden_set" / "golden_independently_verified.csv"

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = RESULTS_DIR / "automated_retrieval_results.csv"
SUMMARY_FILE = RESULTS_DIR / "automated_retrieval_summary.csv"

TOP_K = 5
NUM_QUERIES = 30


# ============================================================
# HELPERS
# ============================================================

def normalize_id(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def get_value(row, columns, default=""):
    for column in columns:
        if column in row.index:
            value = row[column]

            if pd.notna(value):
                return str(value)

    return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# AUTOMATED RELEVANCE HEURISTIC
# ============================================================

def calculate_relevance(
    query_text,
    historical_customer,
    historical_response,
    query_intent,
    historical_intent,
    similarity
):
    """
    Automatic relevance score.

    2 = strongly relevant
    1 = somewhat relevant
    0 = irrelevant

    This is NOT human judgment.

    Signals:
      - intent agreement
      - lexical overlap
      - TF-IDF similarity
    """

    query = str(query_text).lower()
    customer = str(historical_customer).lower()
    response = str(historical_response).lower()

    # --------------------------------------------------------
    # Token overlap
    # --------------------------------------------------------

    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "is",
        "it", "i", "my", "me", "on", "in", "for", "with",
        "this", "that", "have", "has", "be", "can", "do",
        "you", "please", "how", "what", "why", "from",
        "your", "we", "are", "was", "but", "not"
    }

    query_tokens = {
        token.strip(".,!?;:()[]{}'\"")
        for token in query.split()
        if len(token) >= 3
        and token not in stop_words
    }

    customer_tokens = {
        token.strip(".,!?;:()[]{}'\"")
        for token in customer.split()
        if len(token) >= 3
        and token not in stop_words
    }

    response_tokens = {
        token.strip(".,!?;:()[]{}'\"")
        for token in response.split()
        if len(token) >= 3
        and token not in stop_words
    }

    if query_tokens:
        customer_overlap = len(
            query_tokens & customer_tokens
        ) / len(query_tokens)

        response_overlap = len(
            query_tokens & response_tokens
        ) / len(query_tokens)
    else:
        customer_overlap = 0.0
        response_overlap = 0.0

    # --------------------------------------------------------
    # Intent agreement
    # --------------------------------------------------------

    intent_match = (
        query_intent
        and historical_intent
        and query_intent == historical_intent
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = 0.0

    # Similarity is the strongest continuous signal.
    if similarity >= 0.45:
        score += 3.0
    elif similarity >= 0.35:
        score += 2.0
    elif similarity >= 0.25:
        score += 1.0
    elif similarity >= 0.15:
        score += 0.5

    # Intent match.
    if intent_match:
        score += 2.5

    # Customer message overlap.
    if customer_overlap >= 0.50:
        score += 2.0
    elif customer_overlap >= 0.30:
        score += 1.0
    elif customer_overlap >= 0.15:
        score += 0.5

    # Historical response overlap.
    if response_overlap >= 0.30:
        score += 1.0
    elif response_overlap >= 0.15:
        score += 0.5

    # --------------------------------------------------------
    # Convert to 0/1/2
    # --------------------------------------------------------

    if score >= 5.0:
        return 2

    if score >= 2.5:
        return 1

    return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUTOMATED RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Top K: {TOP_K}")
    print(f"Queries: {NUM_QUERIES}")

    print("\nThis evaluation is fully automatic.")
    print("It does NOT create human relevance judgments.")

    # --------------------------------------------------------
    # Load golden dataset
    # --------------------------------------------------------

    if not GOLDEN_FILE.exists():

        print("\nERROR:")
        print("Golden dataset not found:")
        print(GOLDEN_FILE)

        return

    print("\nLoading golden dataset...")

    golden = pd.read_csv(
        GOLDEN_FILE,
        dtype=str
    ).fillna("")

    if "tweet_id" not in golden.columns:

        print("ERROR: tweet_id column not found.")

        print(
            "Available columns:",
            list(golden.columns)
        )

        return

    golden["tweet_id"] = (
        golden["tweet_id"]
        .apply(normalize_id)
    )

    # One query per conversation to avoid
    # evaluating multiple messages from same thread.

    if "conversation_id" in golden.columns:

        golden["conversation_id"] = (
            golden["conversation_id"]
            .apply(normalize_id)
        )

        golden = golden.drop_duplicates(
            subset=["conversation_id"],
            keep="first"
        )

    else:

        golden = golden.drop_duplicates(
            subset=["tweet_id"],
            keep="first"
        )

    golden = golden.head(NUM_QUERIES).copy()

    print(
        f"Evaluation queries selected: "
        f"{len(golden)}"
    )

    # --------------------------------------------------------
    # Load retriever
    # --------------------------------------------------------

    print("\nLoading AppleSupport retriever...")

    retriever = AppleSupportRetriever()

    print("Retriever ready.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_results = []

    query_metrics = []

    for query_number, (_, query_row) in enumerate(
        golden.iterrows(),
        start=1
    ):

        query_tweet_id = normalize_id(
            query_row["tweet_id"]
        )

        conversation_id = get_value(
            query_row,
            ["conversation_id"]
        )

        query_text = get_value(
            query_row,
            ["text", "customer_text", "message"]
        )

        true_intent = get_value(
            query_row,
            ["intent", "human_intent"]
        )

        # ----------------------------------------------------
        # Predict intent
        # ----------------------------------------------------

        predicted_intent = ""

        confidence = 0.0

        try:

            prediction = retriever.predict_intent(
                query_text
            )

            if isinstance(prediction, tuple):

                if len(prediction) >= 1:
                    predicted_intent = str(
                        prediction[0]
                    )

                if len(prediction) >= 2:
                    confidence = safe_float(
                        prediction[1]
                    )

            elif isinstance(prediction, dict):

                predicted_intent = str(
                    prediction.get(
                        "intent",
                        ""
                    )
                )

                confidence = safe_float(
                    prediction.get(
                        "confidence",
                        0
                    )
                )

            else:

                predicted_intent = str(
                    prediction
                )

        except Exception as e:

            print(
                f"\nIntent prediction warning: {e}"
            )

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        try:

            results = retriever.retrieve(
                query_text,
                top_k=TOP_K,
                intent=None,
                exclude_conversation_id=conversation_id
            )

        except Exception as e:

            print(
                f"\nRetrieval error for "
                f"{query_tweet_id}: {e}"
            )

            continue

        print("\n" + "=" * 70)

        print(
            f"QUERY {query_number}/{len(golden)}"
        )

        print("=" * 70)

        print(
            f"Tweet ID: {query_tweet_id}"
        )

        print(
            f"Intent: {true_intent}"
        )

        print(
            f"Predicted: {predicted_intent}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print(
            f"Query: {query_text[:180]}"
        )

        # ----------------------------------------------------
        # Evaluate top K
        # ----------------------------------------------------

        relevance_scores = []

        similarities = []

        for rank, (_, result) in enumerate(
            results.iterrows(),
            start=1
        ):

            similarity = safe_float(
                get_value(
                    result,
                    ["similarity", "score"]
                )
            )

            historical_conversation = get_value(
                result,
                [
                    "conversation_id",
                    "historical_conversation_id"
                ]
            )

            historical_customer = get_value(
                result,
                [
                    "customer_text",
                    "text",
                    "query",
                    "customer_message"
                ]
            )

            historical_response = get_value(
                result,
                [
                    "response_text",
                    "historical_response",
                    "apple_support_response",
                    "response"
                ]
            )

            historical_intent = get_value(
                result,
                [
                    "retrieval_intent",
                    "intent",
                    "historical_intent"
                ]
            )

            # ----------------------------------------------
            # If historical intent is missing, infer it.
            # ----------------------------------------------

            if not historical_intent:

                try:

                    hist_prediction = (
                        retriever.predict_intent(
                            historical_customer
                        )
                    )

                    if isinstance(
                        hist_prediction,
                        tuple
                    ):

                        historical_intent = str(
                            hist_prediction[0]
                        )

                    elif isinstance(
                        hist_prediction,
                        dict
                    ):

                        historical_intent = str(
                            hist_prediction.get(
                                "intent",
                                ""
                            )
                        )

                    else:

                        historical_intent = str(
                            hist_prediction
                        )

                except Exception:

                    historical_intent = ""

            # ----------------------------------------------
            # Automatic relevance
            # ----------------------------------------------

            relevance = calculate_relevance(
                query_text=query_text,
                historical_customer=historical_customer,
                historical_response=historical_response,
                query_intent=predicted_intent,
                historical_intent=historical_intent,
                similarity=similarity
            )

            relevance_scores.append(
                relevance
            )

            similarities.append(
                similarity
            )

            # ----------------------------------------------
            # Display
            # ----------------------------------------------

            print(
                f"\nRank {rank}: "
                f"similarity={similarity:.4f} "
                f"relevance={relevance}"
            )

            print(
                f"Historical intent: "
                f"{historical_intent}"
            )

            print(
                f"Customer: "
                f"{historical_customer[:150]}"
            )

            # ----------------------------------------------
            # Save detailed result
            # ----------------------------------------------

            all_results.append({

                "query_number":
                    query_number,

                "query_tweet_id":
                    query_tweet_id,

                "query_conversation_id":
                    conversation_id,

                "query_text":
                    query_text,

                "true_intent":
                    true_intent,

                "predicted_intent":
                    predicted_intent,

                "prediction_confidence":
                    confidence,

                "rank":
                    rank,

                "similarity":
                    similarity,

                "historical_conversation_id":
                    historical_conversation,

                "historical_intent":
                    historical_intent,

                "historical_customer":
                    historical_customer,

                "historical_response":
                    historical_response,

                "automatic_relevance":
                    relevance
            })

        # ----------------------------------------------------
        # Query-level metrics
        # ----------------------------------------------------

        if relevance_scores:

            relevance_array = np.array(
                relevance_scores
            )

            # Binary relevance:
            # 1 if somewhat or strongly relevant.
            binary = (
                relevance_array > 0
            ).astype(int)

            # Precision@K
            precision_at_5 = (
                binary.sum() / len(binary)
            )

            # Precision@1
            precision_at_1 = (
                binary[0]
                if len(binary) >= 1
                else 0
            )

            # Precision@3
            precision_at_3 = (
                binary[:3].sum() / 3
                if len(binary) >= 3
                else binary.mean()
            )

            # MRR
            reciprocal_rank = 0.0

            for rank, value in enumerate(
                binary,
                start=1
            ):

                if value == 1:

                    reciprocal_rank = (
                        1.0 / rank
                    )

                    break

            # NDCG@5
            try:

                ndcg = ndcg_score(
                    [relevance_array],
                    [relevance_array],
                    k=TOP_K
                )

            except Exception:

                ndcg = 0.0

            query_metrics.append({

                "query_number":
                    query_number,

                "query_tweet_id":
                    query_tweet_id,

                "conversation_id":
                    conversation_id,

                "true_intent":
                    true_intent,

                "predicted_intent":
                    predicted_intent,

                "precision_at_1":
                    precision_at_1,

                "precision_at_3":
                    precision_at_3,

                "precision_at_5":
                    precision_at_5,

                "mrr":
                    reciprocal_rank,

                "ndcg_at_5":
                    ndcg,

                "average_similarity":
                    np.mean(similarities),

                "strongly_relevant_count":
                    int(
                        (relevance_array == 2).sum()
                    ),

                "somewhat_relevant_count":
                    int(
                        (relevance_array == 1).sum()
                    ),

                "irrelevant_count":
                    int(
                        (relevance_array == 0).sum()
                    )
            })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    details_df = pd.DataFrame(
        all_results
    )

    metrics_df = pd.DataFrame(
        query_metrics
    )

    details_df.to_csv(
        DETAIL_FILE,
        index=False
    )

    metrics_df.to_csv(
        RESULTS_DIR /
        "automated_retrieval_query_metrics.csv",
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    if metrics_df.empty:

        print("\nNo evaluation results generated.")

        return

    summary = {

        "queries_evaluated":
            len(metrics_df),

        "top_k":
            TOP_K,

        "precision_at_1":
            metrics_df[
                "precision_at_1"
            ].mean(),

        "precision_at_3":
            metrics_df[
                "precision_at_3"
            ].mean(),

        "precision_at_5":
            metrics_df[
                "precision_at_5"
            ].mean(),

        "mrr":
            metrics_df[
                "mrr"
            ].mean(),

        "ndcg_at_5":
            metrics_df[
                "ndcg_at_5"
            ].mean(),

        "average_similarity":
            metrics_df[
                "average_similarity"
            ].mean(),

        "total_strongly_relevant":
            metrics_df[
                "strongly_relevant_count"
            ].sum(),

        "total_somewhat_relevant":
            metrics_df[
                "somewhat_relevant_count"
            ].sum(),

        "total_irrelevant":
            metrics_df[
                "irrelevant_count"
            ].sum()
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("AUTOMATED RETRIEVAL EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nQueries evaluated: "
        f"{summary['queries_evaluated']}"
    )

    print(
        f"Top K: "
        f"{TOP_K}"
    )

    print(
        f"\nPrecision@1: "
        f"{summary['precision_at_1']:.4f}"
    )

    print(
        f"Precision@3: "
        f"{summary['precision_at_3']:.4f}"
    )

    print(
        f"Precision@5: "
        f"{summary['precision_at_5']:.4f}"
    )

    print(
        f"MRR: "
        f"{summary['mrr']:.4f}"
    )

    print(
        f"NDCG@5: "
        f"{summary['ndcg_at_5']:.4f}"
    )

    print(
        f"Average similarity: "
        f"{summary['average_similarity']:.4f}"
    )

    print(
        f"\nStrongly relevant: "
        f"{summary['total_strongly_relevant']}"
    )

    print(
        f"Somewhat relevant: "
        f"{summary['total_somewhat_relevant']}"
    )

    print(
        f"Irrelevant: "
        f"{summary['total_irrelevant']}"
    )

    print("\nFiles created:")

    print(
        DETAIL_FILE
    )

    print(
        RESULTS_DIR /
        "automated_retrieval_query_metrics.csv"
    )

    print(
        SUMMARY_FILE
    )

    print("\nDone.")


if __name__ == "__main__":
    main()