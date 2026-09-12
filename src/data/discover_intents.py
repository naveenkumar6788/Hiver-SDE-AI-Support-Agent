import pandas as pd
from collections import Counter
import re

INPUT_FILE = "data/processed/applesupport_threads.csv"

print("Loading AppleSupport conversations...")

df = pd.read_csv(INPUT_FILE)

print(f"Total tweets: {len(df):,}")

# Keep only customer messages
customer_df = df[df["inbound"] == True].copy()

print(f"Customer messages: {len(customer_df):,}")

# Remove empty messages
customer_df = customer_df[
    customer_df["text"].notna() &
    (customer_df["text"].str.strip() != "")
]

# Common words that are not useful for intent discovery
stop_words = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "my", "me", "i", "it",
    "this", "that", "have", "has", "had", "be", "been", "can", "can't",
    "you", "your", "do", "does", "did", "how", "why", "what", "when",
    "where", "from", "at", "not", "just", "please", "help", "apple",
    "iphone", "ipad"
}

word_counter = Counter()

for text in customer_df["text"]:
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep words
    words = re.findall(r"\b[a-z]{3,}\b", text)

    for word in words:
        if word not in stop_words:
            word_counter[word] += 1

print("\nTop customer words:\n")

for word, count in word_counter.most_common(100):
    print(f"{word:25} {count}")

# Save cleaned customer messages
output_file = "data/processed/apple_customer_messages.csv"

customer_df[
    [
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "conversation_id"
    ]
].to_csv(output_file, index=False)

print("\nSaved customer messages to:")
print(output_file)