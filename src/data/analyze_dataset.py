import pandas as pd
from collections import Counter

FILE_PATH = "data/raw/twcs/twcs.csv"
CHUNK_SIZE = 50_000

brand_customer_messages = Counter()
brand_replies = Counter()

total_rows = 0

print("Starting dataset analysis...")

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE):

    total_rows += len(chunk)

    # Customer messages
    inbound = chunk[chunk["inbound"] == True]

    # Company/support replies
    outbound = chunk[chunk["inbound"] == False]

    # For customer messages, the account they mention
    for text in inbound["text"].dropna():
        if "@" in text:
            first_word = text.split()[0]
            if first_word.startswith("@"):
                brand_customer_messages[first_word.lower()] += 1

    # Company accounts
    for author in outbound["author_id"].dropna():
        brand_replies[str(author).lower()] += 1

    print(f"Processed {total_rows:,} rows")

print("\nAnalysis complete!")
print(f"Total rows: {total_rows:,}")

print("\nTop customer-mentioned accounts:")
for brand, count in brand_customer_messages.most_common(30):
    print(f"{brand:30} {count:,}")

print("\nTop support accounts:")
for brand, count in brand_replies.most_common(30):
    print(f"{brand:30} {count:,}")