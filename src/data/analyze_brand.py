import pandas as pd

FILE_PATH = "data/raw/twcs/twcs.csv"
BRAND = "AppleSupport"
CHUNK_SIZE = 50_000

customer_messages = []
support_messages = []

print("Searching for AppleSupport conversations...")

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE):

    # Customer messages mentioning AppleSupport
    inbound = chunk[
        (chunk["inbound"] == True) &
        (chunk["text"].fillna("").str.contains(
            "@AppleSupport",
            case=False,
            regex=False
        ))
    ]

    # AppleSupport replies
    outbound = chunk[
        (chunk["inbound"] == False) &
        (chunk["author_id"].astype(str).str.lower() == "applesupport")
    ]

    customer_messages.extend(
        inbound["text"].dropna().tolist()
    )

    support_messages.extend(
        outbound["text"].dropna().tolist()
    )

print("\nAnalysis complete!")

print(f"Customer messages found: {len(customer_messages):,}")
print(f"AppleSupport replies found: {len(support_messages):,}")

print("\nSample customer messages:")
print("=" * 80)

for i, message in enumerate(customer_messages[:50], 1):
    print(f"\n{i}. {message}")

print("\n\nSample AppleSupport replies:")
print("=" * 80)

for i, message in enumerate(support_messages[:30], 1):
    print(f"\n{i}. {message}")