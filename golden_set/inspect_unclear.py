import pandas as pd

INPUT_FILE = "golden_set/final_golden.csv"

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

unclear = df[df["human_intent"] == "other_unclear"]

print("=" * 80)
print("OTHER / UNCLEAR REVIEW")
print("=" * 80)

print(f"Total golden examples: {len(df)}")
print(f"Other/unclear examples: {len(unclear)}")

for i, (_, row) in enumerate(unclear.iterrows(), start=1):

    print("\n" + "-" * 80)
    print(f"UNCLEAR EXAMPLE {i} / {len(unclear)}")
    print("-" * 80)

    print(f"Candidate group : {row['candidate_group']}")
    print(f"Tweet ID       : {row['tweet_id']}")
    print("\nCustomer message:")
    print(row["text"])

    if i >= 50:
        print("\nShowing first 50 unclear examples only.")
        break