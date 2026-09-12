import pandas as pd

INPUT_FILE = "golden_set/golden_candidates.csv"

df = pd.read_csv(INPUT_FILE)

print("\nGolden Candidate Dataset")
print("------------------------")
print(f"Total candidates: {len(df)}")

print("\nCandidate group distribution:")
print(df["candidate_group"].value_counts())

print("\nSample examples from each group:")
print("----------------------------------")

for group in df["candidate_group"].unique():

    print(f"\n### {group}")

    samples = df[
        df["candidate_group"] == group
    ].head(5)

    for _, row in samples.iterrows():

        print(f"\nTweet ID: {row['tweet_id']}")
        print(f"Text: {row['text']}")