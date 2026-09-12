import pandas as pd

INPUT_FILE = "data/processed/apple_customer_messages.csv"

df = pd.read_csv(INPUT_FILE)

# Keywords representing potential problem areas
keywords = {
    "battery": ["battery", "charging", "charge"],
    "software_update": ["update", "ios", "software"],
    "wifi_connectivity": ["wifi", "wi-fi", "internet", "network"],
    "apps": ["app", "apps", "application"],
    "app_store": ["app store", "store", "download"],
    "apple_music": ["music", "spotify", "song", "playlist"],
    "icloud_account": ["icloud", "apple id", "account", "password"],
    "screen_display": ["screen", "display", "brightness"],
    "calls": ["call", "calling", "phone call"],
    "audio": ["sound", "speaker", "volume", "audio"],
    "device_problem": ["iphone", "ipad", "ipod", "device"],
}

for intent, words in keywords.items():

    print("\n" + "=" * 80)
    print(f"INTENT CANDIDATE: {intent}")
    print("=" * 80)

    # Find messages containing at least one keyword
    pattern = "|".join(words)

    matches = df[
        df["text"]
        .astype(str)
        .str.lower()
        .str.contains(pattern, regex=True, na=False)
    ]

    print(f"Matching messages: {len(matches):,}\n")

    # Show 10 examples
    examples = matches.sample(
        min(10, len(matches)),
        random_state=42
    )

    for i, (_, row) in enumerate(examples.iterrows(), 1):
        print(f"{i}. {row['text']}")