import pandas as pd
from pathlib import Path

INPUT_FILE = Path("golden_set/golden_candidates.csv")
OUTPUT_FILE = Path("golden_set/golden_labeled.csv")

INTENTS = {
    "1": "battery_charging",
    "2": "ios_software_update",
    "3": "wifi_connectivity",
    "4": "app_problems",
    "5": "app_store_downloads",
    "6": "apple_music_itunes",
    "7": "apple_id_icloud",
    "8": "screen_display",
    "9": "calls_cellular",
    "10": "audio_speaker",
    "11": "device_hardware",
    "12": "other_unclear",
    "s": "skip",
    "q": "quit"
}


def load_data():
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        print("Resuming existing labeling file.")
    else:
        df = pd.read_csv(INPUT_FILE)
        df["golden_intent"] = ""
        df["label_notes"] = ""
        df["label_status"] = "unlabeled"

    return df


def show_intents():
    print("\n========== INTENTS ==========")

    for key, value in INTENTS.items():
        if key not in ["s", "q"]:
            print(f"{key:>2} -> {value}")

    print(" s -> skip")
    print(" q -> quit")


def main():

    df = load_data()

    for index, row in df.iterrows():

        if row["label_status"] == "labeled":
            continue

        print("\n" + "=" * 80)
        print(f"Candidate {index + 1} / {len(df)}")

        print("\nCandidate group:")
        print(row["candidate_group"])

        print("\nTweet:")
        print(row["text"])

        show_intents()

        while True:

            choice = input("\nYour label: ").strip().lower()

            if choice == "q":
                df.to_csv(OUTPUT_FILE, index=False)
                print("\nProgress saved.")
                print(f"File: {OUTPUT_FILE}")
                return

            if choice == "s":
                df.at[index, "label_status"] = "skipped"
                break

            if choice in INTENTS and choice not in ["s", "q"]:

                intent = INTENTS[choice]

                df.at[index, "golden_intent"] = intent

                note = input("Short note (optional): ").strip()

                df.at[index, "label_notes"] = note
                df.at[index, "label_status"] = "labeled"

                break

            print("Invalid choice. Please select 1-12, s, or q.")

        # Save after every label
        df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("LABELING COMPLETE")

    labeled = df[df["label_status"] == "labeled"]

    print(f"Labeled: {len(labeled)}")
    print(f"Skipped: {len(df) - len(labeled)}")

    print("\nIntent distribution:")
    print(labeled["golden_intent"].value_counts())

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()