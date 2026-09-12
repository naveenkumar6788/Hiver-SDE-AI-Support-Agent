import pandas as pd
from pathlib import Path

INPUT_FILE = Path("golden_set/golden_review_200.csv")
OUTPUT_FILE = Path("golden_set/final_golden.csv")

INTENTS = {
    1: "battery_charging",
    2: "ios_software_update",
    3: "wifi_connectivity",
    4: "app_problems",
    5: "app_store_downloads",
    6: "apple_music_itunes",
    7: "apple_id_icloud",
    8: "screen_display",
    9: "calls_cellular",
    10: "audio_speaker",
    11: "device_hardware",
    12: "other_unclear",
}


def show_intents():
    print("\n" + "=" * 70)
    print("INTENT OPTIONS")
    print("=" * 70)

    for number, intent in INTENTS.items():
        print(f"{number:2}. {intent}")

    print("\nCommands:")
    print("  Enter = accept bulk suggestion")
    print("  1-12   = choose/correct intent")
    print("  s      = skip")
    print("  q      = quit and save")
    print("=" * 70)


def load_data():
    if OUTPUT_FILE.exists():
        print("Existing final_golden.csv found.")
        df = pd.read_csv(OUTPUT_FILE, dtype=str).fillna("")
    else:
        print("Loading golden_review_200.csv...")
        df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

        if "human_intent" not in df.columns:
            df["human_intent"] = ""

        if "human_notes" not in df.columns:
            df["human_notes"] = ""

        if "label_status" not in df.columns:
            df["label_status"] = ""

    return df


def save_data(df):
    df.to_csv(OUTPUT_FILE, index=False)


def main():

    df = load_data()

    print(f"\nTotal golden examples: {len(df)}")

    show_intents()

    for index in range(len(df)):

        # Already reviewed
        if str(df.loc[index, "label_status"]).strip() == "human_verified":
            continue

        text = str(df.loc[index, "text"]).strip()
        candidate_group = str(df.loc[index, "candidate_group"]).strip()

        # Bulk suggestion
        bulk_intent = ""

        if "bulk_intent" in df.columns:
            bulk_intent = str(df.loc[index, "bulk_intent"]).strip()

        if not bulk_intent:
            bulk_intent = str(df.loc[index, "golden_intent"]).strip()

        print("\n\n")
        print("=" * 80)
        print(f"EXAMPLE {index + 1} / {len(df)}")
        print("=" * 80)

        print(f"\nOriginal sampling group:")
        print(f"  {candidate_group}")

        print(f"\nBulk suggestion:")
        print(f"  {bulk_intent if bulk_intent else 'No suggestion'}")

        print("\nCustomer message:")
        print("-" * 80)
        print(text)
        print("-" * 80)

        while True:

            choice = input(
                "\nYour label [Enter=accept suggestion, 1-12, s=skip, q=quit]: "
            ).strip().lower()

            # Quit
            if choice == "q":
                save_data(df)
                print("\nProgress saved.")
                print(f"Output: {OUTPUT_FILE}")
                return

            # Skip
            if choice == "s":
                print("Skipped.")
                break

            # Enter = accept bulk suggestion
            if choice == "":

                if bulk_intent in INTENTS.values():
                    selected_intent = bulk_intent

                    print(f"\nAccepted suggestion: {selected_intent}")

                else:
                    print(
                        "\nNo valid bulk suggestion available."
                        "\nPlease select an intent using 1-12."
                    )
                    continue

            # Number 1-12
            elif choice.isdigit() and 1 <= int(choice) <= 12:

                selected_intent = INTENTS[int(choice)]

                print(f"\nSelected: {selected_intent}")

            else:
                print("Invalid input. Enter 1-12, Enter, s, or q.")
                continue

            # Ask for reasoning
            notes = input(
                "Reason for this label (short note): "
            ).strip()

            if not notes:
                notes = "Human verified based on the message and intent guidelines."

            df.loc[index, "human_intent"] = selected_intent
            df.loc[index, "human_notes"] = notes
            df.loc[index, "label_status"] = "human_verified"

            # Store suggestion separately if possible
            if "bulk_intent" in df.columns:
                df.loc[index, "bulk_intent"] = bulk_intent

            save_data(df)

            print("\nSaved.")

            break

    save_data(df)

    verified = (
        df["label_status"].astype(str).str.strip() == "human_verified"
    ).sum()

    print("\n" + "=" * 80)
    print("GOLDEN SET REVIEW COMPLETE")
    print("=" * 80)

    print(f"Total examples:     {len(df)}")
    print(f"Human verified:     {verified}")
    print(f"Remaining:          {len(df) - verified}")
    print(f"Output:             {OUTPUT_FILE}")


if __name__ == "__main__":
    main()