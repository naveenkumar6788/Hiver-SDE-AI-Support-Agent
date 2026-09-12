import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

GOLDEN_FILE = Path("golden_set/final_golden.csv")
THREAD_FILE = Path("data/processed/applesupport_threads.csv")
OUTPUT_FILE = Path("golden_set/final_golden_human_reviewed.csv")

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
    12: "other_unclear"
}

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading golden set...")
golden = pd.read_csv(GOLDEN_FILE)

print("Loading conversation threads...")
threads = pd.read_csv(THREAD_FILE)

print("Golden rows:", len(golden))
print("Thread rows:", len(threads))


# ---------------------------------------------------------
# PREPARE THREAD DATA
# ---------------------------------------------------------

threads["tweet_id"] = threads["tweet_id"].astype(str)

if "conversation_id" in threads.columns:
    threads["conversation_id"] = threads["conversation_id"].astype(str)

golden["tweet_id"] = golden["tweet_id"].astype(str)

# Keep only useful columns if they exist
available_columns = [
    "tweet_id",
    "conversation_id",
    "author_id",
    "created_at",
    "text",
    "inbound"
]

available_columns = [
    c for c in available_columns
    if c in threads.columns
]

threads_lookup = threads[available_columns].copy()

# ---------------------------------------------------------
# CREATE FAST LOOKUPS
# ---------------------------------------------------------

tweet_lookup = {}

for _, row in threads_lookup.iterrows():
    tweet_id = str(row["tweet_id"])

    tweet_lookup[tweet_id] = row.to_dict()


# ---------------------------------------------------------
# FIND CONVERSATION
# ---------------------------------------------------------

def get_conversation(tweet_id):
    """
    Find the conversation containing the target tweet.
    """

    if tweet_id not in tweet_lookup:
        return []

    row = tweet_lookup[tweet_id]

    conversation_id = row.get("conversation_id")

    if pd.isna(conversation_id):
        return []

    conversation_id = str(conversation_id)

    conversation = threads[
        threads["conversation_id"].astype(str) == conversation_id
    ].copy()

    if conversation.empty:
        return []

    if "created_at" in conversation.columns:
        conversation = conversation.sort_values("created_at")

    return conversation.to_dict("records")


# ---------------------------------------------------------
# DISPLAY CONVERSATION
# ---------------------------------------------------------

def display_conversation(conversation, target_id):

    print("\n" + "=" * 80)
    print("CONVERSATION CONTEXT")
    print("=" * 80)

    if not conversation:
        print("No conversation context found.")
        return

    for msg in conversation:

        tweet_id = str(msg.get("tweet_id", ""))

        text = str(msg.get("text", ""))

        inbound = msg.get("inbound", None)

        if tweet_id == target_id:
            marker = " <<< TARGET"

        else:
            marker = ""

        if inbound is True:
            speaker = "CUSTOMER"

        elif inbound is False:
            speaker = "APPLE SUPPORT"

        else:
            speaker = "UNKNOWN"

        print(f"\n[{speaker}]{marker}")
        print(text)


# ---------------------------------------------------------
# LOAD PREVIOUS HUMAN LABELS
# ---------------------------------------------------------

if OUTPUT_FILE.exists():

    print("\nPrevious review file found.")

    reviewed = pd.read_csv(OUTPUT_FILE)

    reviewed["tweet_id"] = reviewed["tweet_id"].astype(str)

else:

    reviewed = golden.copy()


# ---------------------------------------------------------
# FORCE REVIEW COLUMNS TO STRING
# ---------------------------------------------------------

for column in [
    "human_intent",
    "human_notes",
    "label_status"
]:

    if column not in reviewed.columns:
        reviewed[column] = ""

    reviewed[column] = (
        reviewed[column]
        .fillna("")
        .astype(str)
    )


# ---------------------------------------------------------
# REVIEW LOOP
# ---------------------------------------------------------

total = len(reviewed)

print("\n")
print("=" * 80)
print("HUMAN GOLDEN SET REVIEW")
print("=" * 80)

print("\nIntent numbers:")

for number, intent in INTENTS.items():
    print(f"{number:2d} = {intent}")

print("\nCommands:")
print("  a = accept suggested label")
print("  u = other_unclear")
print("  s = skip")
print("  q = quit and save")


for index in range(total):

    row = reviewed.iloc[index]

    tweet_id = str(row["tweet_id"])

    # Skip already reviewed rows
    if row.get("label_status") == "reviewed":
        continue

    text = str(row["text"])

    suggested = row.get("human_intent", "")

    if pd.isna(suggested) or suggested == "":
        suggested = row.get("golden_intent", "")

    print("\n")
    print("#" * 80)
    print(f"EXAMPLE {index + 1} / {total}")
    print("#" * 80)

    print("\nCUSTOMER MESSAGE:")
    print(text)

    print("\nBULK SUGGESTION:")
    print(suggested)

    # -----------------------------------------------------
    # SHOW CONTEXT
    # -----------------------------------------------------

    conversation = get_conversation(tweet_id)

    display_conversation(
        conversation,
        tweet_id
    )

    # -----------------------------------------------------
    # GET USER DECISION
    # -----------------------------------------------------

    while True:

        choice = input(
            "\nEnter intent number, "
            "'a'=accept, "
            "'u'=unclear, "
            "'s'=skip, "
            "'q'=quit: "
        ).strip().lower()

        # ACCEPT BULK SUGGESTION
        if choice == "a":

            final_intent = suggested

            if pd.isna(final_intent) or final_intent == "":
                print("No suggestion available.")
                continue

            note = input(
                "Reason/note (optional): "
            ).strip()

            reviewed.at[index, "human_intent"] = final_intent
            reviewed.at[index, "human_notes"] = note
            reviewed.at[index, "label_status"] = "reviewed"

            print(
                f"Saved: {final_intent}"
            )

            break

        # OTHER UNCLEAR
        elif choice == "u":

            note = input(
                "Why is this unclear? "
            ).strip()

            reviewed.at[index, "human_intent"] = (
                "other_unclear"
            )

            reviewed.at[index, "human_notes"] = note

            reviewed.at[index, "label_status"] = "reviewed"

            print("Saved: other_unclear")

            break

        # SKIP
        elif choice == "s":

            print("Skipped.")

            break

        # QUIT
        elif choice == "q":

            reviewed.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print("\nReview saved.")

            print(
                f"Output: {OUTPUT_FILE}"
            )

            raise SystemExit

        # DIRECT INTENT NUMBER
        elif choice.isdigit():

            number = int(choice)

            if number not in INTENTS:

                print(
                    "Invalid number. "
                    "Choose 1-12."
                )

                continue

            final_intent = INTENTS[number]

            note = input(
                "Reason for this label: "
            ).strip()

            reviewed.at[index, "human_intent"] = (
                final_intent
            )

            reviewed.at[index, "human_notes"] = note

            reviewed.at[index, "label_status"] = "reviewed"

            print(
                f"Saved: {final_intent}"
            )

            break

        else:

            print(
                "Invalid input."
            )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

reviewed.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("=" * 80)
print("HUMAN REVIEW COMPLETE")
print("=" * 80)

print(
    "\nOutput:",
    OUTPUT_FILE
)

print(
    "\nReviewed:",
    (
        reviewed["label_status"] == "reviewed"
    ).sum()
)

print(
    "Remaining:",
    (
        reviewed["label_status"] != "reviewed"
    ).sum()
)

print("\nFinal distribution:")

print(
    reviewed.loc[
        reviewed["label_status"] == "reviewed",
        "human_intent"
    ].value_counts()
)