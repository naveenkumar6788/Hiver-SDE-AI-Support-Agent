import os
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


GOLDEN_PATH = "golden_set/golden_independently_verified.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep letters and numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


class IntentClassifier:

    def __init__(self, data_path=GOLDEN_PATH, verbose=True):

        if isinstance(data_path, pd.DataFrame):
            df = data_path.copy()
        else:
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Golden dataset not found: {data_path}"
                )

            if verbose:
                print("Loading golden dataset...")

            df = pd.read_csv(data_path)

        required_columns = ["text", "human_intent"]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(
                    f"Missing required column: {column}"
                )

        df["text"] = df["text"].fillna("").astype(str)
        df["human_intent"] = df["human_intent"].fillna("").astype(str)

        df["clean_text"] = df["text"].apply(clean_text)

        # Remove empty examples
        df = df[
            (df["clean_text"].str.len() > 0) &
            (df["human_intent"].str.len() > 0)
        ].copy()

        self.df = df

        X = df["clean_text"]
        y = df["human_intent"]

        if verbose:
            print(f"Training examples: {len(df)}")
            print(f"Number of intents: {y.nunique()}")

        # TF-IDF model
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            sublinear_tf=True
        )

        X_tfidf = self.vectorizer.fit_transform(X)

        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )

        self.model.fit(X_tfidf, y)

        if verbose:
            print("Intent classifier ready.")

    # ---------------------------------------------------------
    # RULE-BASED INTENT DETECTION
    # ---------------------------------------------------------

    def rule_based_intent(self, text):

        text = clean_text(text)

        if not text:
            return None

        # -----------------------------------------------------
        # 1. HARDWARE: Physical SIM tray / drawer (before cellular catches "sim")
        # -----------------------------------------------------
        if re.search(r"\bsim (card )?(drawer|tray|slot)\b", text) or re.search(r"\btray won t open\b", text):
            return "device_hardware"

        # -----------------------------------------------------
        # 2. APPLE ID / ICLOUD / ACCOUNT / AUTHENTICATION
        # -----------------------------------------------------
        account_patterns = [
            r"\bicloud\b",
            r"\bapple id\b",
            r"\bappleid\b",
            r"\bicloud account\b",
            r"\bapple account\b",
            r"\bitunes account\b",
            r"\bforgot password\b",
            r"\baccount password\b",
            r"\bsign in to icloud\b",
            r"\blog into icloud\b",
            r"\bverification code\b",
            r"\b(sign up|create|register).*(itunes|apple|icloud) account\b",
            r"\baccount in \w+\b",
        ]

        for pattern in account_patterns:
            if re.search(pattern, text):
                return "apple_id_icloud"

        # -----------------------------------------------------
        # 3. APP STORE DOWNLOADS
        # -----------------------------------------------------

        app_store_patterns = [
            r"\bdownload (an )?app\b",
            r"\bdownload apps\b",
            r"\binstall (an )?app\b",
            r"\binstall apps\b",
            r"\bapp (is )?not downloading\b",
            r"\bapp (is )?not installing\b",
            r"\bcannot download (an )?app\b",
            r"\bcan t download (an )?app\b",
            r"\bcan t install (an )?app\b",
            r"\bapp store\b",
            r"\bapps? won t download\b",
            r"\bapps? won t install\b",
        ]

        for pattern in app_store_patterns:
            if re.search(pattern, text):
                return "app_store_downloads"

        # -----------------------------------------------------
        # 4. AUDIO / SPEAKER
        # -----------------------------------------------------

        audio_patterns = [
            r"\bspeaker\b",
            r"\bspeakers\b",
            r"\b(no sound|sound not working|sound stopped|sound issue|sound problem|sound crackl|distorted sound|sound cut)\b",
            r"\bvolume\b",
            r"\baudio\b",
            r"\bheadphone\b",
            r"\bheadphones\b",
            r"\bearphones?\b",
            r"\bbluetooth audio\b",
        ]

        for pattern in audio_patterns:
            if re.search(pattern, text):
                return "audio_speaker"

        # -----------------------------------------------------
        # 5. BATTERY / CHARGING
        # -----------------------------------------------------

        battery_patterns = [
            r"\bbattery\b",
            r"\bcharge\b",
            r"\bcharging\b",
            r"\bcharger\b",
            r"\bpower drain\b",
            r"\bdraining\b",
            r"\bdrain\b",
            r"\bbattery health\b",
            r"\bwon t charge\b",
            r"\bnot charging\b",
        ]

        for pattern in battery_patterns:
            if re.search(pattern, text):
                return "battery_charging"

        # -----------------------------------------------------
        # 6. WIFI
        # -----------------------------------------------------

        wifi_patterns = [
            r"\bwifi\b",
            r"\bwi fi\b",
            r"\bwireless\b",
            r"\bwireless network\b",
            r"\bconnect to wifi\b",
            r"\bconnected to wifi\b",
        ]

        for pattern in wifi_patterns:
            if re.search(pattern, text):
                return "wifi_connectivity"

        # -----------------------------------------------------
        # 7. CELLULAR / CALLS
        # -----------------------------------------------------

        calls_patterns = [
            r"\bphone calls?\b",
            r"\bmake calls?\b",
            r"\bmaking calls?\b",
            r"\bcall someone\b",
            r"\bcalls not working\b",
            r"\bcannot make calls?\b",
            r"\bcan t make calls?\b",
            r"\bwon t make calls?\b",
            r"\bno signal\b",
            r"\bno service\b",
            r"\bsearching(\.\.\.)?\b",
            r"\bcellular\b",
            r"\bcellular data\b",
            r"\bmobile data\b",
            r"\bmobile network\b",
            r"\bnetwork signal\b",
            r"\bsim card\b",
            r"\bsim\b",
            r"\binvalid sim\b",
            r"\bsignal\b",
        ]

        for pattern in calls_patterns:
            if re.search(pattern, text):
                return "calls_cellular"

        # -----------------------------------------------------
        # 8. SCREEN / DISPLAY
        # -----------------------------------------------------

        screen_patterns = [
            r"\bscreen\b",
            r"\bdisplay\b",
            r"\bblack screen\b",
            r"\bblank screen\b",
            r"\bscreen is black\b",
            r"\btouch screen\b",
            r"\btouchscreen\b",
            r"\bbrightness\b",
        ]

        for pattern in screen_patterns:
            if re.search(pattern, text):
                return "screen_display"

        # -----------------------------------------------------
        # 9. APP PROBLEMS
        # -----------------------------------------------------

        app_patterns = [
            r"\bapp crashes\b",
            r"\bapp crash\b",
            r"\bapp keeps crashing\b",
            r"\bapp stopped\b",
            r"\bapp not working\b",
            r"\bapplication not working\b",
            r"\bapp problem\b",
            r"\bapp issue\b",
            r"\bapp keeps closing\b",
        ]

        for pattern in app_patterns:
            if re.search(pattern, text):
                return "app_problems"

        # Specific application crash/freeze combination override
        app_names = [
            r"\byoutube\b", r"\bspotify\b", r"\binstagram\b", r"\bfacebook\b",
            r"\bwhatsapp\b", r"\btwitter\b", r"\bnetflix\b", r"\bapp\b", r"\bapplication\b"
        ]
        app_problem_indicators = [
            r"\bcrash(es|ing|ed)?\b", r"\bfreeze(s|ing)?\b", r"\bfrozen\b",
            r"\bwon t open\b", r"\bnot opening\b", r"\bdoesn t work\b",
            r"\bnot working\b", r"\bkeeps closing\b", r"\bstopped working\b",
            r"\bquits\b", r"\bforce close\b", r"\blagging\b"
        ]
        has_app_name = any(re.search(a, text) for a in app_names)
        has_app_problem = any(re.search(p, text) for p in app_problem_indicators)
        if has_app_name and has_app_problem:
            return "app_problems"

        # -----------------------------------------------------
        # 10. APPLE MUSIC / ITUNES
        # -----------------------------------------------------

        music_patterns = [
            r"\bapple music\b",
            r"\bitunes\b",
            r"\bmusic app\b",
            r"\bsongs?\b",
            r"\bmusic\b",
            r"\bplaylist\b",
            r"\bsong download\b",
        ]

        for pattern in music_patterns:
            if re.search(pattern, text):
                return "apple_music_itunes"

        # -----------------------------------------------------
        # 11. DEVICE HARDWARE
        # -----------------------------------------------------

        hardware_patterns = [
            r"\bcamera\b",
            r"\btouch id\b",
            r"\bhome button\b",
            r"\bpower button\b",
            r"\biphone is frozen\b",
            r"\bphone is frozen\b",
            r"\bphone froze\b",
            r"\bstuck on the apple logo\b",
            r"\bstuck on apple logo\b",
            r"\bapple logo\b",
            r"\bstuck at apple logo\b",
            r"\bwon t turn on\b",
            r"\bphone won t turn on\b",
            r"\bhardware\b",
        ]

        for pattern in hardware_patterns:
            if re.search(pattern, text):
                return "device_hardware"

        # -----------------------------------------------------
        # 12. IOS SOFTWARE UPDATE
        # -----------------------------------------------------

        ios_patterns = [
            r"\bios\b",
            r"\bios 11\b",
            r"\bios update\b",
            r"\bsoftware update\b",
            r"\bupdate my iphone\b",
            r"\bupdating iphone\b",
            r"\bupdate failed\b",
            r"\bupdate error\b",
            r"\bafter the update\b",
            r"\bsince the update\b",
            r"\bafter updating\b",
        ]

        for pattern in ios_patterns:
            if re.search(pattern, text):
                return "ios_software_update"

        return None

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    def predict(self, text, use_rules=True):

        cleaned = clean_text(text)

        if not cleaned:
            return {
                "intent": "other_unclear",
                "confidence": 0.0,
                "method": "empty"
            }

        # First use deterministic rules for very clear cases if enabled
        if use_rules:
            rule_intent = self.rule_based_intent(cleaned)

            if rule_intent is not None:
                return {
                    "intent": rule_intent,
                    "confidence": 1.0,
                    "method": "rule"
                }

        # Otherwise use ML model
        X = self.vectorizer.transform([cleaned])

        probabilities = self.model.predict_proba(X)[0]

        best_index = probabilities.argmax()

        intent = self.model.classes_[best_index]
        confidence = probabilities[best_index]

        # If ML confidence is very low, mark unclear (only in hybrid mode)
        if use_rules and confidence < 0.12:
            return {
                "intent": "other_unclear",
                "confidence": float(confidence),
                "method": "ml_low_confidence"
            }

        return {
            "intent": intent,
            "confidence": float(confidence),
            "method": "ml"
        }


if __name__ == "__main__":

    classifier = IntentClassifier()

    test_queries = [
        "My iPhone battery is draining very quickly",
        "I cannot connect to WiFi",
        "My iPhone won't make calls",
        "I cannot download an app",
        "My iCloud account is not working",
        "My screen is completely black",
        "My iPhone speaker is not working",
        "My iPhone camera is not working",
        "My iPhone is stuck on the Apple logo",
        "Apple Music is not playing my songs",
        "The app keeps crashing",
        "My iPhone software update failed"
    ]

    print("\n" + "=" * 80)
    print("INTENT CLASSIFIER TEST")
    print("=" * 80)

    for query in test_queries:

        result = classifier.predict(query)

        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Method: {result['method']}")