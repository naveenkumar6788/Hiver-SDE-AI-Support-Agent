import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.reply_generator import is_evidence_safe, detect_domains, detect_response_domains

cases = [
    {
        "id": "CASE 1",
        "name": "Battery query + Wi-Fi response",
        "query": "My iPhone battery is draining very quickly",
        "intent": "battery_charging",
        "customer": "My iPhone battery drains fast",
        "response": "Go to Settings > Wi-Fi and turn off Wi-Fi.",
        "expect_safe": False,
    },
    {
        "id": "CASE 2",
        "name": "Battery query + battery troubleshooting response",
        "query": "My iPhone battery is draining very quickly",
        "intent": "battery_charging",
        "customer": "battery draining fast",
        "response": "Check Battery Usage in Settings and identify apps using significant power.",
        "expect_safe": True,
    },
    {
        "id": "CASE 3",
        "name": "Wi-Fi query + Wi-Fi troubleshooting response",
        "query": "I cannot connect to WiFi",
        "intent": "wifi_connectivity",
        "customer": "wifi not connecting",
        "response": "Go to Settings > Wi-Fi, tap your network, tap Forget this Network, then reconnect.",
        "expect_safe": True,
    },
    {
        "id": "CASE 4",
        "name": "Wi-Fi query + generic DM us",
        "query": "I cannot connect to WiFi",
        "intent": "wifi_connectivity",
        "customer": "wifi problem",
        "response": "Please send us a DM so we can look into this for you.",
        "expect_safe": False,
    },
    {
        "id": "CASE 5",
        "name": "Apple ID query + camera troubleshooting response",
        "query": "My Apple ID password is not working",
        "intent": "apple_id_icloud",
        "customer": "forgot apple id password",
        "response": "Restart your iPhone and check the camera lens for dirt.",
        "expect_safe": False,
    },
    {
        "id": "CASE 6",
        "name": "Screen query + autocorrect/keyboard response",
        "query": "My screen is completely black",
        "intent": "screen_display",
        "customer": "screen black display",
        "response": "We released iOS 11.1.1 which contains a fix for autocorrect and keyboard typing.",
        "expect_safe": False,
    },
    {
        "id": "CASE 7",
        "name": "SIM query + SIM troubleshooting response",
        "query": "My SIM card says Invalid SIM",
        "intent": "calls_cellular",
        "customer": "no sim card installed error",
        "response": "Remove the SIM card with a paperclip, inspect it for damage, and reinsert it firmly.",
        "expect_safe": True,
    },
    {
        "id": "CASE 8",
        "name": "YouTube query + Apple Music response",
        "query": "The YouTube app keeps crashing",
        "intent": "app_problems",
        "customer": "app crashing",
        "response": "Delete and reinstall the Apple Music app from the App Store.",
        "expect_safe": False,
    },
    {
        "id": "CASE 9",
        "name": "iOS update query + iOS update troubleshooting response",
        "query": "My iPhone is unable to check for iOS update",
        "intent": "ios_software_update",
        "customer": "cannot update ios",
        "response": "Go to Settings > General > iPhone Storage, delete the update file, and download the iOS software update again.",
        "expect_safe": True,
    },
    {
        "id": "CASE 10",
        "name": "Generic customer message + generic AppleSupport response",
        "query": "Hey Apple need help",
        "intent": "other_unclear",
        "customer": "need help",
        "response": "Let us know what is going on and we will be happy to help!",
        "expect_safe": False,
    },
]

all_passed = True
print("=== TESTING 10 REGRESSION CASES ===")
for c in cases:
    safe, reason = is_evidence_safe(
        query=c["query"],
        evidence_candidate=None,
        query_intent=c["intent"],
        evidence_customer=c["customer"],
        evidence_response=c["response"],
        debug=False,
    )
    passed = (safe == c["expect_safe"])
    if not passed:
        all_passed = False
    print(f"[{c['id']}] {c['name']}")
    print(f"    Expected: {'ACCEPT' if c['expect_safe'] else 'REJECT'} | Actual: {'ACCEPT' if safe else 'REJECT'} | Reason: {reason}")
    print(f"    Status: {'PASSED' if passed else 'FAILED'}")

print(f"\nALL 10 REGRESSION CASES PASSED: {all_passed}")
