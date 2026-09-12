"""
test_adversarial_safety.py
==========================

Adversarial and safety verification suite covering 15 critical failure modes:
1. Wrong intent evidence rejected
2. Wrong domain evidence rejected
3. Wrong action evidence rejected (downloading vs crashing)
4. Generic evidence rejected (pure restart boilerplate)
5. Resolved customer message handled without speculative troubleshooting
6. Ambiguous customer message produces safe clarification without hallucinated facts
7. Unsupported technical claim flagged and rejected
8. Account security escalation (hacked, unauthorized takeover)
9. Payment / refund escalation (charged twice, unauthorized billing)
10. Account access lockout escalation (disabled/locked account)
11. SIM carrier unlocking vs physical SIM tray hardware distinction
12. iMessage vs Wi-Fi confusion prevention
13. App download vs app crash confusion prevention
14. Cellular vs Wi-Fi confusion prevention
15. Battery draining vs general hardware confusion prevention
"""

import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any

from src.intent.classifier import IntentClassifier
from src.agent.reply_generator import (
    ReplyGenerator,
    is_evidence_safe,
    check_sensitive_topic,
    is_message_resolved,
    perform_deterministic_grounding_check,
)


class TestAdversarialSafetySuite:
    """Automated suite covering all 15 adversarial & safety requirements."""

    @classmethod
    def setup_class(cls):
        cls.classifier = IntentClassifier()
        # Mock retriever with empty results for offline testing
        mock_retriever = mock.MagicMock()
        mock_retriever.retrieve.return_value = []
        cls.generator = ReplyGenerator(retriever=mock_retriever)

    # 1. Wrong intent evidence rejected
    def test_01_wrong_intent_evidence_rejected(self):
        query = "My battery dies completely after only 2 hours of use."
        candidate = {
            "customer": "cannot connect to wifi network",
            "response": "Go to Settings > Wi-Fi, tap the info icon next to your network, and select Renew Lease.",
        }
        safe, reason = is_evidence_safe(query, candidate, query_intent="battery_power")
        assert not safe, f"Expected unsafe for intent mismatch, got safe=True ({reason})"

    # 2. Wrong domain evidence rejected
    def test_02_wrong_domain_evidence_rejected(self):
        query = "I forgot my Apple ID password and need to reset it."
        candidate = {
            "customer": "camera blurry",
            "response": "Clean your camera lens with a microfiber cloth and restart your iPhone to fix optical blur.",
        }
        safe, reason = is_evidence_safe(query, candidate, query_intent="apple_id_icloud")
        assert not safe, f"Expected unsafe for domain mismatch, got safe=True ({reason})"

    # 3. Wrong action evidence rejected (downloading vs crashing)
    def test_03_wrong_action_evidence_rejected(self):
        query = "I cannot download or install apps from the App Store, the download button is unresponsive."
        candidate = {
            "customer": "app crashing on launch",
            "response": "Force close the app and restart your device when it crashes unexpectedly.",
        }
        safe, reason = is_evidence_safe(query, candidate, query_intent="app_problems")
        assert not safe, f"Expected unsafe for action divergence (download vs crash), got safe=True ({reason})"

    # 4. Generic evidence rejected (pure restart boilerplate)
    def test_04_generic_evidence_rejected(self):
        query = "My screen shows a green horizontal line across the top."
        candidate = {
            "customer": "screen issue",
            "response": "Restart your iPhone.",
        }
        safe, reason = is_evidence_safe(query, candidate, query_intent="screen_display")
        assert not safe, "Expected pure restart boilerplate to be rejected as generic"
        assert "generic" in reason.lower()

    # 5. Resolved customer message handled without troubleshooting
    def test_05_resolved_customer_message_handled(self):
        query = "Never mind, I figured it out on my own and it is working now!"
        assert is_message_resolved(query), "Should identify message as resolved"

        res = self.generator.generate(query)
        assert res["response_type"] == "resolved_acknowledgement"
        assert res["grounded"] is True
        assert not res["escalate"]
        assert len(res["unsupported_claims"]) == 0
        # Must not contain troubleshooting steps
        assert "restart" not in res["reply"].lower()
        assert "settings" not in res["reply"].lower()

    # 6. Ambiguous customer message produces clarification without hallucinated facts
    def test_06_ambiguous_customer_message_clarification(self):
        query = "It is not working today."
        res = self.generator.generate(query)
        assert res["response_type"] == "insufficient_evidence"
        assert res["grounded"] is True
        assert len(res["unsupported_claims"]) == 0
        # Clarification must prompt for details without inventing technical directives
        assert any(phrase in res["reply"].lower() for phrase in ["could you", "narrow this down", "help me understand", "what is happening"])

    # 7. Unsupported technical claim flagged and rejected
    def test_07_unsupported_technical_claim_flagged(self):
        query = "My phone will not charge."
        unsafe_evidence = {
            "evidence_safe": False,
            "rejection_reason": "unsafe_unverified_steps",
        }
        reply = "Based on a similar case, press the hidden reset pin behind the speaker grille."
        chk = perform_deterministic_grounding_check(query, reply, unsafe_evidence)
        assert not chk["grounded"]
        assert len(chk["unsupported_claims"]) > 0

    # 8. Account security escalation (hacked, unauthorized login)
    def test_08_account_security_escalation(self):
        query = "My account was hacked and someone changed my Apple ID email without my permission!"
        sensitive, reason = check_sensitive_topic(query)
        assert sensitive, "Hacking query must be detected as sensitive"

        res = self.generator.generate(query)
        assert res["escalate"] is True
        assert res["escalation_decision"] == "ESCALATE"
        assert "security" in res["escalation_reason"].lower() or "hack" in res["escalation_reason"].lower()

    # 9. Payment / refund escalation (charged twice, unauthorized charge)
    def test_09_payment_refund_escalation(self):
        query = "I was charged twice on my credit card for an in-app purchase and I demand an immediate refund."
        sensitive, reason = check_sensitive_topic(query)
        assert sensitive, "Billing / refund query must be detected as sensitive"

        res = self.generator.generate(query)
        assert res["escalate"] is True
        assert res["escalation_decision"] == "ESCALATE"

    # 10. Account access lockout escalation
    def test_10_account_access_lockout_escalation(self):
        query = "My Apple ID has been disabled and locked for security reasons and I cannot access my files."
        sensitive, reason = check_sensitive_topic(query)
        assert sensitive, "Account lockout must be detected as sensitive"

        res = self.generator.generate(query)
        assert res["escalate"] is True
        assert res["escalation_decision"] == "ESCALATE"

    # 11. SIM carrier unlocking vs physical SIM tray hardware distinction
    def test_11_sim_carrier_unlocking_vs_physical_sim_tray(self):
        query = "How do I unlock my iPhone from AT&T carrier so I can use another SIM?"
        candidate = {
            "customer": "sim tray stuck",
            "response": "Insert a paperclip into the small hole beside the SIM tray to pop open the drawer.",
        }
        safe, reason = is_evidence_safe(query, candidate, query_intent="calls_cellular")
        assert not safe, f"Expected carrier unlock vs physical SIM tray divergence rejection, got {reason}"

        # Test intent classifier distinguishes physical hardware
        hardware_query = "My SIM card tray is physically bent and stuck inside the slot."
        pred = self.classifier.predict(hardware_query)
        assert pred["intent"] == "device_hardware"

    # 12. iMessage vs Wi-Fi confusion prevention
    def test_12_imessage_vs_wifi_confusion_prevention(self):
        query = "My iMessage is stuck on waiting for activation."
        candidate = {
            "customer": "wifi disconnected",
            "response": "Reset your Wi-Fi router and forget this network in Settings > Wi-Fi.",
        }
        safe, reason = is_evidence_safe(query, candidate, query_intent="apple_id_icloud")
        assert not safe, f"Expected iMessage vs Wi-Fi divergence rejection, got {reason}"

    # 13. App download vs app crash confusion prevention
    def test_13_app_download_vs_app_crash_confusion_prevention(self):
        crash_query = "The Instagram app keeps crashing and freezing upon opening."
        download_candidate = {
            "customer": "app store payment",
            "response": "Check your payment method and billing address in the App Store to download apps.",
        }
        safe, reason = is_evidence_safe(crash_query, download_candidate, query_intent="app_problems")
        assert not safe, f"Expected app crash vs download divergence rejection, got {reason}"

    # 14. Cellular vs Wi-Fi confusion prevention
    def test_14_cellular_vs_wifi_confusion_prevention(self):
        cell_query = "I have no cellular data service or LTE signal reception on my iPhone."
        pred_cell = self.classifier.predict(cell_query)
        assert pred_cell["intent"] == "calls_cellular"

        wifi_query = "My Wi-Fi keeps disconnecting from my home wireless network."
        pred_wifi = self.classifier.predict(wifi_query)
        assert pred_wifi["intent"] == "wifi_connectivity"

    # 15. Battery draining vs general hardware confusion prevention
    def test_15_battery_draining_vs_general_hardware_confusion_prevention(self):
        battery_query = "My battery is draining rapidly and loses 50% charge in one hour."
        pred = self.classifier.predict(battery_query)
        assert pred["intent"] == "battery_charging"
        assert pred["confidence"] >= 0.7

        hardware_candidate = {
            "customer": "camera broken",
            "response": "Inspect the camera lens and speaker grill for physical damage.",
        }
        safe, reason = is_evidence_safe(battery_query, hardware_candidate, query_intent="battery_charging")
        assert not safe, f"Expected battery vs hardware mismatch rejection, got {reason}"

    # 16. Hardware safety hazard escalation (swollen / hot battery)
    def test_16_hardware_safety_hazard_escalation(self):
        query = "My battery is swollen and getting hot."
        sensitive, reason = check_sensitive_topic(query)
        assert sensitive, "Swollen and hot battery must be detected as sensitive"
        assert reason == "hardware_hazard"

        res = self.generator.generate(query)
        assert res["escalate"] is True
        assert res["escalation_decision"] == "ESCALATE"
        assert res["escalation_reason"] == "hardware_hazard"
        assert res["safe_to_answer"] is False
        assert res["evidence_used"] is False
        assert "stop using" in res["reply"].lower() or "charge" in res["reply"].lower()
