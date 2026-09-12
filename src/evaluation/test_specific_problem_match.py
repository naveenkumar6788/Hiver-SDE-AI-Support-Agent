import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retrieval.evidence_relevance import (
    extract_problem_terms,
    detect_response_problem,
    detect_problem_conflict,
    calculate_customer_problem_similarity,
    calculate_response_problem_similarity,
    calculate_domain_match,
    calculate_action_match,
    calculate_final_evidence_score,
    is_specific_evidence_safe,
    calculate_evidence_relevance,
    extract_domain_entities,
)


class TestSpecificProblemMatch(unittest.TestCase):
    """Unit tests for the Specific Problem Match Layer in evidence evaluation."""

    def test_01_ads_in_apps_conflict(self):
        query = "my son is being exposed to inappropriate ads in ad supported kid's games we download from the Apple Store. How can I prevent him from seeing them?"
        hist_customer = "Problem with games downloading"
        hist_response = "Have you tried restarting your device yet when downloading those applications?"
        
        q_probs = extract_problem_terms(query)
        conflict, reason = detect_problem_conflict(query, q_probs, hist_customer, hist_response)
        self.assertTrue(conflict)
        self.assertIn("ads_in_apps", reason)
        
        safe, safe_reason = is_specific_evidence_safe(query, hist_customer, hist_response, "app_store_downloads", "app_store_downloads")
        self.assertFalse(safe)

    def test_02_ios_downgrade_conflict(self):
        query = "I wish @AppleSupport would let me download the previous iOS so that my phone can actually work normal again"
        hist_customer = "I want to go back to iOS 10"
        hist_response = "An update has been released to assist with this issue. Please update to the latest iOS."
        
        q_probs = extract_problem_terms(query)
        conflict, reason = detect_problem_conflict(query, q_probs, hist_customer, hist_response)
        self.assertTrue(conflict)
        self.assertIn("ios_downgrade", reason)

        safe, safe_reason = is_specific_evidence_safe(query, hist_customer, hist_response, "ios_update", "ios_update")
        self.assertFalse(safe)

    def test_03_screen_orientation_conflict(self):
        query = "Other applications are the same. Horizontal position is problem when on a call."
        hist_customer = "Horizontal screen not working"
        hist_response = "Check for an update in the App Store for WhatsApp to resolve this."
        
        q_probs = extract_problem_terms(query)
        conflict, reason = detect_problem_conflict(query, q_probs, hist_customer, hist_response)
        self.assertTrue(conflict)
        self.assertIn("screen_orientation", reason)

        safe, _ = is_specific_evidence_safe(query, hist_customer, hist_response, "screen_display", "screen_display")
        self.assertFalse(safe)

    def test_04_device_heating_conflict(self):
        query = "Why did you send me a phone with a heating issue?"
        hist_customer = "Phone is overheating"
        hist_response = "If your screen is dark, try to restore in recovery mode."
        
        q_probs = extract_problem_terms(query)
        conflict, reason = detect_problem_conflict(query, q_probs, hist_customer, hist_response)
        self.assertTrue(conflict)
        self.assertIn("device_heating", reason)

        safe, _ = is_specific_evidence_safe(query, hist_customer, hist_response, "device_hardware", "device_hardware")
        self.assertFalse(safe)

    def test_05_messages_from_email_conflict(self):
        query = "All messages coming from my email instead of phone number."
        hist_customer = "iMessage send address"
        hist_response = "A new software update for iOS 11.1 has been released."
        
        q_probs = extract_problem_terms(query)
        conflict, reason = detect_problem_conflict(query, q_probs, hist_customer, hist_response)
        self.assertTrue(conflict)
        self.assertIn("imessage_send_address", reason)

        safe, _ = is_specific_evidence_safe(query, hist_customer, hist_response, "calls_cellular", "calls_cellular")
        self.assertFalse(safe)

    def test_06_server_port_connection_conflict(self):
        query = "Connections to host on default ports 22, 80, 443 failed"
        hist_customer = "Network connection failed"
        hist_response = "Check your router security settings and re-enter your wifi password."
        
        q_probs = extract_problem_terms(query)
        conflict, reason = detect_problem_conflict(query, q_probs, hist_customer, hist_response)
        self.assertTrue(conflict)
        self.assertIn("server_port_connection", reason)

        safe, _ = is_specific_evidence_safe(query, hist_customer, hist_response, "wifi_connectivity", "wifi_connectivity")
        self.assertFalse(safe)

    def test_07_headphones_vs_macos_conflict(self):
        query = "what should I do now that your shitty headphones broke"
        hist_customer = "headphones broke"
        hist_response = "We've got your back to get macOS working for you. How long have you had this issue?"
        
        q_probs = extract_problem_terms(query)
        conflict, reason = detect_problem_conflict(query, q_probs, hist_customer, hist_response)
        self.assertTrue(conflict)
        self.assertIn("headphones", reason)

        safe, _ = is_specific_evidence_safe(query, hist_customer, hist_response, "device_hardware", "device_hardware")
        self.assertFalse(safe)

    def test_08_mms_vs_wifi_conflict(self):
        query = "Since updating to 11.0.3 I can only send an MMS"
        hist_customer = "MMS issue"
        hist_response = "We understand your concerns with Wi-Fi and we'll be happy to help check your router."
        
        q_probs = extract_problem_terms(query)
        conflict, reason = detect_problem_conflict(query, q_probs, hist_customer, hist_response)
        self.assertTrue(conflict)
        self.assertIn("mms_sms vs wifi", reason)

        safe, _ = is_specific_evidence_safe(query, hist_customer, hist_response, "calls_cellular", "calls_cellular")
        self.assertFalse(safe)

    def test_09_battery_drain_safe_acceptance(self):
        query = "My battery is draining super fast after the update"
        hist_customer = "iPhone battery draining quickly after update"
        hist_response = "Check Settings > Battery to see which app is using the most battery usage, or enable Low Power Mode."
        
        score_info = calculate_final_evidence_score(
            query=query,
            historical_customer_message=hist_customer,
            historical_response=hist_response,
            current_intent="battery_charging",
            historical_intent="battery_charging"
        )
        self.assertFalse(score_info["conflict_detected"])
        self.assertGreaterEqual(score_info["final_evidence_score"], 0.70)
        self.assertTrue(score_info["evidence_safe"])

    def test_10_wifi_disconnecting_safe_acceptance(self):
        query = "My iPhone cannot connect to Wi-Fi at all"
        hist_customer = "Cannot connect to Wi-Fi network"
        hist_response = "Try going to Settings > Wi-Fi, tap your network, tap Forget This Network, then reconnect."
        
        safe, reason = is_specific_evidence_safe(
            query=query,
            evidence_customer=hist_customer,
            evidence_response=hist_response,
            current_intent="wifi_connectivity",
            historical_intent="wifi_connectivity"
        )
        self.assertTrue(safe)
        self.assertIn("accepted", reason)

    def test_11_generic_boilerplate_rejection(self):
        query = "My phone will not charge"
        hist_customer = "Phone not charging"
        hist_response = "We'd be happy to help out. Please send us a DM so we can assist."
        
        safe, reason = is_specific_evidence_safe(
            query=query,
            evidence_customer=hist_customer,
            evidence_response=hist_response,
            current_intent="battery_charging",
            historical_intent="battery_charging"
        )
        self.assertFalse(safe)
        self.assertTrue(any(w in reason for w in ["generic", "no_useful_guidance"]))

    def test_12_compatibility_wrapper(self):
        query = "My battery drains very fast"
        hist_cust = "Battery draining quickly after update"
        hist_resp = "Check Settings > Battery to view battery usage and turn on Low Power Mode."
        
        result = calculate_evidence_relevance(
            query=query,
            historical_customer_message=hist_cust,
            historical_response=hist_resp,
            current_intent="battery_charging",
            historical_intent="battery_charging"
        )
        self.assertIn("evidence_relevance_score", result)
        self.assertIn("problem_match_score", result)
        self.assertIn("customer_problem_similarity", result)
        self.assertIn("response_problem_similarity", result)
        self.assertIn("domain_match", result)
        self.assertIn("action_match", result)
        self.assertIn("conflict_detected", result)
        self.assertIn("evidence_safe", result)
        self.assertTrue(result["evidence_safe"])


if __name__ == "__main__":
    unittest.main()
