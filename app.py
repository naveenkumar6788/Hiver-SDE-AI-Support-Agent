import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.reply_generator import ReplyGenerator

st.set_page_config(
    page_title="Hiver AI Support Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("Hiver AI Support Agent")
st.caption("Manual demo of the evaluated AppleSupport customer-support pipeline")
st.divider()


@st.cache_resource
def load_agent():
    return ReplyGenerator()


try:
    agent = load_agent()
except Exception as e:
    st.error("Failed to load the support agent.")
    st.exception(e)
    st.stop()

st.subheader("Customer Message")

query = st.text_area(
    "Enter a customer support message:",
    placeholder="Example: My iPhone battery is draining very fast.",
    height=120
)

if st.button("Analyze & Generate Reply", type="primary"):
    if not query.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Running support agent..."):
        try:
            result = agent.generate(query=query.strip())
        except Exception as e:
            st.error("Agent execution failed.")
            st.exception(e)
            st.stop()

    st.divider()

    decision = result.get("escalation_decision", "UNKNOWN")
    if decision == "ESCALATE":
        st.error(f"Decision: {decision}")
    else:
        st.success(f"Decision: {decision}")

    st.subheader("1. Intent Classification")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Intent", result.get("intent", "unknown"))

    with col2:
        confidence = result.get("intent_confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        st.metric("Confidence", f"{confidence:.2f}")

    with col3:
        st.metric("Detected Domains", len(result.get("domains", [])))

    domains = result.get("domains", [])
    if domains:
        st.write("**Domains:**", ", ".join(domains))

    st.subheader("2. Escalation Decision")
    escalation_reason = result.get("escalation_reason", "")
    if decision == "ESCALATE":
        st.warning(f"**Reason:** {escalation_reason or 'Sensitive or unsafe case'}")
    else:
        st.info("The agent considers this case suitable for automatic handling.")

    st.subheader("3. Historical Evidence")
    evidence_used = result.get("evidence_used", False)
    evidence_safe = result.get("evidence_safe", False)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Evidence Retrieved", result.get("evidence_retrieved", 0))
    with col2:
        st.metric("Safe Candidates", result.get("evidence_candidates", 0))
    with col3:
        if evidence_safe:
            st.success("Evidence Safe")
        else:
            st.warning("No Safe Evidence")

    evidence = result.get("evidence")
    if evidence:
        with st.expander("View selected historical evidence", expanded=True):
            st.write(
                evidence.get(
                    "response_text",
                    evidence.get(
                        "response",
                        evidence.get(
                            "historical_response",
                            "No response text available."
                        )
                    )
                )
            )
            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Evidence relevance:**", f"{result.get('evidence_relevance', 0.0):.3f}")
            with c2:
                st.write("**Problem match:**", f"{result.get('problem_match_score', 0.0):.3f}")
            with c3:
                st.write("**Final evidence score:**", f"{result.get('final_evidence_score', 0.0):.3f}")
    else:
        st.info("No historical evidence was considered safe enough to use.")
        rejection = result.get("evidence_rejection_reason", "")
        if rejection:
            st.write("**Evidence rejection reason:**", rejection)

    st.subheader("4. Draft Reply")
    reply = result.get("reply", "")
    st.text_area("Customer-facing response:", value=reply, height=180)

    st.subheader("5. Safety & Grounding")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if result.get("grounded", False):
            st.success("Grounded")
        else:
            st.error("Not Grounded")
    with col2:
        if result.get("unsupported_claims"):
            st.error("Unsupported Claims")
        else:
            st.success("No Unsupported Claims")
    with col3:
        if result.get("safe_to_answer", False):
            st.success("Safe to Answer")
        else:
            st.error("Not Safe to Answer")
    with col4:
        if result.get("sensitive", False):
            st.warning("Sensitive")
        else:
            st.success("Not Sensitive")

    st.subheader("6. Response Metadata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Response type:**", result.get("response_type", ""))
    with col2:
        st.write("**Evidence used:**", "Yes" if evidence_used else "No")
    with col3:
        st.write("**Escalate:**", "Yes" if result.get("escalate", False) else "No")

    with st.expander("View complete agent output"):
        st.json(result)