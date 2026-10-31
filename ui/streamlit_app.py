"""Streamlit UI — thin client for the Clinical RAG FastAPI backend.

Calls: POST /ask, GET /sources, GET /health
Does NOT import or call retrieval/generation services directly.
"""
import sys
import time
from typing import Any, Dict, Optional

import httpx
import streamlit as st

BACKEND_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 60.0

DISCLAIMER = (
    "Academic prototype for source-based medication-information lookup. "
    "Not clinically validated. Verify source information before professional use. "
    "Do not enter patient identifiers."
)

STATUS_EMOJI = {
    "answer": "✅",
    "clarify": "❓",
    "insufficient_evidence": "⚠️",
    "out_of_scope": "🚫",
    "drug_not_covered": "📋",
    "processing_error": "❌",
}

STATUS_COLOR = {
    "answer": "green",
    "clarify": "orange",
    "insufficient_evidence": "orange",
    "out_of_scope": "red",
    "drug_not_covered": "gray",
    "processing_error": "red",
}


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical AI Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── helpers ───────────────────────────────────────────────────────────────────
def get_health() -> Optional[Dict[str, Any]]:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_sources() -> Optional[Dict[str, Any]]:
    try:
        r = httpx.get(f"{BACKEND_URL}/sources", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_ask(question: str) -> Optional[Dict[str, Any]]:
    try:
        r = httpx.post(
            f"{BACKEND_URL}/ask",
            json={"question": question},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 422:
            return {"_http_error": True, "status_code": 422, "detail": r.json()}
        if r.status_code in (500, 502, 503):
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            return {"_http_error": True, "status_code": r.status_code, "detail": detail}
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return {"_connection_error": True}
    except Exception as exc:
        return {"_exception": str(exc)}


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💊 Clinical AI Assistant")
    st.caption("Academic medication-label information prototype")
    st.divider()

    health = get_health()
    if health:
        st.success("Backend: Online")
        st.caption(f"Model: `{health.get('model', 'unknown')}`")
        labels = health.get("corpus_labels", [])
        st.caption(f"Corpus: `{', '.join(labels)}`")
    else:
        st.error("Backend: Offline — start FastAPI before using the UI.")
        st.caption(f"Expected at: `{BACKEND_URL}`")

    st.divider()
    sources = get_sources()
    if sources and sources.get("sources"):
        st.subheader("Covered Drug Labels")
        for src in sources["sources"]:
            with st.expander(f"🧪 {src.get('ingredient', src.get('label_id', ''))}"):
                st.write(f"**Product:** {src.get('product_title', '')}")
                st.write(f"**Formulation:** {src.get('formulation', '')} ({src.get('route', '')})")
                st.write(f"**SPL Version:** {src.get('spl_version', '')}")
                st.write(f"**Effective Date:** {src.get('effective_date', '')}")
                st.write(f"**Label ID:** `{src.get('label_id', '')}`")
                if src.get("source_url"):
                    st.write(f"[DailyMed Source]({src['source_url']})")
    else:
        st.caption("Corpus information unavailable.")

    st.divider()
    st.subheader("Scope")
    st.markdown(
        """
- **In scope:** Contraindications, warnings & precautions, renal impairment, drug interactions
- **Out of scope:** Diagnosis, prescribing, dose calculation, patient-specific advice, triage
        """
    )
    st.divider()
    st.warning(DISCLAIMER, icon="⚠️")


# ── main area ─────────────────────────────────────────────────────────────────
st.header("Clinical AI Assistant")
st.caption(
    "An Evidence-Based Clinical AI Assistant for Healthcare Professionals "
    "Using Medical Large Language Models and Retrieval-Augmented Generation"
)
st.info(DISCLAIMER, icon="⚠️")
st.divider()

question_input = st.text_area(
    "Enter your clinical question about a covered medication label:",
    height=100,
    max_chars=1000,
    placeholder="e.g. What does the metformin label say about renal impairment?",
    key="question_input",
)

submit = st.button("Submit Question", type="primary", disabled=not bool(question_input.strip()))

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if submit and question_input.strip():
    with st.spinner("Querying the clinical RAG pipeline…"):
        result = post_ask(question_input.strip())
    st.session_state.last_response = result

response = st.session_state.last_response

if response:
    st.divider()

    if response.get("_connection_error"):
        st.error(
            f"Cannot connect to the backend at `{BACKEND_URL}`. "
            "Please start the FastAPI server first:\n\n"
            "`python -m uvicorn app.main:app --reload --port 8000`"
        )

    elif response.get("_exception"):
        st.error(f"Unexpected error: {response['_exception']}")

    elif response.get("_http_error"):
        code = response.get("status_code")
        detail = response.get("detail", "")
        if code == 422:
            st.error(f"**Invalid Request (422):** {detail}")
        elif code in (502, 503):
            st.error(f"**Provider/Gateway Error ({code}):** {detail}")
        else:
            st.error(f"**Server Error ({code}):** {detail}")

    else:
        raw_status = response.get("status", "unknown")
        emoji = STATUS_EMOJI.get(raw_status, "ℹ️")
        color = STATUS_COLOR.get(raw_status, "gray")

        col1, col2 = st.columns([1, 5])
        with col1:
            st.metric("Status", f"{emoji} {raw_status.upper()}")
        with col2:
            req_id = response.get("request_id", "")
            latency = response.get("latency_seconds", 0.0)
            prompt_t = response.get("prompt_tokens", 0)
            comp_t = response.get("completion_tokens", 0)
            total_t = response.get("total_tokens", 0)
            st.caption(
                f"Request ID: `{req_id}` | "
                f"Model: `{response.get('model', '')}` | "
                f"Prompt: `{response.get('prompt_version', '')}` | "
                f"Latency: **{latency:.2f}s** | "
                f"Tokens: {prompt_t} + {comp_t} = {total_t}"
            )

        if raw_status == "answer":
            st.subheader("Answer")
            st.write(response.get("answer", ""))

            evidence = response.get("evidence", [])
            if evidence:
                st.subheader(f"Supporting Evidence ({len(evidence)} citation(s))")
                for i, ev in enumerate(evidence, start=1):
                    pid = ev.get("passage_id", "")
                    sec_title = ev.get("section_title", "")
                    sec_type = ev.get("section_type", "")
                    sec_path = ev.get("section_path", "")
                    quote = ev.get("quote", "")
                    src_url = ev.get("source_url", "")
                    ver = ev.get("label_version", "")
                    eff = ev.get("effective_date", "")

                    with st.expander(f"[{i}] {pid} — {sec_title}", expanded=(i == 1)):
                        st.markdown(f"**Section:** {sec_title} (`{sec_type}`)")
                        st.markdown(f"**Path:** {sec_path}")
                        st.markdown("**Exact Quote:**")
                        st.info(f'"{quote}"')
                        st.markdown(
                            f"**Source:** [DailyMed Label]({src_url})  "
                            f"| SPL v{ver} | Effective: {eff}"
                        )

        elif raw_status == "clarify":
            st.subheader("Clarification Needed")
            st.warning(response.get("clarification_question", ""))

        elif raw_status == "insufficient_evidence":
            st.subheader("Insufficient Evidence in Corpus")
            st.warning(response.get("answer", "The retrieved passages do not explicitly support an answer."))

        elif raw_status == "out_of_scope":
            st.subheader("Request Out of Scope")
            st.error(response.get("answer", ""))

        elif raw_status == "drug_not_covered":
            st.subheader("Drug Not Covered")
            st.warning(response.get("answer", ""))

        limitations = response.get("limitations", [])
        if limitations:
            st.caption("Limitations: " + " | ".join(limitations))
