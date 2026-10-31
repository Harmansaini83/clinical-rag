"""Streamlit UI: client for the Clinical RAG FastAPI backend.

Calls: POST /ask, GET /sources, GET /health
Does NOT import or call retrieval/generation services directly.
"""
from html import escape as esc
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

BACKEND_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 60.0

TITLE = (
    "An Evidence-Based Clinical AI Assistant for Healthcare Professionals "
    "Using Medical Large Language Models and Retrieval-Augmented Generation"
)

DISCLAIMER = (
    "Academic project for source-based medication-information lookup. "
    "Not clinically validated. Verify source information before professional use. "
    "Do not enter patient identifiers."
)

STATUS_META = {
    "answer": ("Answer", "✅", "#22c55e", "Evidence found and quotes validated against the label."),
    "clarify": ("Clarification needed", "❓", "#f59e0b", "The question needs more detail before it can be answered."),
    "insufficient_evidence": ("Insufficient evidence", "⚠️", "#f59e0b", "The indexed label passages do not explicitly support an answer."),
    "out_of_scope": ("Out of scope", "🚫", "#ef4444", "This type of request is not supported."),
    "drug_not_covered": ("Drug not covered", "📋", "#94a3b8", "This drug is not in the indexed corpus."),
    "processing_error": ("Processing error", "❌", "#ef4444", "The request could not be completed."),
}

EXAMPLES = [
    ("🧾 Contraindications", "What are the contraindications of metformin?"),
    ("🫘 Renal impairment", "What does the label recommend for use in renal impairment?"),
    ("🧪 Lactic acidosis", "What are the clinical features of metformin-associated lactic acidosis?"),
    ("💊 Vitamin B12", "What does the label say about vitamin B12 deficiency and how often to monitor?"),
    ("🚫 Out of scope", "Calculate the dose for a 60 kg patient with an eGFR of 40."),
    ("📋 Other drug", "What are the side effects of atorvastatin?"),
]

PIPELINE_STEPS = [
    "Scope check",
    "Drug resolution",
    "Retrieval",
    "Evidence check",
    "Generation",
    "Quote validation",
]


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical AI Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 55%, #8b5cf6 100%);
    border-radius: 20px;
    padding: 34px 38px;
    margin-bottom: 22px;
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.28);
}
.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.18);
    color: #ffffff;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 14px;
}
.hero h1 { color: #ffffff; font-size: 2.5rem; font-weight: 800; margin: 0 0 8px 0; padding: 0; }
.hero p { color: rgba(255, 255, 255, 0.88); font-size: 0.98rem; margin: 0 0 16px 0; line-height: 1.5; }
.chip {
    display: inline-block;
    background: rgba(255, 255, 255, 0.16);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.28);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    margin: 0 8px 6px 0;
}

.notice {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.40);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.88rem;
    margin-bottom: 18px;
}

.section-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.65;
    margin: 18px 0 8px 0;
}

.status-card {
    border-radius: 16px;
    padding: 18px 22px;
    margin: 6px 0 14px 0;
    border: 1px solid rgba(148, 163, 184, 0.25);
    background: rgba(148, 163, 184, 0.07);
}
.status-title { font-size: 1.3rem; font-weight: 800; margin: 0; }
.status-desc { opacity: 0.8; font-size: 0.92rem; margin-top: 4px; }

.answer-card {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.30);
    border-radius: 14px;
    padding: 18px 22px;
    line-height: 1.65;
    font-size: 1.02rem;
}

.quote-card {
    background: rgba(14, 165, 233, 0.08);
    border-left: 4px solid #0ea5e9;
    border-radius: 8px;
    padding: 14px 18px;
    font-style: italic;
    line-height: 1.6;
    margin: 6px 0 12px 0;
}
.meta-line { opacity: 0.75; font-size: 0.85rem; }

.pipeline { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 4px 0; }
.step {
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid rgba(148, 163, 184, 0.30);
}
.step.done { background: rgba(34, 197, 94, 0.14); border-color: rgba(34, 197, 94, 0.55); }
.step.stop { background: rgba(239, 68, 68, 0.14); border-color: rgba(239, 68, 68, 0.55); }
.step.skip { opacity: 0.4; }

div.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.25);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    border: none;
    color: #ffffff;
}

[data-testid="stMetric"] {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 14px;
    padding: 12px 16px;
}

.footer { text-align: center; opacity: 0.6; font-size: 0.82rem; margin-top: 36px; }
</style>
""",
    unsafe_allow_html=True,
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


def pipeline_html(status: str) -> str:
    """Show how far the request travelled through the pipeline."""
    if status == "answer":
        states = ["done"] * 6
    elif status == "out_of_scope":
        states = ["stop"] + ["skip"] * 5
    elif status in ("drug_not_covered", "clarify"):
        states = ["done", "stop"] + ["skip"] * 4
    elif status == "insufficient_evidence":
        states = ["done", "done", "done", "stop", "skip", "skip"]
    else:
        return ""
    icons = {"done": "✓", "stop": "■", "skip": "·"}
    parts = [
        f'<span class="step {state}">{icons[state]} {name}</span>'
        for name, state in zip(PIPELINE_STEPS, states)
    ]
    return '<div class="pipeline">' + "".join(parts) + "</div>"


def text_html(text: str) -> str:
    return esc(text).replace("\n", "<br>")


def use_example(question: str) -> None:
    st.session_state.question_input = question
    st.session_state.pending_submit = True


def clear_question() -> None:
    st.session_state.question_input = ""
    st.session_state.last_response = None


# ── session state ─────────────────────────────────────────────────────────────
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_submit" not in st.session_state:
    st.session_state.pending_submit = False
if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💊 Clinical AI Assistant")
    st.caption("Source-grounded medication-label information")
    st.divider()

    health = get_health()
    if health:
        st.success("Backend: Online")
        st.caption(f"Model: `{health.get('model', 'unknown')}`")
        labels = health.get("corpus_labels", [])
        st.caption(f"Corpus: `{', '.join(labels)}`")
    else:
        st.error("Backend: Offline. Start FastAPI before using the UI.")
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

    if st.session_state.history:
        st.divider()
        st.subheader("Recent questions")
        for item in reversed(st.session_state.history[-6:]):
            emoji = STATUS_META.get(item["status"], ("", "ℹ️", "", ""))[1]
            short = item["question"]
            if len(short) > 55:
                short = short[:55] + "…"
            st.caption(f"{emoji} {short}")

    st.divider()
    st.warning(DISCLAIMER, icon="⚠️")


# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="hero">
<div class="hero-badge">FDA DAILYMED · HYBRID RETRIEVAL · EXACT-QUOTE VALIDATION</div>
<h1>Clinical AI Assistant</h1>
<p>{esc(TITLE)}</p>
<span class="chip">📚 Answers only from the official label</span>
<span class="chip">🔎 Hybrid BM25 + dense retrieval</span>
<span class="chip">📌 Character-exact quotes</span>
<span class="chip">🛑 Abstains when evidence is missing</span>
</div>
<div class="notice">⚠️ {esc(DISCLAIMER)}</div>
""",
    unsafe_allow_html=True,
)

# ── question input ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
example_cols = st.columns(3)
for idx, (label, question) in enumerate(EXAMPLES):
    with example_cols[idx % 3]:
        st.button(
            label,
            key=f"example_{idx}",
            use_container_width=True,
            on_click=use_example,
            args=(question,),
        )

st.markdown('<div class="section-label">Ask a question</div>', unsafe_allow_html=True)
question_input = st.text_area(
    "Enter your clinical question about a covered medication label:",
    height=110,
    max_chars=1000,
    placeholder="e.g. What does the metformin label say about renal impairment?",
    key="question_input",
)

btn_col1, btn_col2, _ = st.columns([2, 1, 5])
with btn_col1:
    submit = st.button(
        "🔍 Submit Question",
        type="primary",
        use_container_width=True,
        disabled=not bool(question_input.strip()),
    )
with btn_col2:
    st.button("Clear", use_container_width=True, on_click=clear_question)

should_run = (submit or st.session_state.pending_submit) and bool(question_input.strip())
st.session_state.pending_submit = False

if should_run:
    with st.spinner("Querying the clinical RAG pipeline…"):
        result = post_ask(question_input.strip())
    st.session_state.last_response = result
    status_for_history = "error"
    if isinstance(result, dict) and result.get("status"):
        status_for_history = result["status"]
    st.session_state.history.append(
        {"question": question_input.strip(), "status": status_for_history}
    )

response = st.session_state.last_response

# ── response ──────────────────────────────────────────────────────────────────
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
        title, emoji, color, desc = STATUS_META.get(
            raw_status, (raw_status.replace("_", " ").title(), "ℹ️", "#94a3b8", "")
        )

        st.markdown(
            f"""
<div class="status-card" style="border-left: 6px solid {color};">
<p class="status-title">{emoji} {esc(title)}</p>
<div class="status-desc">{esc(desc)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        pipe = pipeline_html(raw_status)
        if pipe:
            st.markdown('<div class="section-label">Pipeline trace</div>', unsafe_allow_html=True)
            st.markdown(pipe, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency", f"{response.get('latency_seconds', 0.0):.2f} s")
        m2.metric("Prompt tokens", f"{response.get('prompt_tokens', 0):,}")
        m3.metric("Completion tokens", f"{response.get('completion_tokens', 0):,}")
        m4.metric("Total tokens", f"{response.get('total_tokens', 0):,}")
        st.caption(
            f"Request ID: `{response.get('request_id', '')}` | "
            f"Model: `{response.get('model', '')}` | "
            f"Prompt: `{response.get('prompt_version', '')}`"
        )

        if raw_status == "answer":
            st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-card">{text_html(response.get("answer", ""))}</div>',
                unsafe_allow_html=True,
            )

            evidence = response.get("evidence", [])
            if evidence:
                st.markdown(
                    f'<div class="section-label">Supporting evidence · {len(evidence)} citation(s)</div>',
                    unsafe_allow_html=True,
                )
                for i, ev in enumerate(evidence, start=1):
                    pid = ev.get("passage_id", "")
                    sec_title = ev.get("section_title", "")
                    sec_type = ev.get("section_type", "")
                    sec_path = ev.get("section_path", "")
                    quote = ev.get("quote", "")
                    src_url = ev.get("source_url", "")
                    ver = ev.get("label_version", "")
                    eff = ev.get("effective_date", "")

                    with st.expander(f"[{i}] {pid} · {sec_title}", expanded=(i == 1)):
                        st.markdown(f"**Section:** {sec_title} (`{sec_type}`)")
                        st.markdown(f"**Path:** {sec_path}")
                        st.markdown("**Exact quote from the label:**")
                        st.markdown(
                            f'<div class="quote-card">“{text_html(quote)}”</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Source:** [DailyMed Label]({src_url})  "
                            f"| SPL v{ver} | Effective: {eff}"
                        )

        elif raw_status == "clarify":
            st.markdown('<div class="section-label">Clarification needed</div>', unsafe_allow_html=True)
            st.warning(response.get("clarification_question", ""))

        elif raw_status == "insufficient_evidence":
            st.markdown('<div class="section-label">Why no answer was given</div>', unsafe_allow_html=True)
            st.warning(
                response.get("answer", "The retrieved passages do not explicitly support an answer.")
            )

        elif raw_status == "out_of_scope":
            st.markdown('<div class="section-label">Why this was declined</div>', unsafe_allow_html=True)
            st.error(response.get("answer", ""))

        elif raw_status == "drug_not_covered":
            st.markdown('<div class="section-label">Coverage</div>', unsafe_allow_html=True)
            st.warning(response.get("answer", ""))

        limitations: List[str] = response.get("limitations", [])
        if limitations:
            st.caption("Limitations: " + " | ".join(limitations))

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer"> Answers are drawn only from the indexed DailyMed label · Not clinically validated</div>',
    unsafe_allow_html=True,
)
