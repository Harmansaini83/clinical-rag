# Clinical RAG Prototype

**An Evidence-Based Clinical AI Assistant for Healthcare Professionals Using Medical Large Language Models and Retrieval-Augmented Generation**

> Academic prototype for source-based medication-information lookup. Not clinically validated. Verify source information before professional use. Do not enter patient identifiers.

The prototype uses the general-purpose openai/gpt-oss-120b model through Groq within a medical-information RAG workflow.

---

## One-Paragraph Description

The Clinical RAG Prototype is an evidence-based medication-information assistant designed to answer clinical queries directly and exclusively from official FDA DailyMed package inserts. The system couples dense semantic vector retrieval (all-MiniLM-L6-v2 via persistent ChromaDB) and lexical BM25 retrieval fused using Reciprocal Rank Fusion (RRF $k=60$) with an openai/gpt-oss-120b generator running through Groq. A local deterministic evidence gate verifies label coverage, relevant section presence, and mandatory clinical qualifiers before generation, enabling early abstention on unanswerable questions without invoking external LLM calls. All model outputs are subjected to strict schema validation and character-exact quote matching against the authoritative label passages, guaranteeing traceability, numerical precision, and provenance for healthcare professionals.

---

## Architecture Diagram

```
[ Healthcare Professional / User Query ]
                  │
                  ▼
         [ 1. Local Scope Check ]
         (Rejects diagnosis, dosing, triage)
                  │ In Scope
                  ▼
         [ 2. Drug Entity Resolution ]
         (Normalises name via aliases.json)
                  │ Covered Drug (L001)
                  ▼
    ┌────────────────────────────────────────┐
    │       3. Retrieval Architecture        │
    │  - Dense Retrieval (ChromaDB top-5)    │
    │  - BM25 Retrieval (Lexical top-5)      │
    │  - Reciprocal Rank Fusion (RRF k=60)   │
    └──────────────────┬─────────────────────┘
                       │ Top-5 Candidate Passages
                       ▼
         [ 4. Local Evidence Gate ]
         (G1: Coverage | G2: Section | G3: Qualifier)
             /                           \
       Gate Failed                   Gate Passed
           /                               \
          ▼                                 ▼
   [ Early Abstention ]         [ 5. Guarded Generation ]
   (No Groq token spend)        (openai/gpt-oss-120b via Groq)
                                            │
                                            ▼
                                [ 6. Evidence Validation ]
                                (Character-exact quote matcher,
                                 schema validator, passage ID checker)
                                            │ Validated
                                            ▼
                               [ Formatted Grounded Response ]
```

---

## Features

- **Strict Source Grounding:** Relies strictly on the indexed DailyMed label text; prior medical knowledge and speculative extrapolation are forbidden.
- **2 × 2 Controlled Evaluation Matrix:** Evaluates four distinct architectural paradigms:
  - **Variant A:** Dense retrieval, unguarded.
  - **Variant B:** Hybrid retrieval (Dense + BM25 + RRF), unguarded.
  - **Variant C:** Dense retrieval + Local deterministic evidence gate.
  - **Variant D:** Hybrid retrieval + Local deterministic evidence gate.
- **Local Evidence Gate:** Multi-rule deterministic gate (G1 drug coverage, G2 label section relevance, G3 qualifier detection) that prevents ungrounded generation and saves token budget.
- **Exact Quotation Verification:** Enforces character-exact contiguous substring quotation matching against retrieved passages with automatic single-attempt retry on formatting discrepancies.
- **Controlled Benchmarking:** Automated, reproducible benchmark harness supporting frozen development (20 questions) and held-out (40 questions) test sets with SQLite persistence and resumable caching.
- **Dual Presentation Formats:** REST API endpoints via FastAPI and an interactive clinical search interface via Streamlit.

---

## Scope and Exclusions

- **In Scope:** Label-grounded queries regarding contraindications, warnings, precautions, drug interactions, and dosage guidelines for indexed drugs (currently Metformin Hydrochloride `L001`).
- **Excluded Requests:**
  - Individual patient diagnosis.
  - Patient-specific dose calculations or prescribing orders.
  - Emergency bedside medical resuscitation/triage.
  - Non-covered drugs without official DailyMed indexation.

---

## Data Provenance

| Label ID | Drug | Official Product Title | SPL Version | Effective Date | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L001** | metformin hydrochloride | METFORMIN HYDROCHLORIDE TABLET [REMEDYREPACK INC.] | 22 | 20261026 | DailyMed SPL XML |

- Verified set ID: `30658e19-8e6e-49aa-a007-e414224797c5`
- Document ID: `30658e19-8e6e-49aa-a007-e414224797c5_v22`
- Authoritative corpus: 11 passages in `data/passages/L001.jsonl`

---

## Setup & Environment

### Prerequisites

- Python 3.11
- Groq API Key (set in `.env`)

### Installation

```bash
# Clone repository and enter directory
cd clinical-rag

# Activate Python 3.11 virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration (`.env`)

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0.0
GROQ_MAX_TOKENS=2048
PROMPT_VERSION=v1
RETRIEVAL_TOP_K=5
```

---

## Execution Commands

### 1. Database & Ingestion

```bash
# Initialise SQLite database and create all 7 tables
python scripts/init_db.py

# Ingest authoritative L001 passages into persistent ChromaDB
python scripts/ingest.py
```

### 2. Testing

```bash
# Run complete test suite (89 tests, 0 Groq calls)
pytest -v
```

### 3. Application Services

```bash
# Start FastAPI backend server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Start Streamlit UI interface
streamlit run ui/streamlit_app.py
```

### 4. Benchmark Evaluations

```bash
# Evaluate Variant A on development benchmark (20 questions)
python scripts/evaluate.py --variant A

# Evaluate Variant B (Hybrid Dense+BM25)
python scripts/evaluate.py --variant B

# Evaluate Variant C (Dense + Local Evidence Gate)
python scripts/evaluate.py --variant C

# Evaluate Variant D (Hybrid + Local Evidence Gate)
python scripts/evaluate.py --variant D

# Evaluate Variant A on held-out benchmark (40 questions)
python scripts/evaluate.py --variant A --benchmark heldout
```

### 5. Figure Generation

```bash
# Generate all 8 development figures + held-out preliminary figure (PNG + PDF)
python scripts/generate_figures.py
```

---

## Result Interpretation

### Primary Development Comparison (20 Questions)

| Metric | Variant A (Dense) | Variant B (Hybrid) | Variant C (Dense+Gate) | Variant D (Hybrid+Gate) |
| :--- | :--- | :--- | :--- | :--- |
| **Run ID** | `run_A_f4abde8f0bf9` | `run_B_e87bda7469d5` | `run_C_641db18ad808` | `run_D_6e9b602d7417` |
| **Status match** | 17/20 (85.0%) | 19/20 (95.0%) | 17/20 (85.0%) | 18/20 (90.0%) |
| **Evidence hit@5** | 8/10 (80.0%) | 10/10 (100.0%) | 8/10 (80.0%) | 10/10 (100.0%) |
| **Answerable answered** | 8/10 (80.0%) | 10/10 (100.0%) | 8/10 (80.0%) | 9/10 (90.0%) |
| **Appropriate abstention** | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| **Appropriate scope refusal**| 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |
| **Over-refusal** | 2/10 (20.0%) | 0/10 (0.0%) | 2/10 (20.0%) | 1/10 (10.0%) |
| **Processing errors** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (5.0%) |
| **Mean latency** | 5.420s | 8.004s | 3.985s | 6.978s |
| **Total tokens consumed** | 32,833 | 37,385 | 28,618 (-12.8%) | 32,681 (-12.6%) |
| **Groq generation calls** | 16 | 16 | 14 | 14 |
| **Gate-avoided calls** | 0 | 0 | 2 | 0 |

1. **Hybrid Retrieval Gains:** Adding lexical BM25 with Reciprocal Rank Fusion ($k=60$) resolved term-matching misses on specific drug interactions, raising Evidence Hit@5 from 80% to 100%.
2. **Gate Efficiency:** The local deterministic evidence gate saved 4,215 tokens (-12.8%) and lowered mean latency by 26.5% (down to 3.985s) in dense retrieval configurations without degrading abstention accuracy.

---

## Known Limitations

Held-out evaluation was completed for Variant A only because of provider-side daily token-quota constraints. Therefore, held-out results are preliminary and are not used for cross-variant comparison. The primary four-variant comparison uses the frozen 20-question development benchmark.

- **Single Label Scope:** Evaluated exclusively on FDA DailyMed label `L001` (metformin hydrochloride). Queries spanning non-indexed concomitant drugs require clarification.
- **External Provider Dependency:** Rate limits and quota restrictions on third-party inference APIs necessitate enterprise provisioning for high-volume clinical deployments.

---

## Future Work

- Expansion of corpus to multi-label pharmacology classes (e.g. cardiovascular, endocrinology).
- Incorporation of semantic cross-encoder rerankers upstream of the local gate.
- Offline quantized medical LLM inference to eliminate external API rate limit dependencies.

---

## Disclaimer

This system is an academic prototype for source-grounded retrieval from DailyMed product inserts. It has not undergone clinical validation, clinical safety testing, or healthcare-professional validation. It must not be used for medical diagnosis, individualised dosing, or patient management. No claims of superiority over commercial systems, human practitioners, or alternative LLM architectures are made.
