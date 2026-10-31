# PROJECT STATUS

## Current Status: FUNCTIONALLY COMPLETE

```text
PROJECT STATUS: FUNCTIONALLY COMPLETE
Completed:
- Terminal RAG workflow.
- FastAPI backend.
- SQLite database.
- Streamlit UI.
- ChromaDB retrieval.
- Groq generation.
- Evidence validation.
- Deterministic evidence gate.
- Hybrid retrieval.
- Development benchmark evaluation for Variants A, B, C, and D.
- Held-out benchmark creation.
- Held-out Variant A preliminary evaluation.
- Matplotlib visualisations.
- Automated tests.
Not completed:
- Held-out evaluation for Variants B, C, and D.
- Reason: Groq daily token quota constraints.
- Decision: skipped to avoid additional API usage.
- Impact: held-out results are preliminary and are not used for cross-variant comparison.
```

---

## 1. Verified Corpus State

- **Label ID**: `L001`
- **Product Name**: METFORMIN HYDROCHLORIDE TABLET [REMEDYREPACK INC.]
- **Ingredient**: metformin hydrochloride
- **Document ID**: `30658e19-8e6e-49aa-a007-e414224797c5_v22`
- **Set ID**: `30658e19-8e6e-49aa-a007-e414224797c5`
- **SPL Version**: `22`
- **Effective Date**: `20261026`
- **Passage Count**: 11 authoritative passages in `data/passages/L001.jsonl`
- **Passage IDs**:
  - `L001-2.3-p01` (renal_impairment)
  - `L001-4-p01` (contraindications)
  - `L001-5.1-p01` (warnings_precautions)
  - `L001-5.1-p02` (warnings_precautions)
  - `L001-5.1-p03` (warnings_precautions)
  - `L001-5.2-p01` (warnings_precautions)
  - `L001-5.3-p01` (warnings_precautions)
  - `L001-5.4-p01` (warnings_precautions)
  - `L001-7-p01` (drug_interactions)
  - `L001-7-p02` (drug_interactions)
  - `L001-8.6-p01` (renal_impairment)

---

## 2. Experimental Benchmark Matrix

### Primary 4-Variant Comparison (Frozen 20-Question Development Benchmark)

| Variant | Run ID | Retrieval | Evidence Gate | Status Match | Evidence Hit@5 | Mean Latency | Total Tokens |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Variant A** | `run_A_f4abde8f0bf9` | Dense | None | 17/20 (85.0%) | 8/10 (80.0%) | 5.420s | 32,833 |
| **Variant B** | `run_B_e87bda7469d5` | Hybrid (BM25+Dense) | None | 19/20 (95.0%) | 10/10 (100.0%) | 8.004s | 37,385 |
| **Variant C** | `run_C_641db18ad808` | Dense | Local Gate | 17/20 (85.0%) | 8/10 (80.0%) | 3.985s | 28,618 |
| **Variant D** | `run_D_6e9b602d7417` | Hybrid (BM25+Dense) | Local Gate | 18/20 (90.0%) | 10/10 (100.0%) | 6.978s | 32,681 |

### Preliminary Generalisation Check (40-Question Held-Out Benchmark)

- **Variant A**: `run_A_29dae73de52f` — 32/40 (80.0%) status match, 0 processing errors, mean latency 9.07s.
- Variants B, C, and D were not evaluated on the held-out benchmark due to provider daily token quota constraints.

---

## 3. Matplotlib Visualisations (`data/results/figures/`)

All 8 development figures and 1 held-out preliminary figure exist as both PNG (300 DPI) and PDF:
1. `accuracy_comparison` (.png, .pdf)
2. `safety_behaviour` (.png, .pdf)
3. `retrieval_quality` (.png, .pdf)
4. `efficiency_comparison` (.png, .pdf)
5. `latency_distribution` (.png, .pdf)
6. `error_analysis` (.png, .pdf)
7. `gate_call_avoidance` (.png, .pdf)
8. `retrieval_source_contribution` (.png, .pdf)
9. `heldout_variant_a_preliminary` (.png, .pdf)
