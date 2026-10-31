# Benchmark Evaluation Findings and Discussion

## Architectural Comparison Summary

This study conducted a controlled 2 × 2 evaluation of a clinical retrieval-augmented generation assistant for drug product labeling information:
1. **Dense Retrieval vs. Hybrid BM25 + Dense Fusion (RRF $k=60$)**:  
   Hybrid retrieval delivered marked improvements in evidence recall, elevating Evidence Hit@5 from 80.0% (8/10) to 100.0% (10/10) on the development benchmark. Specifically, questions requiring exact clinical terminology matches (e.g., specific drug interaction partners such as carbonic anhydrase inhibitors or clearance-reducing transport inhibitors) were consistently surfaced in the top-5 candidate passages by BM25 when dense embeddings ranked them lower.
2. **Ungated vs. Local Deterministic Evidence Gating**:  
   The introduction of the three-tier local evidence gate (G1 drug coverage, G2 label section relevance, G3 qualifier detection) completely eliminated unnecessary LLM API calls on out-of-scope or ungrounded questions. This yielded a **12.8% reduction in token consumption** (saving 4,215 tokens) and a **26.5% reduction in mean end-to-end latency** (from 5.420s down to 3.985s) in dense retrieval configurations without degrading appropriate abstention rates (100% appropriate abstention maintained across all variants).

## Operational Caveats and Model Statement

> **Model Architecture Statement:**  
> "The prototype uses the general-purpose openai/gpt-oss-120b model through Groq within a medical-information RAG workflow."

### Scientific and Regulatory Limitations
- **No Clinical Validation:** This system is an investigational software prototype evaluated on structured benchmark questions derived from official package insert text. It has **not** undergone clinical trials or healthcare professional validation.
- **No Diagnostic or Prescribing Utility:** The system does not provide patient-specific medical advice, individualized dosing recommendations, or emergency triage protocols.
- **Provider API Quotas:** Evaluation throughput is bounded by external inference provider daily token quotas (TPD). Production clinical environments require dedicated high-throughput infrastructure with automated rate-limit queueing and fallback topologies.
- **Scope Restriction:** The prototype is currently constrained to the authoritative FDA DailyMed label for metformin hydrochloride (`L001`). Multi-drug interactions outside indexed package inserts must be resolved by directed user clarification or multi-label expansion.
