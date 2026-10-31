# Benchmark Result Tables

This document compiles the quantitative evaluation results for the 2 × 2 experimental matrix across retrieval methods (dense vs. hybrid BM25 + dense with Reciprocal Rank Fusion) and evidence gating strategies (no gate vs. local deterministic evidence gate).

## Development Benchmark Comparison (20 Questions)

Corpus: Metformin hydrochloride (`L001`, DailyMed SPL v22, Set ID: `30658e19-8e6e-49aa-a007-e414224797c5`).  
Generator: `openai/gpt-oss-120b` via Groq.

| Metric | Variant A (Dense, No Gate) | Variant B (Hybrid, No Gate) | Variant C (Dense + Gate) | Variant D (Hybrid + Gate) |
| :--- | :--- | :--- | :--- | :--- |
| **Run ID** | `run_A_f4abde8f0bf9` | `run_B_e87bda7469d5` | `run_C_641db18ad808` | `run_D_6e9b602d7417` |
| **Retrieval Architecture** | Dense (top-5) | Hybrid Dense + BM25 (RRF k=60) | Dense (top-5) | Hybrid Dense + BM25 (RRF k=60) |
| **Evidence Gate** | None | None | Local Deterministic (G1-G3) | Local Deterministic (G1-G3) |
| **Status Match** | 17/20 (85.0%) | 19/20 (95.0%) | 17/20 (85.0%) | 19/20 (95.0%) |
| **Evidence Hit@5** | 8/10 (80.0%) | 10/10 (100.0%) | 8/10 (80.0%) | 10/10 (100.0%) |
| **Answerable Answered** | 8/10 (80.0%) | 10/10 (100.0%) | 8/10 (80.0%) | 9/10 (90.0%) |
| **Appropriate Abstention**| 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| **Over-Refusal Rate** | 2/10 (20.0%) | 0/10 (0.0%) | 2/10 (20.0%) | 1/10 (10.0%) |
| **Processing Errors** | 0/20 (0.0%) | 0/20 (0.0%) | 0/20 (0.0%) | 0/20 (0.0%)* |
| **Groq Generator Calls** | 16 | 18 | 14 | 16 |
| **Total Tokens Consumed** | 32,833 | 37,385 | 28,618 | 32,681 |
| **Mean Latency (s)** | 5.420s | 8.004s | 3.985s | 6.978s |

*\*Note: DEV-Q005 in Variant D was repaired via contiguous quotation instruction sharpening and 1-attempt validation retry logic.*

---

## Held-Out Benchmark Results (40 Questions)

Benchmark: `data/benchmark/heldout_questions.jsonl` (20 answerable, 8 insufficient evidence, 5 clarify, 4 out of scope, 3 incorrect premise).  
Corpus: Authoritative 11 L001 passages.

| Variant | Run ID | Status Match (%) | Processing Errors | Mean Latency (s) | Total Questions Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Variant A** | `run_A_29dae73de52f` | 32/40 (80.0%) | 0 | 9.07s | 40 |
