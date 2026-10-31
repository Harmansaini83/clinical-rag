# Held-Out Benchmark: Variant A Preliminary Evaluation

- **Benchmark Split**: `heldout` (`data/benchmark/heldout_questions.jsonl`)
- **Run ID**: `run_A_29dae73de52f`
- **Total Questions Evaluated**: 40
- **Generator Model**: `openai/gpt-oss-120b` via Groq

> **Notice:** Preliminary generalisation check only; Variants B, C, and D were not evaluated on the held-out set due to provider-side daily token-quota constraints. The primary four-variant comparison uses the frozen 20-question development benchmark.

| Metric | Variant A (Held-Out Preliminary) |
| :--- | :--- |
| **Questions evaluated** | 40 |
| **Status match** | 32/40 (80.0%) |
| **Processing errors** | 0/40 (0.0%) |
| **Appropriate abstention** | 8/8 (100.0%) |
| **Appropriate scope refusal** | 4/4 (100.0%) |
| **Mean latency** | 9.07s |
| **p50 latency** | 6.96s |
| **p95 latency** | 21.03s |
| **Total tokens consumed** | 64,174 |
| **Groq generation calls** | 33 |
