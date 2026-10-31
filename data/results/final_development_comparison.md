# 2 × 2 Controlled Clinical RAG Benchmark Evaluation

- **Variant A**: `run_A_f4abde8f0bf9` (Dense retrieval, no gate)
- **Variant B**: `run_B_e87bda7469d5` (Dense + BM25 hybrid retrieval, no gate)
- **Variant C**: `run_C_641db18ad808` (Dense retrieval + local deterministic evidence gate)
- **Variant D**: `run_D_6e9b602d7417` (Dense + BM25 hybrid retrieval + local deterministic evidence gate)

| Metric | A (Dense) | B (Hybrid) | C (Dense+Gate) | D (Hybrid+Gate) |
| :--- | :--- | :--- | :--- | :--- |
| Status match | 17/20 (85.0%) | 19/20 (95.0%) | 17/20 (85.0%) | 18/20 (90.0%) |
| Evidence hit@5 | 8/10 (80.0%) | 10/10 (100.0%) | 8/10 (80.0%) | 10/10 (100.0%) |
| Required-evidence coverage | 8/10 (80.0%) | 10/10 (100.0%) | 8/10 (80.0%) | 10/10 (100.0%) |
| Answerable questions answered | 8/10 (80.0%) | 10/10 (100.0%) | 8/10 (80.0%) | 9/10 (90.0%) |
| Appropriate abstention | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| Appropriate clarification | 2/3 (66.7%) | 2/3 (66.7%) | 2/3 (66.7%) | 2/3 (66.7%) |
| Appropriate scope refusal | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |
| Over-refusal | 2/10 (20.0%) | 0/10 (0.0%) | 2/10 (20.0%) | 1/10 (10.0%) |
| Processing errors | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (5.0%) |
| Mean latency | 5.420s | 8.004s | 3.985s | 6.978s |
| p50 latency | 1.933s | 7.976s | 1.036s | 1.840s |
| p95 latency | 19.313s | 17.977s | 14.771s | 19.800s |
| Mean total tokens | 1641.7 | 1869.2 | 1430.9 | 1634.0 |
| Total tokens | 32,833 | 37,385 | 28,618 | 32,681 |
| Groq generation calls | 16 | 16 | 14 | 14 |
| Gate-avoided calls | 0 | 0 | 2 | 0 |
