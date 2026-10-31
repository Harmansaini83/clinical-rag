# Controlled Benchmark Error Analysis

Total Discrepancies: 6

| Run ID | Variant | Question ID | Expected | Actual | Error Category | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `run_A_f4abde8f0bf9` | A | DEV-Q006 | answer | insufficient_evidence | `retrieval_miss` | Gold passage(s) ['L001-5.1-p02'] not found in retrieved top-5 ['L001-4-p01', 'L001-8.6-p01', 'L001-5.4-p01', 'L001-2.3-p01', 'L001-5.1-p01'] |
| `run_A_f4abde8f0bf9` | A | DEV-Q010 | answer | insufficient_evidence | `retrieval_miss` | Gold passage(s) ['L001-5.1-p03'] not found in retrieved top-5 ['L001-4-p01', 'L001-8.6-p01', 'L001-5.1-p01', 'L001-5.4-p01', 'L001-5.1-p02'] |
| `run_A_f4abde8f0bf9` | A | DEV-Q016 | clarify | insufficient_evidence | `missed_clarification` | Question required clarification (e.g. multi-drug query) but was routed as insufficient evidence. |
| `run_C_641db18ad808` | C | DEV-Q006 | answer | insufficient_evidence | `retrieval_miss` | Gold passage(s) ['L001-5.1-p02'] not found in retrieved top-5 ['L001-4-p01', 'L001-8.6-p01', 'L001-5.4-p01', 'L001-2.3-p01', 'L001-5.1-p01'] |
| `run_C_641db18ad808` | C | DEV-Q010 | answer | insufficient_evidence | `retrieval_miss` | Gold passage(s) ['L001-5.1-p03'] not found in retrieved top-5 ['L001-4-p01', 'L001-8.6-p01', 'L001-5.1-p01', 'L001-5.4-p01', 'L001-5.1-p02'] |
| `run_C_641db18ad808` | C | DEV-Q016 | clarify | insufficient_evidence | `missed_clarification` | Question required clarification (e.g. multi-drug query) but was routed as insufficient evidence. |
