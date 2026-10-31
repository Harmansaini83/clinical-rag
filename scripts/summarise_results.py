"""Summarise deterministic metrics and generate A/B/C/D comparison and error analysis."""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sqlmodel import Session, select
from app.db import get_engine
from app.models import BenchmarkQuestion, Result, Run
from app.services.metrics import compute_metrics_for_run


def format_pct(num: float) -> str:
    return f"{num * 100:.1f}%"


def generate_single_summary(run_id: str, engine=None):
    metrics = compute_metrics_for_run(run_id, engine=engine)
    print("=" * 80)
    print(f"METRIC SUMMARY FOR RUN: {metrics['run_id']} (Variant {metrics['variant']})")
    print("=" * 80)
    print(f"Total Questions:              {metrics['total_questions']}")
    print(f"Status Match:                 {metrics['status_matches']} / {metrics['total_questions']} ({format_pct(metrics['status_match_rate'])})")
    print(f"Evidence hit@5:               {metrics['evidence_hits_top5']} / {metrics['retrieval_eligible_questions']} ({format_pct(metrics['evidence_hit_rate'])})")
    print(f"Required-evidence coverage:   {metrics['required_evidence_covered']} / {metrics['retrieval_eligible_questions']} ({format_pct(metrics['required_evidence_coverage_rate'])})")
    print(f"Answer rate (answerable):     {metrics['answerable_answered']} / {metrics['answerable_questions']} ({format_pct(metrics['answer_rate'])})")
    print(f"Over-refusal rate:            {metrics['over_refusals']} / {metrics['answerable_questions']} ({format_pct(metrics['over_refusal_rate'])})")
    print(f"Appropriate abstention:       {metrics['actual_insufficient']} / {metrics['expected_insufficient']} ({format_pct(metrics['appropriate_abstention_rate'])})")
    print(f"Appropriate clarification:    {metrics['actual_clarify']} / {metrics['expected_clarify']} ({format_pct(metrics['appropriate_clarification_rate'])})")
    print(f"Appropriate scope refusal:    {metrics['actual_scope']} / {metrics['expected_scope']} ({format_pct(metrics['appropriate_scope_refusal_rate'])})")
    print(f"Processing errors:            {metrics['processing_errors']} ({format_pct(metrics['processing_error_rate'])})")
    print(f"Mean latency:                 {metrics['mean_latency']:.3f}s (p50: {metrics['p50_latency']:.3f}s, p95: {metrics['p95_latency']:.3f}s)")
    print(f"Total tokens:                 {metrics['total_tokens']:,} (mean: {metrics['mean_tokens']:.1f})")
    print(f"Groq calls:                   {metrics['groq_generation_calls']} (Gate avoided: {metrics['groq_calls_avoided_by_gate']})")
    print("=" * 80)


def generate_comparison_and_errors(run_a_id: str, run_c_id: str, engine=None):
    if engine is None:
        engine = get_engine()

    mA = compute_metrics_for_run(run_a_id, engine=engine)
    mC = compute_metrics_for_run(run_c_id, engine=engine)

    comp_rows = [
        ("Status match", f"{mA['status_matches']} / {mA['total_questions']} ({format_pct(mA['status_match_rate'])})", f"{mC['status_matches']} / {mC['total_questions']} ({format_pct(mC['status_match_rate'])})", f"{format_pct(mC['status_match_rate'] - mA['status_match_rate'])}"),
        ("Evidence hit@5", f"{mA['evidence_hits_top5']} / {mA['retrieval_eligible_questions']} ({format_pct(mA['evidence_hit_rate'])})", f"{mC['evidence_hits_top5']} / {mC['retrieval_eligible_questions']} ({format_pct(mC['evidence_hit_rate'])})", f"{format_pct(mC['evidence_hit_rate'] - mA['evidence_hit_rate'])}"),
        ("Required-evidence coverage", f"{mA['required_evidence_covered']} / {mA['retrieval_eligible_questions']} ({format_pct(mA['required_evidence_coverage_rate'])})", f"{mC['required_evidence_covered']} / {mC['retrieval_eligible_questions']} ({format_pct(mC['required_evidence_coverage_rate'])})", f"{format_pct(mC['required_evidence_coverage_rate'] - mA['required_evidence_coverage_rate'])}"),
        ("Answerable questions answered", f"{mA['answerable_answered']} / {mA['answerable_questions']} ({format_pct(mA['answer_rate'])})", f"{mC['answerable_answered']} / {mC['answerable_questions']} ({format_pct(mC['answer_rate'])})", f"{format_pct(mC['answer_rate'] - mA['answer_rate'])}"),
        ("Appropriate abstention", f"{mA['actual_insufficient']} / {mA['expected_insufficient']} ({format_pct(mA['appropriate_abstention_rate'])})", f"{mC['actual_insufficient']} / {mC['expected_insufficient']} ({format_pct(mC['appropriate_abstention_rate'])})", f"{format_pct(mC['appropriate_abstention_rate'] - mA['appropriate_abstention_rate'])}"),
        ("Appropriate clarification", f"{mA['actual_clarify']} / {mA['expected_clarify']} ({format_pct(mA['appropriate_clarification_rate'])})", f"{mC['actual_clarify']} / {mC['expected_clarify']} ({format_pct(mC['appropriate_clarification_rate'])})", f"{format_pct(mC['appropriate_clarification_rate'] - mA['appropriate_clarification_rate'])}"),
        ("Appropriate scope refusal", f"{mA['actual_scope']} / {mA['expected_scope']} ({format_pct(mA['appropriate_scope_refusal_rate'])})", f"{mC['actual_scope']} / {mC['expected_scope']} ({format_pct(mC['appropriate_scope_refusal_rate'])})", f"{format_pct(mC['appropriate_scope_refusal_rate'] - mA['appropriate_scope_refusal_rate'])}"),
        ("Over-refusal", f"{mA['over_refusals']} / {mA['answerable_questions']} ({format_pct(mA['over_refusal_rate'])})", f"{mC['over_refusals']} / {mC['answerable_questions']} ({format_pct(mC['over_refusal_rate'])})", f"{format_pct(mC['over_refusal_rate'] - mA['over_refusal_rate'])}"),
        ("Processing errors", f"{mA['processing_errors']} ({format_pct(mA['processing_error_rate'])})", f"{mC['processing_errors']} ({format_pct(mC['processing_error_rate'])})", f"{format_pct(mC['processing_error_rate'] - mA['processing_error_rate'])}"),
        ("Mean latency", f"{mA['mean_latency']:.3f}s", f"{mC['mean_latency']:.3f}s", f"{mC['mean_latency'] - mA['mean_latency']:+.3f}s"),
        ("p50 latency", f"{mA['p50_latency']:.3f}s", f"{mC['p50_latency']:.3f}s", f"{mC['p50_latency'] - mA['p50_latency']:+.3f}s"),
        ("p95 latency", f"{mA['p95_latency']:.3f}s", f"{mC['p95_latency']:.3f}s", f"{mC['p95_latency'] - mA['p95_latency']:+.3f}s"),
        ("Mean total tokens", f"{mA['mean_tokens']:.1f}", f"{mC['mean_tokens']:.1f}", f"{mC['mean_tokens'] - mA['mean_tokens']:+.1f}"),
        ("Total tokens", f"{mA['total_tokens']:,}", f"{mC['total_tokens']:,}", f"{mC['total_tokens'] - mA['total_tokens']:+,}"),
        ("Groq generation calls", f"{mA['groq_generation_calls']}", f"{mC['groq_generation_calls']}", f"{mC['groq_generation_calls'] - mA['groq_generation_calls']:+d}"),
        ("Gate avoided calls", "0 (no gate)", f"{mC['groq_calls_avoided_by_gate']}", f"+{mC['groq_calls_avoided_by_gate']}"),
    ]

    print("\n" + "=" * 90)
    print(f"BENCHMARK COMPARISON: Variant A ({run_a_id}) vs Variant C ({run_c_id})")
    print("=" * 90)
    print(f"{'Metric':<32} | {'Variant A':<20} | {'Variant C':<20} | {'Difference':<12}")
    print("-" * 90)
    for metric, va, vc, diff in comp_rows:
        print(f"{metric:<32} | {va:<20} | {vc:<20} | {diff:<12}")
    print("=" * 90)

    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "summary_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Variant A", "Variant C", "Difference"])
        for r in comp_rows:
            writer.writerow(r)

    md_path = results_dir / "summary_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Controlled Clinical RAG Benchmark: Variant A vs Variant C\n\n")
        f.write(f"- **Variant A Run ID**: `{run_a_id}` (Dense Retrieval + Guarded Generation)\n")
        f.write(f"- **Variant C Run ID**: `{run_c_id}` (Dense Retrieval + Local Deterministic Evidence Gate + Guarded Generation)\n")
        f.write(f"- **Model**: openai/gpt-oss-120b via Groq\n\n")
        f.write("| Metric | Variant A | Variant C | Difference |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for metric, va, vc, diff in comp_rows:
            f.write(f"| {metric} | {va} | {vc} | {diff} |\n")


def generate_four_way_comparison(runs_by_variant: Dict[str, str], engine=None):
    """Generate 4-way comparison for Variants A, B, C, and D."""
    if engine is None:
        engine = get_engine()

    metrics = {}
    for var in ["A", "B", "C", "D"]:
        rid = runs_by_variant[var]
        metrics[var] = compute_metrics_for_run(rid, engine=engine)

    mA, mB, mC, mD = metrics["A"], metrics["B"], metrics["C"], metrics["D"]

    comp_rows = [
        ("Status match",
         f"{mA['status_matches']}/{mA['total_questions']} ({format_pct(mA['status_match_rate'])})",
         f"{mB['status_matches']}/{mB['total_questions']} ({format_pct(mB['status_match_rate'])})",
         f"{mC['status_matches']}/{mC['total_questions']} ({format_pct(mC['status_match_rate'])})",
         f"{mD['status_matches']}/{mD['total_questions']} ({format_pct(mD['status_match_rate'])})"),
        ("Evidence hit@5",
         f"{mA['evidence_hits_top5']}/{mA['retrieval_eligible_questions']} ({format_pct(mA['evidence_hit_rate'])})",
         f"{mB['evidence_hits_top5']}/{mB['retrieval_eligible_questions']} ({format_pct(mB['evidence_hit_rate'])})",
         f"{mC['evidence_hits_top5']}/{mC['retrieval_eligible_questions']} ({format_pct(mC['evidence_hit_rate'])})",
         f"{mD['evidence_hits_top5']}/{mD['retrieval_eligible_questions']} ({format_pct(mD['evidence_hit_rate'])})"),
        ("Required-evidence coverage",
         f"{mA['required_evidence_covered']}/{mA['retrieval_eligible_questions']} ({format_pct(mA['required_evidence_coverage_rate'])})",
         f"{mB['required_evidence_covered']}/{mB['retrieval_eligible_questions']} ({format_pct(mB['required_evidence_coverage_rate'])})",
         f"{mC['required_evidence_covered']}/{mC['retrieval_eligible_questions']} ({format_pct(mC['required_evidence_coverage_rate'])})",
         f"{mD['required_evidence_covered']}/{mD['retrieval_eligible_questions']} ({format_pct(mD['required_evidence_coverage_rate'])})"),
        ("Answerable questions answered",
         f"{mA['answerable_answered']}/{mA['answerable_questions']} ({format_pct(mA['answer_rate'])})",
         f"{mB['answerable_answered']}/{mB['answerable_questions']} ({format_pct(mB['answer_rate'])})",
         f"{mC['answerable_answered']}/{mC['answerable_questions']} ({format_pct(mC['answer_rate'])})",
         f"{mD['answerable_answered']}/{mD['answerable_questions']} ({format_pct(mD['answer_rate'])})"),
        ("Appropriate abstention",
         f"{mA['actual_insufficient']}/{mA['expected_insufficient']} ({format_pct(mA['appropriate_abstention_rate'])})",
         f"{mB['actual_insufficient']}/{mB['expected_insufficient']} ({format_pct(mB['appropriate_abstention_rate'])})",
         f"{mC['actual_insufficient']}/{mC['expected_insufficient']} ({format_pct(mC['appropriate_abstention_rate'])})",
         f"{mD['actual_insufficient']}/{mD['expected_insufficient']} ({format_pct(mD['appropriate_abstention_rate'])})"),
        ("Appropriate clarification",
         f"{mA['actual_clarify']}/{mA['expected_clarify']} ({format_pct(mA['appropriate_clarification_rate'])})",
         f"{mB['actual_clarify']}/{mB['expected_clarify']} ({format_pct(mB['appropriate_clarification_rate'])})",
         f"{mC['actual_clarify']}/{mC['expected_clarify']} ({format_pct(mC['appropriate_clarification_rate'])})",
         f"{mD['actual_clarify']}/{mD['expected_clarify']} ({format_pct(mD['appropriate_clarification_rate'])})"),
        ("Appropriate scope refusal",
         f"{mA['actual_scope']}/{mA['expected_scope']} ({format_pct(mA['appropriate_scope_refusal_rate'])})",
         f"{mB['actual_scope']}/{mB['expected_scope']} ({format_pct(mB['appropriate_scope_refusal_rate'])})",
         f"{mC['actual_scope']}/{mC['expected_scope']} ({format_pct(mC['appropriate_scope_refusal_rate'])})",
         f"{mD['actual_scope']}/{mD['expected_scope']} ({format_pct(mD['appropriate_scope_refusal_rate'])})"),
        ("Over-refusal",
         f"{mA['over_refusals']}/{mA['answerable_questions']} ({format_pct(mA['over_refusal_rate'])})",
         f"{mB['over_refusals']}/{mB['answerable_questions']} ({format_pct(mB['over_refusal_rate'])})",
         f"{mC['over_refusals']}/{mC['answerable_questions']} ({format_pct(mC['over_refusal_rate'])})",
         f"{mD['over_refusals']}/{mD['answerable_questions']} ({format_pct(mD['over_refusal_rate'])})"),
        ("Processing errors",
         f"{mA['processing_errors']} ({format_pct(mA['processing_error_rate'])})",
         f"{mB['processing_errors']} ({format_pct(mB['processing_error_rate'])})",
         f"{mC['processing_errors']} ({format_pct(mC['processing_error_rate'])})",
         f"{mD['processing_errors']} ({format_pct(mD['processing_error_rate'])})"),
        ("Mean latency",
         f"{mA['mean_latency']:.3f}s", f"{mB['mean_latency']:.3f}s", f"{mC['mean_latency']:.3f}s", f"{mD['mean_latency']:.3f}s"),
        ("p50 latency",
         f"{mA['p50_latency']:.3f}s", f"{mB['p50_latency']:.3f}s", f"{mC['p50_latency']:.3f}s", f"{mD['p50_latency']:.3f}s"),
        ("p95 latency",
         f"{mA['p95_latency']:.3f}s", f"{mB['p95_latency']:.3f}s", f"{mC['p95_latency']:.3f}s", f"{mD['p95_latency']:.3f}s"),
        ("Mean total tokens",
         f"{mA['mean_tokens']:.1f}", f"{mB['mean_tokens']:.1f}", f"{mC['mean_tokens']:.1f}", f"{mD['mean_tokens']:.1f}"),
        ("Total tokens",
         f"{mA['total_tokens']:,}", f"{mB['total_tokens']:,}", f"{mC['total_tokens']:,}", f"{mD['total_tokens']:,}"),
        ("Groq generation calls",
         f"{mA['groq_generation_calls']}", f"{mB['groq_generation_calls']}", f"{mC['groq_generation_calls']}", f"{mD['groq_generation_calls']}"),
        ("Gate-avoided calls",
         "0", "0", f"{mC['groq_calls_avoided_by_gate']}", f"{mD['groq_calls_avoided_by_gate']}"),
    ]

    print("\n" + "=" * 110)
    print(f"2 × 2 BENCHMARK COMPARISON: VARIANTS A, B, C, AND D")
    print("=" * 110)
    print(f"{'Metric':<30} | {'A (Dense)':<18} | {'B (Hybrid)':<18} | {'C (Dense+Gate)':<18} | {'D (Hybrid+Gate)':<18}")
    print("-" * 110)
    for m, a, b, c, d in comp_rows:
        print(f"{m:<30} | {a:<18} | {b:<18} | {c:<18} | {d:<18}")
    print("=" * 110)

    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_all = results_dir / "summary_comparison_all.csv"
    with open(csv_all, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "A (Dense)", "B (Hybrid)", "C (Dense+Gate)", "D (Hybrid+Gate)"])
        for r in comp_rows:
            writer.writerow(r)

    # Save Markdown
    md_all = results_dir / "summary_comparison_all.md"
    with open(md_all, "w", encoding="utf-8") as f:
        f.write("# 2 × 2 Controlled Clinical RAG Benchmark Evaluation\n\n")
        f.write(f"- **Variant A**: `{runs_by_variant['A']}` (Dense retrieval, no gate)\n")
        f.write(f"- **Variant B**: `{runs_by_variant['B']}` (Dense + BM25 hybrid retrieval, no gate)\n")
        f.write(f"- **Variant C**: `{runs_by_variant['C']}` (Dense retrieval + local deterministic evidence gate)\n")
        f.write(f"- **Variant D**: `{runs_by_variant['D']}` (Dense + BM25 hybrid retrieval + local deterministic evidence gate)\n\n")
        f.write("| Metric | A (Dense) | B (Hybrid) | C (Dense+Gate) | D (Hybrid+Gate) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for m, a, b, c, d in comp_rows:
            f.write(f"| {m} | {a} | {b} | {c} | {d} |\n")

    # Error Analysis across all four variants
    with Session(engine) as session:
        questions = session.exec(select(BenchmarkQuestion)).all()
        q_map = {q.question_id: q for q in questions}

    error_rows_all = []
    for var in ["A", "B", "C", "D"]:
        rid = runs_by_variant[var]
        with Session(engine) as session:
            res_list = session.exec(select(Result).where(Result.run_id == rid)).all()

        for r in res_list:
            q = q_map.get(r.question_id)
            if not q or r.status == q.expected_action:
                continue

            gold_ids = json.loads(q.gold_passage_ids_json or "[]")
            retrieved_ids = json.loads(r.retrieved_ids_json or "[]")
            has_hit = any(gid in retrieved_ids for gid in gold_ids) if gold_ids else True

            if r.status == "processing_error":
                cat = "processing_error"
                exp = f"Validation or runtime exception: {r.error}"
            elif q.expected_action == "answer" and r.status == "insufficient_evidence":
                if not has_hit:
                    cat = "retrieval_miss"
                    exp = f"Gold passage(s) {gold_ids} not found in retrieved top-5 {retrieved_ids}"
                else:
                    cat = "over_refusal"
                    exp = f"Passage was retrieved with evidence, but model or gate abstained."
            elif q.expected_action == "clarify" and r.status == "insufficient_evidence":
                cat = "missed_clarification"
                exp = "Question required clarification (multi-drug query) but was routed as insufficient evidence."
            elif q.expected_action == "insufficient_evidence" and r.status == "answer":
                cat = "under_refusal"
                exp = "Model generated answer despite missing corpus evidence."
            elif q.expected_action == "out_of_scope" and r.status != "out_of_scope":
                cat = "incorrect_scope_refusal"
                exp = "Clinical scope policy failed to filter question."
            else:
                cat = "other"
                exp = f"Status mismatch: expected {q.expected_action}, got {r.status}."

            error_rows_all.append({
                "run_id": rid,
                "variant": var,
                "question_id": r.question_id,
                "category": q.category,
                "expected_action": q.expected_action,
                "actual_status": r.status,
                "error_category": cat,
                "explanation": exp,
            })

    # Save All Error CSV
    err_csv_all = results_dir / "error_analysis_all.csv"
    with open(err_csv_all, "w", newline="", encoding="utf-8") as f:
        fields = ["run_id", "variant", "question_id", "category", "expected_action", "actual_status", "error_category", "explanation"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(error_rows_all)

    # Save All Error Markdown
    err_md_all = results_dir / "error_analysis_all.md"
    with open(err_md_all, "w", encoding="utf-8") as f:
        f.write("# 4-Way Controlled Benchmark Error Analysis\n\n")
        f.write(f"Total Discrepancies across all four variants: {len(error_rows_all)}\n\n")
        f.write("| Run ID | Variant | Question ID | Expected | Actual | Error Category | Explanation |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for e in error_rows_all:
            f.write(f"| `{e['run_id']}` | {e['variant']} | {e['question_id']} | {e['expected_action']} | {e['actual_status']} | `{e['error_category']}` | {e['explanation']} |\n")

    print(f"\nSaved 4-way comparison to: {csv_all} and {md_all}")
    print(f"Saved 4-way error analysis to: {err_csv_all} and {err_md_all}")


def main():
    parser = argparse.ArgumentParser(description="Summarise results and compare evaluation runs.")
    parser.add_argument("--run-id", type=str, help="Specific Run ID to summarise.")
    parser.add_argument("--latest", action="store_true", help="Summarise latest runs.")
    parser.add_argument("--compare", action="store_true", help="Generate A/C comparison report.")
    parser.add_argument("--compare-all", action="store_true", help="Generate 4-way comparison report for A, B, C, and D.")

    args = parser.parse_args()
    engine = get_engine()

    if args.run_id:
        generate_single_summary(args.run_id, engine=engine)
    elif args.compare_all:
        with Session(engine) as session:
            runs_by_var = {}
            for v in ["A", "B", "C", "D"]:
                latest = session.exec(select(Run).where(Run.variant == v).order_by(Run.created_at.desc())).first()
                if not latest:
                    print(f"Missing run for Variant {v}. All 4 variants are required for --compare-all.")
                    sys.exit(1)
                runs_by_var[v] = latest.run_id

        for v in ["A", "B", "C", "D"]:
            generate_single_summary(runs_by_var[v], engine=engine)
        generate_four_way_comparison(runs_by_var, engine=engine)

    elif args.compare or args.latest:
        with Session(engine) as session:
            latest_a = session.exec(select(Run).where(Run.variant == "A").order_by(Run.created_at.desc())).first()
            latest_c = session.exec(select(Run).where(Run.variant == "C").order_by(Run.created_at.desc())).first()

        if not latest_a or not latest_c:
            print("Both Variant A and Variant C runs are required for comparison.")
            sys.exit(1)

        generate_single_summary(latest_a.run_id, engine=engine)
        generate_single_summary(latest_c.run_id, engine=engine)
        generate_comparison_and_errors(latest_a.run_id, latest_c.run_id, engine=engine)
    else:
        print("Must specify --run-id, --latest, --compare, or --compare-all")
        sys.exit(1)


if __name__ == "__main__":
    main()
