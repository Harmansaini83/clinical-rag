"""CLI runner for RAG benchmark evaluation (Variant A and Variant C).

Usage:
  python scripts/evaluate.py --variant A --limit 3
  python scripts/evaluate.py --variant C --limit 3
  python scripts/evaluate.py --variant A
  python scripts/evaluate.py --variant C
  python scripts/evaluate.py --variant A --question-id DEV-Q001
  python scripts/evaluate.py --variant C --question-id DEV-Q001
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.config import settings
from app.db import create_db_and_tables, get_engine
from app.services.evaluation import evaluate_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled clinical RAG benchmark evaluation."
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["A", "B", "C", "D"],
        help="Evaluation variant: 'A' (dense guarded), 'B' (hybrid guarded), 'C' (dense + gate + guarded), 'D' (hybrid + gate + guarded).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="dev",
        choices=["dev", "heldout"],
        help="Benchmark set to evaluate: 'dev' (dev_questions.jsonl) or 'heldout' (heldout_questions.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of benchmark questions to evaluate.",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Target specific benchmark question ID (e.g. DEV-Q001).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Groq LLM response caching.",
    )

    args = parser.parse_args()

    engine = get_engine()
    create_db_and_tables(engine)

    bench_path = (
        settings.data_dir / "benchmark" / "heldout_questions.jsonl"
        if args.benchmark == "heldout"
        else settings.data_dir / "benchmark" / "dev_questions.jsonl"
    )

    print("=" * 80)
    print("CLINICAL AI ASSISTANT — CONTROLLED BENCHMARK EVALUATION")
    print(f"Variant: {args.variant} | Benchmark: {args.benchmark} | Model: {settings.groq_model} | Caching: {not args.no_cache}")
    if args.question_id:
        print(f"Target Question: {args.question_id}")
    elif args.limit:
        print(f"Execution Limit: {args.limit} question(s)")
    print("=" * 80)

    try:
        summary = evaluate_benchmark(
            variant=args.variant,
            engine=engine,
            limit=args.limit,
            target_question_id=args.question_id,
            benchmark_path=bench_path,
            use_cache=not args.no_cache,
        )
    except Exception as e:
        print(f"FATAL ERROR during evaluation: {e}")
        sys.exit(1)

    print("\n" + "-" * 80)
    print(f"RUN SUMMARY: {summary['run_id']}")
    print("-" * 80)
    print(f"Variant:                {summary['variant']}")
    print(f"Total Selected:         {summary['total']}")
    print(f"Executed:               {summary['executed']}")
    print(f"Skipped (Resumed):      {summary['skipped']}")
    print(f"Status Matches:         {summary['passed_status_match']} / {summary['total']}")
    print(f"Processing Errors:      {summary['errors']}")
    print("-" * 80)
    print(f"{'QUESTION ID':<14} | {'ACTUAL STATUS':<22} | {'EXPECTED':<22} | {'MATCH':<6} | {'LATENCY':<8}")
    print("-" * 80)

    for item in summary["results"]:
        match_str = "YES" if item["status_match"] else "NO"
        cached_str = " (resumed)" if item.get("cached_or_skipped") else ""
        print(
            f"{item['question_id']:<14} | {item['status']:<22} | {item['expected']:<22} | "
            f"{match_str:<6} | {item['latency_seconds']:.3f}s{cached_str}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
