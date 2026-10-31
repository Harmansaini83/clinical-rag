"""CLI script to generate benchmark evaluation figures.

Usage:
  python scripts/generate_figures.py
  python scripts/generate_figures.py --benchmark development
  python scripts/generate_figures.py --benchmark heldout
  python scripts/generate_figures.py --latest
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.visualisation import (
    plot_accuracy_comparison,
    plot_safety_behaviour,
    plot_retrieval_quality,
    plot_efficiency_comparison,
    plot_latency_distribution,
    plot_error_analysis,
    plot_gate_call_avoidance,
    plot_retrieval_source_contribution,
    plot_heldout_variant_a_preliminary,
    generate_all_figures,
    get_figures_dir,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready Matplotlib figures for RAG benchmark results."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all", "development", "heldout"],
        help="Which benchmark figures to produce: 'development', 'heldout', or 'all' (default).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Regenerate all figures using latest database records.",
    )
    args = parser.parse_args()

    out_dir = get_figures_dir()
    print("=" * 70)
    print("CLINICAL AI ASSISTANT — FIGURE GENERATION")
    print(f"Target: {args.benchmark} | Output Directory: {out_dir}")
    print("=" * 70)

    if args.benchmark in ("all", "development") or args.latest:
        print("Generating Development Benchmark Figures (A, B, C, D)...")
        plot_accuracy_comparison(out_dir)
        print(" [OK] accuracy_comparison (PNG + PDF)")
        plot_safety_behaviour(out_dir)
        print(" [OK] safety_behaviour (PNG + PDF)")
        plot_retrieval_quality(out_dir)
        print(" [OK] retrieval_quality (PNG + PDF)")
        plot_efficiency_comparison(out_dir)
        print(" [OK] efficiency_comparison (PNG + PDF)")
        plot_latency_distribution(out_dir)
        print(" [OK] latency_distribution (PNG + PDF)")
        plot_error_analysis(out_dir)
        print(" [OK] error_analysis (PNG + PDF)")
        plot_gate_call_avoidance(out_dir)
        print(" [OK] gate_call_avoidance (PNG + PDF)")
        plot_retrieval_source_contribution(out_dir)
        print(" [OK] retrieval_source_contribution (PNG + PDF)")

    if args.benchmark in ("all", "heldout") or args.latest:
        print("Generating Held-out Benchmark Figures...")
        plot_heldout_variant_a_preliminary(out_dir)
        print(" [OK] heldout_variant_a_preliminary (PNG + PDF)")

    print("=" * 70)
    print("All requested figures generated successfully.")


if __name__ == "__main__":
    main()
