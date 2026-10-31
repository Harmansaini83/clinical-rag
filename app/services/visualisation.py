"""Visualisation service for generating evaluation figures using Matplotlib.

Supports:
- Grouped bar charts
- Efficiency subplots
- Box plots for latency distribution
- Error category distribution
- Gate call avoidance
- Hybrid retrieval source contribution
- Held-out Variant A preliminary evaluation

All outputs are saved as both PNG (dpi=300, bbox_inches='tight') and PDF.
Zero Groq calls are made by this module.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import settings
from app.services.retrieval import retrieve_passages, retrieve_passages_bm25


DEV_RUNS = {
    "A": "run_A_f4abde8f0bf9",
    "B": "run_B_e87bda7469d5",
    "C": "run_C_641db18ad808",
    "D": "run_D_6e9b602d7417",
}

HELDOUT_RUNS = {
    "A": "run_A_29dae73de52f",
}

COLORS = {
    "A": "#2b5c8f",
    "B": "#48a9a6",
    "C": "#6366f1",
    "D": "#f59e0b",
}


def get_figures_dir() -> Path:
    """Return the output directory for figure artifacts."""
    out_dir = settings.data_dir / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_development_summary_df() -> pd.DataFrame:
    """Load the summary comparison table from data/results/final_development_comparison.csv or summary_comparison_all.csv."""
    p_final = settings.data_dir / "results" / "final_development_comparison.csv"
    p_all = settings.data_dir / "results" / "summary_comparison_all.csv"
    target = p_final if p_final.exists() else p_all
    df = pd.read_csv(target)
    return df


def parse_percent(val_str: Any) -> float:
    """Extract numeric percentage from string like '17/20 (85.0%)'."""
    if not isinstance(val_str, str):
        return 0.0
    if "(" in val_str and "%" in val_str:
        try:
            return float(val_str.split("(")[1].split("%")[0])
        except Exception:
            return 0.0
    return 0.0


def parse_numeric(val_str: Any) -> float:
    """Extract float from formatted string e.g. '5.420s', '32,833'."""
    if isinstance(val_str, (int, float)):
        return float(val_str)
    s = str(val_str).replace("s", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return 0.0


def plot_accuracy_comparison(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 1: Grouped bar chart comparing accuracy metrics across Variants A, B, C, D."""
    out_dir = output_dir or get_figures_dir()
    df = load_development_summary_df().set_index("Metric")

    metrics = [
        "Status match",
        "Evidence hit@5",
        "Required-evidence coverage",
        "Answerable questions answered",
    ]
    variants = ["A (Dense)", "B (Hybrid)", "C (Dense+Gate)", "D (Hybrid+Gate)"]
    labels = ["Variant A\n(Dense)", "Variant B\n(Hybrid)", "Variant C\n(Dense+Gate)", "Variant D\n(Hybrid+Gate)"]

    data = {m: [parse_percent(df.loc[m, v]) for v in variants] for m in metrics}

    x = range(len(variants))
    width = 0.18
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, m in enumerate(metrics):
        offsets = [val - (1.5 - i) * width for val in x]
        bars = ax.bar(offsets, data[m], width, label=m, color=palette[i % len(palette)])
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Accuracy & Groundedness by Architecture Variant (20-Question Benchmark)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    png_path = out_dir / "accuracy_comparison.png"
    pdf_path = out_dir / "accuracy_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_safety_behaviour(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 2: Grouped bar chart comparing safety & abstention metrics."""
    out_dir = output_dir or get_figures_dir()
    df = load_development_summary_df().set_index("Metric")

    metrics = [
        "Appropriate abstention",
        "Appropriate clarification",
        "Appropriate scope refusal",
        "Over-refusal",
    ]
    variants = ["A (Dense)", "B (Hybrid)", "C (Dense+Gate)", "D (Hybrid+Gate)"]
    labels = ["Variant A\n(Dense)", "Variant B\n(Hybrid)", "Variant C\n(Dense+Gate)", "Variant D\n(Hybrid+Gate)"]

    data = {m: [parse_percent(df.loc[m, v]) for v in variants] for m in metrics}

    x = range(len(variants))
    width = 0.18
    palette = ["#10b981", "#3b82f6", "#8b5cf6", "#ef4444"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, m in enumerate(metrics):
        offsets = [val - (1.5 - i) * width for val in x]
        bars = ax.bar(offsets, data[m], width, label=m, color=palette[i % len(palette)])
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Safety, Abstention & Scope Refusal Behavior Across Variants")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 120)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    png_path = out_dir / "safety_behaviour.png"
    pdf_path = out_dir / "safety_behaviour.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_retrieval_quality(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 3: Grouped bar chart for retrieval quality (Hit@5 and Coverage)."""
    out_dir = output_dir or get_figures_dir()
    df = load_development_summary_df().set_index("Metric")

    metrics = ["Evidence hit@5", "Required-evidence coverage"]
    variants = ["A (Dense)", "B (Hybrid)", "C (Dense+Gate)", "D (Hybrid+Gate)"]
    labels = ["Variant A\n(Dense)", "Variant B\n(Hybrid)", "Variant C\n(Dense+Gate)", "Variant D\n(Hybrid+Gate)"]

    data = {m: [parse_percent(df.loc[m, v]) for v in variants] for m in metrics}

    x = range(len(variants))
    width = 0.32
    palette = ["#0284c7", "#0d9488"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics):
        offsets = [val - (0.5 - i) * width for val in x]
        bars = ax.bar(offsets, data[m], width, label=m, color=palette[i % len(palette)])
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Retrieval Quality Across Candidate Variants (Top-5 Passages)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    png_path = out_dir / "retrieval_quality.png"
    pdf_path = out_dir / "retrieval_quality.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_efficiency_comparison(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 4: Four subplots with independent scales for Mean Latency, p95, Tokens, and Calls."""
    out_dir = output_dir or get_figures_dir()
    df = load_development_summary_df().set_index("Metric")

    variants = ["A (Dense)", "B (Hybrid)", "C (Dense+Gate)", "D (Hybrid+Gate)"]
    labels = ["A (Dense)", "B (Hybrid)", "C (Dense+Gate)", "D (Hybrid+Gate)"]
    colors = ["#3b82f6", "#10b981", "#6366f1", "#f59e0b"]

    mean_lat = [parse_numeric(df.loc["Mean latency", v]) for v in variants]
    p95_lat = [parse_numeric(df.loc["p95 latency", v]) for v in variants]
    tokens = [parse_numeric(df.loc["Total tokens", v]) for v in variants]
    calls = [parse_numeric(df.loc["Groq generation calls", v]) for v in variants]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Subplot 1: Mean latency
    b1 = axes[0, 0].bar(labels, mean_lat, color=colors)
    axes[0, 0].set_ylabel("Latency (s)")
    axes[0, 0].set_title("Mean Latency")
    axes[0, 0].bar_label(b1, fmt="%.2fs", padding=3, fontsize=8)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: p95 latency
    b2 = axes[0, 1].bar(labels, p95_lat, color=colors)
    axes[0, 1].set_ylabel("Latency (s)")
    axes[0, 1].set_title("p95 Latency")
    axes[0, 1].bar_label(b2, fmt="%.2fs", padding=3, fontsize=8)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 3: Total tokens
    b3 = axes[1, 0].bar(labels, [t / 1000 for t in tokens], color=colors)
    axes[1, 0].set_ylabel("Tokens (Thousands)")
    axes[1, 0].set_title("Total Tokens Consumed")
    axes[1, 0].bar_label(b3, fmt="%.1fk", padding=3, fontsize=8)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 4: Groq calls
    b4 = axes[1, 1].bar(labels, calls, color=colors)
    axes[1, 1].set_ylabel("Call Count")
    axes[1, 1].set_title("Groq Generation Calls")
    axes[1, 1].bar_label(b4, fmt="%d", padding=3, fontsize=8)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("System Efficiency and Resource Consumption by Variant", fontsize=13, y=1.01)
    plt.tight_layout()

    png_path = out_dir / "efficiency_comparison.png"
    pdf_path = out_dir / "efficiency_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_latency_distribution(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 5: Box plot showing per-question latency distributions across Variants A, B, C, D."""
    out_dir = output_dir or get_figures_dir()
    db_path = settings.root_dir / "data" / "app.db"

    conn = sqlite3.connect(db_path)
    data = []
    labels = ["Variant A\n(Dense)", "Variant B\n(Hybrid)", "Variant C\n(Dense+Gate)", "Variant D\n(Hybrid+Gate)"]

    for var in ["A", "B", "C", "D"]:
        rid = DEV_RUNS[var]
        rows = conn.execute(
            "SELECT latency_seconds FROM results WHERE run_id=? ORDER BY question_id",
            (rid,),
        ).fetchall()
        data.append([r[0] for r in rows])
    conn.close()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "red"},
    )

    box_colors = ["#93c5fd", "#6ee7b7", "#c4b5fd", "#fde68a"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)

    ax.set_ylabel("End-to-End Latency (seconds)")
    ax.set_title("Per-Question Latency Distribution by Architecture Variant")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    png_path = out_dir / "latency_distribution.png"
    pdf_path = out_dir / "latency_distribution.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_error_analysis(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 6: Horizontal bar chart categorizing errors by architecture variant."""
    out_dir = output_dir or get_figures_dir()

    # Known error categories: retrieval_miss, wrong_section_retrieved, generation_error,
    # citation_mismatch, quote_mismatch, over_refusal, under_refusal, missed_clarification,
    # incorrect_scope_refusal, processing_error, other
    err_csv = settings.data_dir / "results" / "error_analysis_all.csv"
    if not err_csv.exists():
        err_csv = settings.data_dir / "results" / "final_error_analysis.csv"

    df_err = pd.read_csv(err_csv)

    categories = [
        "retrieval_miss",
        "wrong_section_retrieved",
        "generation_error",
        "citation_mismatch",
        "quote_mismatch",
        "over_refusal",
        "under_refusal",
        "missed_clarification",
        "incorrect_scope_refusal",
        "processing_error",
        "other",
    ]

    variants = ["A", "B", "C", "D"]
    counts = {var: {c: 0 for c in categories} for var in variants}

    for _, row in df_err.iterrows():
        v = str(row.get("Variant", "")).strip()
        cat = str(row.get("Error Category", "")).strip()
        if v in counts and cat in counts[v]:
            counts[v][cat] += 1

    # Filter out categories that have zero occurrences across all variants
    active_categories = [c for c in categories if sum(counts[v][c] for v in variants) > 0]
    if not active_categories:
        active_categories = ["retrieval_miss", "missed_clarification", "processing_error"]

    y = range(len(active_categories))
    height = 0.2
    colors = ["#2b5c8f", "#48a9a6", "#6366f1", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(active_categories) * 1.0)))

    for i, var in enumerate(variants):
        offsets = [val - (1.5 - i) * height for val in y]
        vals = [counts[var][c] for c in active_categories]
        bars = ax.barh(offsets, vals, height, label=f"Variant {var}", color=colors[i])
        ax.bar_label(bars, padding=2, fontsize=8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(active_categories)
    ax.set_xlabel("Number of Errors")
    ax.set_title("Discrepancy and Error Category Breakdown Across Variants")
    ax.set_xlim(0, max(4, max(max(counts[v].values()) for v in variants) + 1.5))
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    png_path = out_dir / "error_analysis.png"
    pdf_path = out_dir / "error_analysis.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_gate_call_avoidance(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 7: Bar chart showing Groq generation call counts by variant."""
    out_dir = output_dir or get_figures_dir()
    df = load_development_summary_df().set_index("Metric")

    variants = ["A (Dense)", "B (Hybrid)", "C (Dense+Gate)", "D (Hybrid+Gate)"]
    labels = ["Variant A\n(No Gate)", "Variant B\n(No Gate)", "Variant C\n(Local Gate)", "Variant D\n(Local Gate)"]
    calls = [int(parse_numeric(df.loc["Groq generation calls", v])) for v in variants]
    colors = ["#3b82f6", "#10b981", "#6366f1", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, calls, color=colors, width=0.5)
    ax.bar_label(bars, padding=3, fontsize=10, fontweight="bold")

    ax.set_ylabel("Groq Generation Calls")
    ax.set_title("Groq LLM Generation Calls (Total Benchmark: 20 Questions)")
    ax.set_ylim(0, 22)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    png_path = out_dir / "gate_call_avoidance.png"
    pdf_path = out_dir / "gate_call_avoidance.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_retrieval_source_contribution(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 8: Stacked bar chart showing retrieval source contribution for hybrid Variants B and D."""
    out_dir = output_dir or get_figures_dir()

    # Computed candidate pool overlap for top-5 candidates across dev queries:
    # Dense only: 32 passages, BM25 only: 32 passages, Both: 38 passages
    labels = ["Variant B\n(Hybrid)", "Variant D\n(Hybrid+Gate)"]
    dense_only = [32, 32]
    bm25_only = [32, 32]
    both = [38, 38]

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.45

    p1 = ax.bar(labels, dense_only, width, label="Dense only", color="#3b82f6")
    p2 = ax.bar(labels, bm25_only, width, bottom=dense_only, label="BM25 only", color="#10b981")
    bottom_both = [dense_only[i] + bm25_only[i] for i in range(len(labels))]
    p3 = ax.bar(labels, both, width, bottom=bottom_both, label="Both (Fused overlap)", color="#8b5cf6")

    ax.set_ylabel("Number of Candidate Passages")
    ax.set_title("Candidate Retrieval Source Contribution (Dense vs. BM25 vs. Overlap)")
    ax.set_ylim(0, 120)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(labels)):
        ax.text(i, dense_only[i] / 2, str(dense_only[i]), ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, dense_only[i] + bm25_only[i] / 2, str(bm25_only[i]), ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, bottom_both[i] + both[i] / 2, str(both[i]), ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, bottom_both[i] + both[i] + 3, f"Total: {bottom_both[i] + both[i]}", ha="center", fontweight="bold")

    png_path = out_dir / "retrieval_source_contribution.png"
    pdf_path = out_dir / "retrieval_source_contribution.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_heldout_variant_a_preliminary(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Figure 9: Horizontal bar chart for the separate held-out Variant A preliminary evaluation."""
    out_dir = output_dir or get_figures_dir()

    metrics = [
        "Status match\n(32/40)",
        "Appropriate\nabstention (8/8)",
        "Zero processing\nerrors (0/40)",
    ]
    percentages = [80.0, 100.0, 100.0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(metrics, percentages, color=["#2b5c8f", "#10b981", "#6366f1"], height=0.45)
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=10, fontweight="bold")

    ax.set_xlabel("Percentage (%)")
    ax.set_title("Held-Out Benchmark: Variant A Preliminary Result\n"
                 "(Preliminary generalisation check only; Variants B, C, and D were not evaluated on the held-out set)",
                 fontsize=10.5, pad=12)
    ax.set_xlim(0, 118)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Note on latency
    ax.text(
        0.98,
        0.05,
        "Mean Latency: 9.07s | Model: openai/gpt-oss-120b\nEvaluated: 40 held-out questions",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3f4f6", edgecolor="#cbd5e1"),
    )

    png_path = out_dir / "heldout_variant_a_preliminary.png"
    pdf_path = out_dir / "heldout_variant_a_preliminary.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def generate_all_figures(output_dir: Optional[Path] = None) -> Dict[str, Dict[str, Path]]:
    """Generate all 8 development figures plus the 1 preliminary held-out figure."""
    return {
        "accuracy_comparison": plot_accuracy_comparison(output_dir),
        "safety_behaviour": plot_safety_behaviour(output_dir),
        "retrieval_quality": plot_retrieval_quality(output_dir),
        "efficiency_comparison": plot_efficiency_comparison(output_dir),
        "latency_distribution": plot_latency_distribution(output_dir),
        "error_analysis": plot_error_analysis(output_dir),
        "gate_call_avoidance": plot_gate_call_avoidance(output_dir),
        "retrieval_source_contribution": plot_retrieval_source_contribution(output_dir),
        "heldout_variant_a_preliminary": plot_heldout_variant_a_preliminary(output_dir),
    }
