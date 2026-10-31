"""Unit tests for visualization services, file generation, and non-Groq safety."""
from pathlib import Path
import pytest
from app.config import settings
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


@pytest.fixture
def tmp_figures_dir(tmp_path):
    """Provide a temporary directory for generated figures during test."""
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def test_all_development_figures_generated_as_png_and_pdf(tmp_figures_dir):
    """Verify all 8 development benchmark figures are saved as both PNG and PDF."""
    funcs = [
        plot_accuracy_comparison,
        plot_safety_behaviour,
        plot_retrieval_quality,
        plot_efficiency_comparison,
        plot_latency_distribution,
        plot_error_analysis,
        plot_gate_call_avoidance,
        plot_retrieval_source_contribution,
    ]
    for fn in funcs:
        res = fn(output_dir=tmp_figures_dir)
        assert res["png"].exists(), f"{fn.__name__} PNG missing"
        assert res["pdf"].exists(), f"{fn.__name__} PDF missing"
        assert res["png"].stat().st_size > 1000
        assert res["pdf"].stat().st_size > 1000


def test_heldout_variant_a_preliminary_figure_generated(tmp_figures_dir):
    """Verify held-out Variant A preliminary figure exists as PNG and PDF."""
    res = plot_heldout_variant_a_preliminary(output_dir=tmp_figures_dir)
    assert res["png"].exists()
    assert res["pdf"].exists()
    assert "heldout_variant_a_preliminary" in res["png"].name


def test_figures_use_variant_order_a_b_c_d(monkeypatch):
    """Verify plotting functions respect variant order A, B, C, D."""
    from app.services.visualisation import DEV_RUNS
    assert list(DEV_RUNS.keys()) == ["A", "B", "C", "D"]


def test_no_groq_calls_during_visualisation(monkeypatch, tmp_figures_dir):
    """Verify that generating all figures makes 0 calls to Groq API client."""
    def forbidden_groq(*args, **kwargs):
        raise RuntimeError("Groq client called during figure generation!")

    monkeypatch.setattr("groq.Groq", forbidden_groq)
    # Generate all figures; should run purely on local SQLite / CSV
    results = generate_all_figures(output_dir=tmp_figures_dir)
    assert len(results) == 9
