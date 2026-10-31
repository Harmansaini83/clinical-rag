"""Additional test suite for Milestone 4: Held-out benchmark validation, figures, and retry logic."""
import json
from pathlib import Path
import pytest
from app.config import settings
from app.services.qa import validate_generated_response, ResponseValidationError


def test_heldout_benchmark_file_exists_and_has_40_questions():
    """Verify held-out benchmark exists and contains exactly 40 questions."""
    path = settings.data_dir / "benchmark" / "heldout_questions.jsonl"
    assert path.exists(), "heldout_questions.jsonl missing"
    
    with open(path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
        
    assert len(questions) == 40, f"Expected 40 held-out questions, found {len(questions)}"


def test_heldout_benchmark_category_distribution():
    """Verify exact category counts required for held-out benchmark."""
    path = settings.data_dir / "benchmark" / "heldout_questions.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    cats = [q["category"] for q in questions]
    assert cats.count("answerable") == 20
    assert cats.count("insufficient_evidence") == 8
    assert cats.count("clarify") == 5
    assert cats.count("out_of_scope") == 4
    assert cats.count("incorrect_premise") == 3


def test_heldout_gold_passage_ids_strictly_match_authoritative_corpus():
    """Verify all gold passages exist in L001 authoritative set."""
    authoritative_ids = {
        "L001-2.3-p01", "L001-4-p01", "L001-5.1-p01", "L001-5.1-p02", "L001-5.1-p03",
        "L001-5.2-p01", "L001-5.3-p01", "L001-5.4-p01", "L001-7-p01", "L001-7-p02", "L001-8.6-p01"
    }
    
    path = settings.data_dir / "benchmark" / "heldout_questions.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    for q in questions:
        for pid in q["gold_passage_ids"]:
            assert pid in authoritative_ids, f"Invalid gold passage {pid} in {q['question_id']}"
        for pid in q["acceptable_alternative_passages"]:
            assert pid in authoritative_ids, f"Invalid alt passage {pid} in {q['question_id']}"


def test_figures_generated():
    """Verify research paper comparison figures exist and are non-empty."""
    fig1 = settings.data_dir / "results" / "figures" / "accuracy_comparison.png"
    fig2 = settings.data_dir / "results" / "figures" / "efficiency_comparison.png"
    
    assert fig1.exists() and fig1.stat().st_size > 1000
    assert fig2.exists() and fig2.stat().st_size > 1000
