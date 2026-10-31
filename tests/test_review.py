"""Unit tests for review export and manual judgment preparation."""
import csv
import json
import pytest
from pathlib import Path
from sqlmodel import Session, SQLModel, create_engine

from app.models import BenchmarkQuestion, Result, Run
from app.services.review import export_review_csv, generate_review_rows, REVIEW_COLUMNS


@pytest.fixture
def mock_db_engine():
    """Create in-memory SQLite database populated with review test run."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        run = Run(
            run_id="run_review_test",
            variant="A",
            model_id="test-model",
            prompt_version="v1.0",
            corpus_manifest_hash="hash",
            config_json="{}",
        )
        session.add(run)

        q1 = BenchmarkQuestion(
            question_id="Q1",
            question="What is the eGFR cutoff?",
            category="answerable",
            drug="metformin",
            expected_action="answer",
            gold_passage_ids_json=json.dumps(["L001-4-p01"]),
        )
        q2 = BenchmarkQuestion(
            question_id="Q2",
            question="Is it safe in pregnancy?",
            category="insufficient_evidence",
            drug="metformin",
            expected_action="insufficient_evidence",
            gold_passage_ids_json=json.dumps([]),
        )
        session.add_all([q1, q2])

        r1 = Result(
            run_id="run_review_test",
            question_id="Q1",
            status="answer",
            answer="Contraindicated if eGFR < 30",
            retrieved_ids_json=json.dumps(["L001-4-p01", "L001-8.6-p01"]),
            latency_seconds=1.2,
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
        )
        r2 = Result(
            run_id="run_review_test",
            question_id="Q2",
            status="insufficient_evidence",
            answer="No label evidence found",
            retrieved_ids_json=json.dumps(["L001-4-p01"]),
            latency_seconds=0.8,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        session.add_all([r1, r2])
        session.commit()

    return engine


def test_generate_review_rows_fields_and_blanks(mock_db_engine):
    rows = generate_review_rows("run_review_test", engine=mock_db_engine)
    assert len(rows) == 2

    # Check required columns exist
    for col in REVIEW_COLUMNS:
        assert col in rows[0]

    # Check manual review fields are blank
    assert rows[0]["generated_claim_support"] == ""
    assert rows[0]["correct"] == ""
    assert rows[0]["over_refusal"] == ""
    assert rows[0]["reviewer_notes"] == ""
    assert rows[0]["reviewer"] == ""

    # Check computed flags
    assert rows[0]["status_match"] == "yes"
    assert rows[0]["evidence_hit"] == "yes"
    assert rows[0]["required_evidence_covered"] == "yes"

    # Q2 has no gold passages -> not_applicable
    assert rows[1]["evidence_hit"] == "not_applicable"
    assert rows[1]["required_evidence_covered"] == "not_applicable"


def test_export_review_csv(mock_db_engine, tmp_path):
    out_file = tmp_path / "test_review.csv"
    count = export_review_csv(["run_review_test"], out_file, engine=mock_db_engine)
    assert count == 2
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        read_rows = list(reader)
        assert len(read_rows) == 2
        assert read_rows[0]["question_id"] == "Q1"
        assert read_rows[0]["actual_status"] == "answer"


def test_unknown_run_id_raises_error(mock_db_engine):
    with pytest.raises(ValueError, match="Run ID not found"):
        generate_review_rows("non_existent_run", engine=mock_db_engine)
