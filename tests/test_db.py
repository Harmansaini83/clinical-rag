"""Database tests using temporary SQLite databases (no data/app.db touched)."""
import json
import tempfile
from pathlib import Path

import pytest
from sqlmodel import Session, create_engine, select

from app.db import create_db_and_tables
from app.models import (
    BenchmarkQuestion,
    Document,
    Judgment,
    LLMCache,
    Passage,
    Result,
    Run,
)


@pytest.fixture
def tmp_engine():
    """Provide an isolated in-memory SQLite engine for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    create_db_and_tables(engine)
    yield engine
    engine.dispose()


def test_db_initialisation_creates_all_seven_tables(tmp_engine):
    """All seven required tables must be present after init."""
    from sqlalchemy import inspect
    inspector = inspect(tmp_engine)
    tables = set(inspector.get_table_names())
    required = {
        "documents",
        "passages",
        "benchmark_questions",
        "runs",
        "results",
        "judgments",
        "llm_cache",
    }
    assert required.issubset(tables), f"Missing tables: {required - tables}"


def test_document_record_insert_and_retrieve(tmp_engine):
    doc = Document(
        document_id="30658e19-8e6e-49aa-a007-e414224797c5_v22",
        set_id="30658e19-8e6e-49aa-a007-e414224797c5",
        spl_version="22",
        effective_date="20261026",
        source_url="https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/test.xml",
        source_hash="abc123",
        raw_file="data/raw/test.xml",
        label_id="L001",
        ingredient="metformin hydrochloride",
        product_title="METFORMIN HYDROCHLORIDE TABLET",
        labeler="RemedyRepack Inc.",
        formulation="tablet",
        route="oral",
    )
    with Session(tmp_engine) as session:
        session.add(doc)
        session.commit()

    with Session(tmp_engine) as session:
        found = session.exec(select(Document).where(Document.label_id == "L001")).first()
        assert found is not None
        assert found.ingredient == "metformin hydrochloride"
        assert found.spl_version == "22"


def test_passage_record_insert_and_retrieve(tmp_engine):
    passage = Passage(
        passage_id="L001-4-p01",
        document_id="30658e19-8e6e-49aa-a007-e414224797c5_v22",
        label_id="L001",
        section_type="contraindications",
        section_title="4 CONTRAINDICATIONS",
        section_path="4 CONTRAINDICATIONS",
        text="Metformin hydrochloride tablets are contraindicated in patients with severe renal impairment.",
        word_count=14,
        flags_json="[]",
    )
    with Session(tmp_engine) as session:
        session.add(passage)
        session.commit()

    with Session(tmp_engine) as session:
        found = session.exec(
            select(Passage).where(Passage.passage_id == "L001-4-p01")
        ).first()
        assert found is not None
        assert found.section_type == "contraindications"
        assert found.word_count == 14
        # JSON field round-trips correctly
        assert json.loads(found.flags_json) == []


def test_benchmark_question_insert(tmp_engine):
    bq = BenchmarkQuestion(
        question_id="BQ-001",
        question="What does the metformin label say about renal impairment?",
        category="renal_impairment",
        drug="metformin hydrochloride",
        expected_action="answer",
        gold_answer_elements_json=json.dumps(["eGFR below 30", "contraindicated"]),
        gold_passage_ids_json=json.dumps(["L001-8.6-p01", "L001-2.3-p01"]),
        acceptable_alternative_passages_json="[]",
        evidence_group_id="EG-001",
        notes="Renal threshold question",
        split="dev",
    )
    with Session(tmp_engine) as session:
        session.add(bq)
        session.commit()

    with Session(tmp_engine) as session:
        found = session.exec(select(BenchmarkQuestion).where(BenchmarkQuestion.question_id == "BQ-001")).first()
        assert found is not None
        assert found.expected_action == "answer"
        gold_ids = json.loads(found.gold_passage_ids_json)
        assert "L001-8.6-p01" in gold_ids


def test_run_and_result_insert(tmp_engine):
    run = Run(
        run_id="run-001",
        variant="A",
        model_id="openai/gpt-oss-120b",
        prompt_version="v1",
        corpus_manifest_hash="abc123hash",
        config_json=json.dumps({"top_k": 5, "temperature": 0.0}),
    )
    with Session(tmp_engine) as session:
        session.add(run)
        session.commit()

    result = Result(
        run_id="run-001",
        question_id="BQ-001",
        status="answer",
        answer="Contraindicated in severe renal impairment.",
        retrieved_ids_json=json.dumps(["L001-8.6-p01", "L001-4-p01"]),
        latency_seconds=1.52,
        prompt_tokens=1500,
        completion_tokens=300,
        total_tokens=1800,
        error="",
    )
    with Session(tmp_engine) as session:
        session.add(result)
        session.commit()

    with Session(tmp_engine) as session:
        found = session.exec(
            select(Result).where(Result.run_id == "run-001")
        ).first()
        assert found is not None
        assert found.status == "answer"
        retrieved = json.loads(found.retrieved_ids_json)
        assert "L001-8.6-p01" in retrieved


def test_judgment_insert(tmp_engine):
    judgment = Judgment(
        run_id="run-001",
        question_id="BQ-001",
        correct=True,
        citation_supported=True,
        over_refusal=False,
        notes="Correct renal threshold cited.",
        reviewer="harman",
    )
    with Session(tmp_engine) as session:
        session.add(judgment)
        session.commit()

    with Session(tmp_engine) as session:
        found = session.exec(
            select(Judgment).where(Judgment.run_id == "run-001")
        ).first()
        assert found is not None
        assert found.correct is True
        assert found.reviewer == "harman"


def test_llm_cache_store_and_retrieve(tmp_engine):
    cache_entry = LLMCache(
        cache_key="sha256-test-key-abc123",
        model="openai/gpt-oss-120b",
        prompt_version="v1",
        variant="A",
        question_id="BQ-001",
        passage_ids_json=json.dumps(["L001-4-p01", "L001-8.6-p01"]),
        settings_json=json.dumps({"temperature": 0.0, "max_tokens": 2048}),
        response_json=json.dumps({"status": "answer", "answer": "Cached answer.", "evidence": []}),
        prompt_tokens=1400,
        completion_tokens=200,
        total_tokens=1600,
        original_latency_seconds=1.85,
    )
    with Session(tmp_engine) as session:
        session.add(cache_entry)
        session.commit()

    with Session(tmp_engine) as session:
        found = session.exec(
            select(LLMCache).where(LLMCache.cache_key == "sha256-test-key-abc123")
        ).first()
        assert found is not None
        assert found.model == "openai/gpt-oss-120b"
        assert found.original_latency_seconds == pytest.approx(1.85)
        response = json.loads(found.response_json)
        assert response["status"] == "answer"
        ids = json.loads(found.passage_ids_json)
        assert "L001-4-p01" in ids


def test_llm_cache_returns_none_for_unknown_key(tmp_engine):
    with Session(tmp_engine) as session:
        found = session.exec(
            select(LLMCache).where(LLMCache.cache_key == "nonexistent-key")
        ).first()
        assert found is None


def test_json_fields_roundtrip_correctly(tmp_engine):
    """All JSON text fields must serialize and deserialize correctly."""
    data = {
        "gold_elements": ["eGFR < 30", "contraindicated", "discontinue if eGFR falls"],
        "passage_ids": ["L001-4-p01", "L001-8.6-p01", "L001-2.3-p01"],
    }
    bq = BenchmarkQuestion(
        question_id="BQ-JSON-TEST",
        question="Test JSON roundtrip",
        category="renal_impairment",
        drug="metformin hydrochloride",
        expected_action="answer",
        gold_answer_elements_json=json.dumps(data["gold_elements"]),
        gold_passage_ids_json=json.dumps(data["passage_ids"]),
        acceptable_alternative_passages_json="[]",
        evidence_group_id="EG-JSON",
        notes="",
        split="dev",
    )
    with Session(tmp_engine) as session:
        session.add(bq)
        session.commit()

    with Session(tmp_engine) as session:
        found = session.exec(
            select(BenchmarkQuestion).where(BenchmarkQuestion.question_id == "BQ-JSON-TEST")
        ).first()
        assert json.loads(found.gold_answer_elements_json) == data["gold_elements"]
        assert json.loads(found.gold_passage_ids_json) == data["passage_ids"]
