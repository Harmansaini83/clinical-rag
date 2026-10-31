"""Tests for benchmark evaluation harness, Variant A, Variant C, caching, and persistence."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import BenchmarkQuestion, Result, Run, LLMCache
from app.services.evaluation import (
    evaluate_benchmark,
    execute_variant_a,
    execute_variant_b,
    execute_variant_c,
    execute_variant_d,
    load_benchmark_questions,
    sync_benchmark_questions_to_db,
)


@pytest.fixture
def temp_engine():
    """In-memory SQLite engine for evaluation harness tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def test_benchmark_loads_exactly_20_questions():
    """dev_questions.jsonl must contain exactly 20 questions."""
    questions = load_benchmark_questions()
    assert len(questions) == 20


def test_every_gold_passage_id_exists_in_l001():
    """Every gold_passage_id must strictly exist in authoritative L001.jsonl."""
    l001_path = Path("data/passages/L001.jsonl")
    assert l001_path.exists(), "data/passages/L001.jsonl missing"

    authoritative_ids = set()
    with open(l001_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                authoritative_ids.add(p["passage_id"])

    questions = load_benchmark_questions()
    for q in questions:
        for gid in q.get("gold_passage_ids", []):
            assert gid in authoritative_ids, f"Gold ID {gid} in {q['question_id']} not in L001.jsonl"


def test_every_question_has_valid_category_and_expected_action():
    """Every question must use valid category and action values."""
    valid_categories = {"answerable", "insufficient_evidence", "clarify", "out_of_scope"}
    valid_actions = {"answer", "insufficient_evidence", "clarify", "out_of_scope"}

    questions = load_benchmark_questions()
    for q in questions:
        assert q["category"] in valid_categories, f"Invalid category {q['category']} in {q['question_id']}"
        assert q["expected_action"] in valid_actions, f"Invalid action {q['expected_action']} in {q['question_id']}"


@patch("app.services.evaluation.retrieve_passages")
@patch("app.services.evaluation.generate_grounded_response")
def test_variant_a_runs_with_mocked_generation(mock_gen, mock_ret):
    """Variant A executes dense retrieval and guarded generation."""
    mock_ret.return_value = [
        {
            "passage_id": "L001-2.3-p01",
            "document_id": "30658e19-8e6e-49aa-a007-e414224797c5_v22",
            "section_type": "renal_impairment",
            "section_title": "2.3 Renal Impairment",
            "section_path": "2 DOSAGE AND ADMINISTRATION > 2.3",
            "text": "Metformin is contraindicated with eGFR below 30 mL/minute/1.73 m 2.",
        }
    ]
    mock_gen.return_value = (
        {
            "status": "answer",
            "answer": "Metformin is contraindicated with eGFR below 30.",
            "evidence": [{"passage_id": "L001-2.3-p01", "quote": "contraindicated with eGFR below 30"}],
            "clarification_question": "",
            "limitations": [],
        },
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        1.25,
    )

    res, retrieved = execute_variant_a(
        question="What does the metformin label say about renal impairment?",
        question_id="DEV-Q001",
        groq_client=MagicMock(),
        use_cache=False,
    )

    assert res["status"] == "answer"
    assert len(res["evidence"]) == 1
    assert len(retrieved) == 1
    mock_gen.assert_called_once()


@patch("app.services.evaluation.retrieve_passages")
@patch("app.services.evaluation.generate_grounded_response")
def test_variant_c_runs_with_mocked_generation(mock_gen, mock_ret):
    """Variant C executes evidence gate and proceeds when evidence is sufficient."""
    mock_ret.return_value = [
        {
            "passage_id": "L001-2.3-p01",
            "document_id": "30658e19-8e6e-49aa-a007-e414224797c5_v22",
            "section_type": "renal_impairment",
            "section_title": "2.3 Renal Impairment",
            "section_path": "2 DOSAGE AND ADMINISTRATION > 2.3",
            "text": "Metformin is contraindicated with eGFR below 30 mL/minute/1.73 m 2.",
        }
    ]
    mock_gen.return_value = (
        {
            "status": "answer",
            "answer": "Contraindicated below 30.",
            "evidence": [{"passage_id": "L001-2.3-p01", "quote": "contraindicated with eGFR below 30"}],
            "clarification_question": "",
            "limitations": [],
        },
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        0.95,
    )

    res, retrieved = execute_variant_c(
        question="What does the metformin label say about renal impairment?",
        question_id="DEV-Q001",
        groq_client=MagicMock(),
        use_cache=False,
    )

    assert res["status"] == "answer"
    mock_gen.assert_called_once()


@patch("app.services.evaluation.retrieve_passages")
@patch("app.services.evaluation.generate_grounded_response")
def test_variant_c_gate_prevents_generation_when_evidence_insufficient(mock_gen, mock_ret):
    """Variant C evidence gate blocks generation call when evidence is insufficient (e.g. pregnancy qualifier)."""
    mock_ret.return_value = [
        {
            "passage_id": "L001-2.3-p01",
            "section_type": "renal_impairment",
            "text": "Renal dosing information and eGFR cutoffs.",
        }
    ]

    res, retrieved = execute_variant_c(
        question="What is the recommended metformin dosage during pregnancy?",
        question_id="DEV-Q011",
        groq_client=MagicMock(),
        use_cache=False,
    )

    assert res["status"] == "insufficient_evidence"
    assert "required_qualifier_not_found" in res["limitations"]
    # Crucial assertion: generation was prevented!
    mock_gen.assert_not_called()


@patch("app.services.evaluation.execute_variant_a")
def test_results_saved_to_sqlite(mock_exec, temp_engine):
    """Evaluation results and run records are saved into SQLite."""
    mock_exec.return_value = (
        {
            "status": "answer",
            "answer": "Mocked test answer",
            "tokens_used": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            "latency_seconds": 1.1,
            "error": "",
        },
        [{"passage_id": "L001-2.3-p01"}],
    )

    summary = evaluate_benchmark(
        variant="A",
        engine=temp_engine,
        limit=2,
        use_cache=False,
    )

    assert summary["executed"] == 2
    with Session(temp_engine) as session:
        runs = session.exec(select(Run)).all()
        assert len(runs) == 1
        assert runs[0].variant == "A"

        results = session.exec(select(Result)).all()
        assert len(results) == 2
        assert results[0].status == "answer"
        assert results[0].question_id == "DEV-Q001"


@patch("app.services.evaluation.execute_variant_a")
def test_rerunning_same_configuration_skips_completed_questions(mock_exec, temp_engine):
    """Resumable: running twice with same run_id skips already evaluated questions."""
    mock_exec.return_value = (
        {
            "status": "answer",
            "answer": "Mocked test answer",
            "tokens_used": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            "latency_seconds": 1.1,
        },
        [{"passage_id": "L001-2.3-p01"}],
    )

    # First run: 2 executed
    s1 = evaluate_benchmark(variant="A", engine=temp_engine, limit=2, use_cache=False)
    assert s1["executed"] == 2
    assert s1["skipped"] == 0

    # Second run: 0 executed, 2 skipped
    s2 = evaluate_benchmark(variant="A", engine=temp_engine, limit=2, use_cache=False)
    assert s2["executed"] == 0
    assert s2["skipped"] == 2
    assert mock_exec.call_count == 2  # Only called during run 1!


@patch("app.services.evaluation.execute_variant_a")
def test_failed_question_does_not_stop_entire_run(mock_exec, temp_engine):
    """If one question encounters an exception, subsequent questions continue executing."""
    # First question raises, second succeeds
    mock_exec.side_effect = [
        RuntimeError("Provider network timeout"),
        (
            {
                "status": "answer",
                "answer": "Success on question 2",
                "tokens_used": {},
                "latency_seconds": 0.5,
            },
            [],
        ),
    ]

    summary = evaluate_benchmark(variant="A", engine=temp_engine, limit=2, use_cache=False)

    assert summary["total"] == 2
    assert summary["executed"] == 2
    assert summary["errors"] == 1
    assert summary["results"][0]["status"] == "processing_error"
    assert summary["results"][1]["status"] == "answer"


def test_cache_is_used_for_repeated_identical_requests(temp_engine):
    """When cache is enabled, lookup_llm_cache retrieves stored response without Groq."""
    with patch("app.db.get_engine", return_value=temp_engine):
        from app.services.generation import (
            compute_llm_cache_key,
            generate_grounded_response,
            lookup_llm_cache,
            store_llm_cache,
        )

        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "status": "answer",
                "answer": "Cached answer text",
                "evidence": [],
                "clarification_question": "",
                "limitations": [],
            })))
        ]
        mock_completion.usage = MagicMock(prompt_tokens=120, completion_tokens=40, total_tokens=160)
        mock_client.chat.completions.create.return_value = mock_completion

        passages = [{"passage_id": "L001-2.3-p01", "text": "eGFR below 30"}]

        # Call 1: uncached, calls Groq and caches result
        data1, usage1, lat1 = generate_grounded_response(
            question="What is the contraindication for metformin?",
            passages=passages,
            client=mock_client,
            question_id="DEV-Q001",
            variant="A",
            use_cache=True,
        )
        assert data1["status"] == "answer"
        assert mock_client.chat.completions.create.call_count == 1

        # Call 2: should hit SQLite llm_cache, zero Groq calls
        data2, usage2, lat2 = generate_grounded_response(
            question="What is the contraindication for metformin?",
            passages=passages,
            client=mock_client,
            question_id="DEV-Q001",
            variant="A",
            use_cache=True,
        )
        assert data2["status"] == "answer"
        assert data2["answer"] == "Cached answer text"
        assert mock_client.chat.completions.create.call_count == 1  # Still 1!


@patch("app.services.evaluation.retrieve_passages_hybrid")
@patch("app.services.evaluation.generate_grounded_response")
def test_variant_b_runs_with_mocked_generation(mock_gen, mock_ret):
    """Variant B executes hybrid retrieval and guarded generation."""
    mock_ret.return_value = [
        {
            "passage_id": "L001-2.3-p01",
            "document_id": "30658e19-8e6e-49aa-a007-e414224797c5_v22",
            "section_type": "renal_impairment",
            "section_title": "2.3 Renal Impairment",
            "section_path": "2 DOSAGE AND ADMINISTRATION > 2.3",
            "text": "Metformin is contraindicated with eGFR below 30 mL/minute/1.73 m 2.",
            "retrieval_source": "both",
        }
    ]
    mock_gen.return_value = (
        {
            "status": "answer",
            "answer": "Contraindicated below 30.",
            "evidence": [{"passage_id": "L001-2.3-p01", "quote": "contraindicated with eGFR below 30"}],
            "clarification_question": "",
            "limitations": [],
        },
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        1.05,
    )

    res, retrieved = execute_variant_b(
        question="What does the metformin label say about renal impairment?",
        question_id="DEV-Q001",
        groq_client=MagicMock(),
        use_cache=False,
    )

    assert res["status"] == "answer"
    assert len(retrieved) == 1
    assert retrieved[0]["retrieval_source"] == "both"
    mock_gen.assert_called_once()


@patch("app.services.evaluation.retrieve_passages_hybrid")
@patch("app.services.evaluation.generate_grounded_response")
def test_variant_d_runs_with_mocked_generation(mock_gen, mock_ret):
    """Variant D executes hybrid retrieval and local evidence gate, then guarded generation."""
    mock_ret.return_value = [
        {
            "passage_id": "L001-2.3-p01",
            "document_id": "30658e19-8e6e-49aa-a007-e414224797c5_v22",
            "section_type": "renal_impairment",
            "section_title": "2.3 Renal Impairment",
            "section_path": "2 DOSAGE AND ADMINISTRATION > 2.3",
            "text": "Metformin is contraindicated with eGFR below 30 mL/minute/1.73 m 2.",
            "retrieval_source": "both",
        }
    ]
    mock_gen.return_value = (
        {
            "status": "answer",
            "answer": "Contraindicated below 30.",
            "evidence": [{"passage_id": "L001-2.3-p01", "quote": "contraindicated with eGFR below 30"}],
            "clarification_question": "",
            "limitations": [],
        },
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        0.85,
    )

    res, retrieved = execute_variant_d(
        question="What does the metformin label say about renal impairment?",
        question_id="DEV-Q001",
        groq_client=MagicMock(),
        use_cache=False,
    )

    assert res["status"] == "answer"
    mock_gen.assert_called_once()


@patch("app.services.evaluation.retrieve_passages_hybrid")
@patch("app.services.evaluation.generate_grounded_response")
def test_variant_d_gate_prevents_generation_when_evidence_insufficient(mock_gen, mock_ret):
    """Variant D gate blocks generation when evidence is insufficient."""
    mock_ret.return_value = [
        {
            "passage_id": "L001-2.3-p01",
            "section_type": "renal_impairment",
            "text": "Renal dosing information.",
            "retrieval_source": "both",
        }
    ]

    res, retrieved = execute_variant_d(
        question="What is the recommended metformin dosage during pregnancy?",
        question_id="DEV-Q011",
        groq_client=MagicMock(),
        use_cache=False,
    )

    assert res["status"] == "insufficient_evidence"
    assert "required_qualifier_not_found" in res["limitations"]
    mock_gen.assert_not_called()


def test_b_and_d_results_saved_to_sqlite(temp_engine):
    """Variant B and D evaluation saves results to SQLite."""
    with patch("app.services.evaluation.execute_variant_b") as mock_exec_b, \
         patch("app.services.evaluation.execute_variant_d") as mock_exec_d:
        
        mock_exec_b.return_value = (
            {"status": "answer", "answer": "Ans B", "latency_seconds": 1.0, "tokens_used": {"total_tokens": 100}},
            [{"passage_id": "L001-2.3-p01"}],
        )
        mock_exec_d.return_value = (
            {"status": "answer", "answer": "Ans D", "latency_seconds": 0.8, "tokens_used": {"total_tokens": 90}},
            [{"passage_id": "L001-2.3-p01"}],
        )

        sum_b = evaluate_benchmark(variant="B", engine=temp_engine, limit=1)
        sum_d = evaluate_benchmark(variant="D", engine=temp_engine, limit=1)

        with Session(temp_engine) as session:
            runs = session.exec(select(Run)).all()
            assert len(runs) == 2
            results = session.exec(select(Result)).all()
            assert len(results) == 2


def test_rerunning_b_and_d_skips_completed_questions(temp_engine):
    """Resumable: rerunning Variant B/D skips completed questions."""
    with patch("app.services.evaluation.execute_variant_b") as mock_exec:
        mock_exec.return_value = (
            {"status": "answer", "answer": "Ans", "latency_seconds": 1.0, "tokens_used": {}},
            [],
        )

        sum1 = evaluate_benchmark(variant="B", engine=temp_engine, limit=2)
        assert sum1["executed"] == 2
        assert sum1["skipped"] == 0

        sum2 = evaluate_benchmark(variant="B", engine=temp_engine, limit=2)
        assert sum2["executed"] == 0
        assert sum2["skipped"] == 2

