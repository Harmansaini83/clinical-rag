"""Unit tests for deterministic metrics computation."""
import json
import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.models import BenchmarkQuestion, Result, Run
from app.services.metrics import compute_metrics_for_run


@pytest.fixture
def mock_db_engine():
    """Create in-memory SQLite database populated with synthetic evaluation run."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # 1. Create Run
        run_a = Run(
            run_id="run_A_mock_test",
            variant="A",
            model_id="test-model",
            prompt_version="v1.0",
            corpus_manifest_hash="hashA",
            config_json="{}",
        )
        run_c = Run(
            run_id="run_C_mock_test",
            variant="C",
            model_id="test-model",
            prompt_version="v1.0",
            corpus_manifest_hash="hashC",
            config_json="{}",
        )
        session.add(run_a)
        session.add(run_c)

        # 2. Create Benchmark Questions
        # Q1: answerable, has gold passages
        q1 = BenchmarkQuestion(
            question_id="Q1",
            question="What is the dosage?",
            category="answerable",
            drug="metformin",
            expected_action="answer",
            gold_passage_ids_json=json.dumps(["P1", "P2"]),
        )
        # Q2: answerable, has gold passages
        q2 = BenchmarkQuestion(
            question_id="Q2",
            question="What are contraindications?",
            category="answerable",
            drug="metformin",
            expected_action="answer",
            gold_passage_ids_json=json.dumps(["P3"]),
        )
        # Q3: insufficient_evidence, NO gold passages
        q3 = BenchmarkQuestion(
            question_id="Q3",
            question="Pregnancy dosage?",
            category="insufficient_evidence",
            drug="metformin",
            expected_action="insufficient_evidence",
            gold_passage_ids_json=json.dumps([]),
        )
        # Q4: clarify, NO gold passages
        q4 = BenchmarkQuestion(
            question_id="Q4",
            question="What is the dose?",
            category="clarify",
            drug="",
            expected_action="clarify",
            gold_passage_ids_json=json.dumps([]),
        )
        # Q5: out_of_scope, NO gold passages
        q5 = BenchmarkQuestion(
            question_id="Q5",
            question="Give me medical diagnosis",
            category="out_of_scope",
            drug="",
            expected_action="out_of_scope",
            gold_passage_ids_json=json.dumps([]),
        )
        session.add_all([q1, q2, q3, q4, q5])

        # 3. Create Results for Run A
        r1_a = Result(
            run_id="run_A_mock_test",
            question_id="Q1",
            status="answer",
            answer="Dosing details",
            retrieved_ids_json=json.dumps(["P1", "P4"]),  # Hit (P1 in gold), but not covered (P2 missing)
            latency_seconds=1.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        r2_a = Result(
            run_id="run_A_mock_test",
            question_id="Q2",
            status="insufficient_evidence",  # Over-refusal!
            answer="No info",
            retrieved_ids_json=json.dumps(["P5"]),  # Miss
            latency_seconds=2.0,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        r3_a = Result(
            run_id="run_A_mock_test",
            question_id="Q3",
            status="insufficient_evidence",  # Appropriate abstention
            answer="No evidence in label",
            retrieved_ids_json=json.dumps(["P1"]),
            latency_seconds=1.5,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        r4_a = Result(
            run_id="run_A_mock_test",
            question_id="Q4",
            status="clarify",  # Appropriate clarification
            answer="Which drug?",
            retrieved_ids_json=json.dumps([]),
            latency_seconds=0.1,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        r5_a = Result(
            run_id="run_A_mock_test",
            question_id="Q5",
            status="out_of_scope",  # Appropriate scope refusal
            answer="Out of scope",
            retrieved_ids_json=json.dumps([]),
            latency_seconds=0.1,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        session.add_all([r1_a, r2_a, r3_a, r4_a, r5_a])

        # 4. Create Results for Run C (Q3 blocked by gate with 0 tokens)
        r1_c = Result(
            run_id="run_C_mock_test",
            question_id="Q1",
            status="answer",
            answer="Dosing details",
            retrieved_ids_json=json.dumps(["P1", "P2"]),  # Full coverage!
            latency_seconds=1.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        r2_c = Result(
            run_id="run_C_mock_test",
            question_id="Q2",
            status="answer",  # Correct answer!
            answer="Contraindicated if P3",
            retrieved_ids_json=json.dumps(["P3"]),
            latency_seconds=1.2,
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
        )
        r3_c = Result(
            run_id="run_C_mock_test",
            question_id="Q3",
            status="insufficient_evidence",
            answer="Gate blocked",
            retrieved_ids_json=json.dumps(["P1"]),
            latency_seconds=0.001,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,  # Gate avoided LLM call
        )
        r4_c = Result(
            run_id="run_C_mock_test",
            question_id="Q4",
            status="clarify",
            answer="Which drug?",
            retrieved_ids_json=json.dumps([]),
            latency_seconds=0.1,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        r5_c = Result(
            run_id="run_C_mock_test",
            question_id="Q5",
            status="out_of_scope",
            answer="Out of scope",
            retrieved_ids_json=json.dumps([]),
            latency_seconds=0.1,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        session.add_all([r1_c, r2_c, r3_c, r4_c, r5_c])
        session.commit()

    return engine


def test_evidence_hit_and_coverage(mock_db_engine):
    metrics_a = compute_metrics_for_run("run_A_mock_test", engine=mock_db_engine)
    # Only Q1 and Q2 have gold passages
    assert metrics_a["retrieval_eligible_questions"] == 2
    # Q1 has P1 (hit), Q2 has no hit (P5 retrieved, P3 gold)
    assert metrics_a["evidence_hits_top5"] == 1
    assert metrics_a["evidence_hit_rate"] == 0.5
    # Required coverage: Q1 only got P1 (needed P1 and P2) -> 0
    assert metrics_a["required_evidence_covered"] == 0
    assert metrics_a["required_evidence_coverage_rate"] == 0.0

    metrics_c = compute_metrics_for_run("run_C_mock_test", engine=mock_db_engine)
    assert metrics_c["evidence_hits_top5"] == 2
    assert metrics_c["evidence_hit_rate"] == 1.0
    assert metrics_c["required_evidence_covered"] == 2
    assert metrics_c["required_evidence_coverage_rate"] == 1.0


def test_status_match_and_over_refusal(mock_db_engine):
    metrics_a = compute_metrics_for_run("run_A_mock_test", engine=mock_db_engine)
    # Q1 matched (answer), Q2 missed (insufficient vs answer), Q3 matched, Q4 matched, Q5 matched -> 4/5
    assert metrics_a["status_matches"] == 4
    assert metrics_a["status_match_rate"] == 0.8
    # Q2 was over-refusal
    assert metrics_a["over_refusals"] == 1
    assert metrics_a["over_refusal_rate"] == 0.5


def test_latency_percentiles_and_token_totals(mock_db_engine):
    metrics_a = compute_metrics_for_run("run_A_mock_test", engine=mock_db_engine)
    assert metrics_a["total_tokens"] == 150 + 120 + 120 + 0 + 0
    assert metrics_a["groq_generation_calls"] == 3
    assert metrics_a["p50_latency"] > 0
    assert metrics_a["p95_latency"] >= metrics_a["p50_latency"]


def test_questions_without_gold_passages_excluded_from_retrieval(mock_db_engine):
    metrics_a = compute_metrics_for_run("run_A_mock_test", engine=mock_db_engine)
    # Total questions = 5, but retrieval eligible = 2
    assert metrics_a["total_questions"] == 5
    assert metrics_a["retrieval_eligible_questions"] == 2


def test_gate_avoided_groq_calls(mock_db_engine):
    metrics_c = compute_metrics_for_run("run_C_mock_test", engine=mock_db_engine)
    assert metrics_c["groq_calls_avoided_by_gate"] == 1
