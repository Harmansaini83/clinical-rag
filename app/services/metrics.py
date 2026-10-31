"""Deterministic metrics computation for clinical RAG evaluation runs."""
import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from sqlmodel import Session, select
from app.db import get_engine
from app.models import BenchmarkQuestion, Result, Run

logger = logging.getLogger(__name__)


def compute_metrics_for_run(run_id: str, engine=None) -> Dict[str, Any]:
    """Compute all deterministic evaluation metrics for a specific run_id.

    Returns dictionary of metric names mapped to numeric values and formatted strings.
    """
    if engine is None:
        engine = get_engine()

    with Session(engine) as session:
        run = session.exec(select(Run).where(Run.run_id == run_id)).first()
        if not run:
            raise ValueError(f"Run ID not found: {run_id}")

        results = session.exec(select(Result).where(Result.run_id == run_id)).all()
        questions = session.exec(select(BenchmarkQuestion)).all()

    if not results:
        raise ValueError(f"No results found for run ID: {run_id}")

    q_map = {q.question_id: q for q in questions}

    total_questions = len(results)
    status_matches = 0
    processing_errors = 0

    # Question category subsets
    answerable_count = 0
    answerable_answered = 0
    over_refusals = 0

    expected_insufficient = 0
    actual_insufficient = 0

    expected_clarify = 0
    actual_clarify = 0

    expected_scope = 0
    actual_scope = 0

    # Retrieval metrics (only eligible questions with non-empty gold passages)
    retrieval_eligible = 0
    evidence_hits_top5 = 0
    required_evidence_covered = 0

    latencies = []
    tokens = []
    groq_calls_avoided = 0

    for r in results:
        q = q_map.get(r.question_id)
        if not q:
            continue

        latencies.append(r.latency_seconds)
        tokens.append(r.total_tokens)

        # Check groq calls avoided by gate (Variant C with zero tokens and insufficient_evidence)
        if run.variant == "C" and r.status == "insufficient_evidence" and r.total_tokens == 0:
            groq_calls_avoided += 1

        if r.status == "processing_error":
            processing_errors += 1

        if r.status == q.expected_action:
            status_matches += 1

        gold_ids = json.loads(q.gold_passage_ids_json or "[]")
        retrieved_ids = json.loads(r.retrieved_ids_json or "[]")

        # Retrieval hit and coverage for eligible questions
        if gold_ids:
            retrieval_eligible += 1
            retrieved_set = set(retrieved_ids)
            gold_set = set(gold_ids)

            # Hit@5: at least one gold passage in retrieved
            if any(gid in retrieved_set for gid in gold_ids):
                evidence_hits_top5 += 1

            # Required evidence coverage: all gold passages retrieved
            if gold_set.issubset(retrieved_set):
                required_evidence_covered += 1

        # Category specific evaluations
        if q.expected_action == "answer":
            answerable_count += 1
            if r.status == "answer":
                answerable_answered += 1
            else:
                over_refusals += 1

        elif q.expected_action == "insufficient_evidence":
            expected_insufficient += 1
            if r.status == "insufficient_evidence":
                actual_insufficient += 1

        elif q.expected_action == "clarify":
            expected_clarify += 1
            if r.status == "clarify":
                actual_clarify += 1

        elif q.expected_action == "out_of_scope":
            expected_scope += 1
            if r.status == "out_of_scope":
                actual_scope += 1

    # Computations
    latencies_arr = np.array(latencies) if latencies else np.array([0.0])
    tokens_arr = np.array(tokens) if tokens else np.array([0])

    p50_latency = float(np.percentile(latencies_arr, 50))
    p95_latency = float(np.percentile(latencies_arr, 95))
    mean_latency = float(np.mean(latencies_arr))

    mean_tokens = float(np.mean(tokens_arr))
    total_tokens = int(np.sum(tokens_arr))

    # Calculate generation calls (non-zero tokens)
    groq_generation_calls = sum(1 for t in tokens if t > 0)

    metrics = {
        "run_id": run_id,
        "variant": run.variant,
        "model_id": run.model_id,
        "total_questions": total_questions,
        "status_matches": status_matches,
        "status_match_rate": status_matches / total_questions if total_questions else 0.0,
        "retrieval_eligible_questions": retrieval_eligible,
        "evidence_hits_top5": evidence_hits_top5,
        "evidence_hit_rate": evidence_hits_top5 / retrieval_eligible if retrieval_eligible else 0.0,
        "required_evidence_covered": required_evidence_covered,
        "required_evidence_coverage_rate": required_evidence_covered / retrieval_eligible if retrieval_eligible else 0.0,
        "answerable_questions": answerable_count,
        "answerable_answered": answerable_answered,
        "answer_rate": answerable_answered / answerable_count if answerable_count else 0.0,
        "over_refusals": over_refusals,
        "over_refusal_rate": over_refusals / answerable_count if answerable_count else 0.0,
        "expected_insufficient": expected_insufficient,
        "actual_insufficient": actual_insufficient,
        "appropriate_abstention_rate": actual_insufficient / expected_insufficient if expected_insufficient else 0.0,
        "expected_clarify": expected_clarify,
        "actual_clarify": actual_clarify,
        "appropriate_clarification_rate": actual_clarify / expected_clarify if expected_clarify else 0.0,
        "expected_scope": expected_scope,
        "actual_scope": actual_scope,
        "appropriate_scope_refusal_rate": actual_scope / expected_scope if expected_scope else 0.0,
        "processing_errors": processing_errors,
        "processing_error_rate": processing_errors / total_questions if total_questions else 0.0,
        "mean_latency": mean_latency,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "mean_tokens": mean_tokens,
        "total_tokens": total_tokens,
        "groq_generation_calls": groq_generation_calls,
        "groq_calls_avoided_by_gate": groq_calls_avoided,
    }
    return metrics
