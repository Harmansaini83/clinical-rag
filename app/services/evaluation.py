"""Evaluation harness for controlled RAG benchmarking (Variants A, B, C, and D).

Manages frozen benchmark loading, SQLite run and result persistence,
deterministic evidence gating, Groq caching, and resumable execution.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from groq import Groq

from sqlmodel import Session, select

from app.config import settings
from app.db import get_engine
from app.models import BenchmarkQuestion, Result, Run
from app.services.evidence import evaluate_evidence_gate
from app.services.generation import generate_grounded_response
from app.services.qa import enrich_evidence_items, get_documents_manifest
from app.services.retrieval import retrieve_passages, retrieve_passages_hybrid
from app.services.scope import check_scope, resolve_drug
from app.services.validation import ResponseValidationError, validate_generated_response

logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_benchmark_questions(benchmark_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load benchmark questions from JSONL file."""
    path = benchmark_path or (settings.data_dir / "benchmark" / "dev_questions.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Benchmark questions file not found: {path}")

    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def sync_benchmark_questions_to_db(engine, questions: List[Dict[str, Any]]) -> int:
    """Sync benchmark questions into SQLite benchmark_questions table."""
    seeded = 0
    with Session(engine) as session:
        for q in questions:
            qid = q["question_id"]
            existing = session.exec(
                select(BenchmarkQuestion).where(BenchmarkQuestion.question_id == qid)
            ).first()
            if existing:
                continue

            record = BenchmarkQuestion(
                question_id=qid,
                question=q.get("question", ""),
                category=q.get("category", ""),
                drug=q.get("drug", ""),
                expected_action=q.get("expected_action", ""),
                gold_answer_elements_json=json.dumps(q.get("gold_answer_elements", [])),
                gold_passage_ids_json=json.dumps(q.get("gold_passage_ids", [])),
                acceptable_alternative_passages_json=json.dumps(
                    q.get("acceptable_alternative_passages", [])
                ),
                evidence_group_id=q.get("evidence_group_id", ""),
                notes=q.get("notes", ""),
                split=q.get("split", "dev"),
            )
            session.add(record)
            seeded += 1
        session.commit()
    return seeded


def compute_run_id(
    variant: str,
    model: str,
    prompt_version: str,
    corpus_manifest_hash: str,
    benchmark_file_hash: str,
    config_dict: Dict[str, Any],
) -> str:
    """Compute deterministic run_id from configuration components."""
    payload = {
        "variant": variant,
        "model": model,
        "prompt_version": prompt_version,
        "corpus_manifest_hash": corpus_manifest_hash,
        "benchmark_file_hash": benchmark_file_hash,
        "config": config_dict,
    }
    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"run_{variant}_{digest}"


def get_or_create_run(
    engine,
    variant: str,
    model_id: str,
    prompt_version: str,
    corpus_manifest_hash: str,
    benchmark_file_hash: str,
    config_dict: Dict[str, Any],
) -> Run:
    """Get existing Run or create a new Run record in SQLite."""
    run_id = compute_run_id(
        variant=variant,
        model=model_id,
        prompt_version=prompt_version,
        corpus_manifest_hash=corpus_manifest_hash,
        benchmark_file_hash=benchmark_file_hash,
        config_dict=config_dict,
    )

    with Session(engine) as session:
        existing = session.exec(select(Run).where(Run.run_id == run_id)).first()
        if existing:
            return existing

        run = Run(
            run_id=run_id,
            variant=variant,
            model_id=model_id,
            prompt_version=prompt_version,
            corpus_manifest_hash=corpus_manifest_hash,
            config_json=json.dumps(
                {**config_dict, "benchmark_file_hash": benchmark_file_hash}, sort_keys=True
            ),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


def execute_variant_a(
    question: str,
    question_id: str,
    groq_client: Optional[Groq] = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute Variant A: Dense retrieval + Guarded generation."""
    clean_q = " ".join((question or "").split())
    base_response = {
        "status": "processing_error",
        "answer": "",
        "evidence": [],
        "clarification_question": "",
        "limitations": [],
        "latency_seconds": 0.0,
        "tokens_used": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error": "",
    }

    # 1. Scope check
    scope_result = check_scope(clean_q)
    if scope_result is not None:
        return {**base_response, **scope_result}, []

    # 2. Drug lookup
    drug_res = resolve_drug(clean_q)
    if drug_res["status"] in ("clarify", "drug_not_covered"):
        return {
            **base_response,
            "status": drug_res["status"],
            "answer": drug_res.get("answer", ""),
            "clarification_question": drug_res.get("clarification_question", ""),
            "limitations": drug_res.get("limitations", []),
        }, []

    label_id = drug_res["label_id"]

    # 3. Dense retrieval
    passages = retrieve_passages(
        query_text=clean_q,
        n_results=settings.retrieval_top_k,
        label_id=label_id,
    )

    if not passages:
        return {
            **base_response,
            "status": "insufficient_evidence",
            "answer": f"No relevant label passages were found for {drug_res['drug']} in the corpus.",
            "limitations": ["No passages retrieved matching query."],
        }, []

    # 4. Guarded generation using Groq
    t0 = time.perf_counter()
    try:
        generated_data, usage, latency = generate_grounded_response(
            question=clean_q,
            passages=passages,
            client=groq_client,
            question_id=question_id,
            variant="A",
            use_cache=use_cache,
        )
    except Exception as e:
        logger.exception("Generation error in Variant A for %s: %s", question_id, e)
        return {
            **base_response,
            "status": "processing_error",
            "error": str(e),
            "answer": "An error occurred while calling the language model provider.",
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }, passages

    # 5. Validation (with 1 retry on quote validation mismatch)
    try:
        validated = validate_generated_response(
            raw_output=generated_data,
            supplied_passages=passages,
        )
    except ResponseValidationError as e:
        logger.warning("Validation failed in Variant A for %s: %s. Attempting retry...", question_id, e)
        try:
            generated_data_retry, usage_retry, latency_retry = generate_grounded_response(
                question=clean_q,
                passages=passages,
                client=groq_client,
                question_id=question_id,
                variant="A",
                use_cache=False,
            )
            validated = validate_generated_response(
                raw_output=generated_data_retry,
                supplied_passages=passages,
            )
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0) + usage_retry.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0) + usage_retry.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0) + usage_retry.get("total_tokens", 0),
            }
            latency += latency_retry
        except ResponseValidationError as retry_err:
            logger.warning("Retry validation also failed in Variant A for %s: %s", question_id, retry_err)
            return {
                **base_response,
                "status": "processing_error",
                "error": retry_err.message,
                "answer": f"Output validation failed: {retry_err.message}",
                "latency_seconds": round(latency, 3),
                "tokens_used": usage,
            }, passages
        except Exception as retry_gen_err:
            return {
                **base_response,
                "status": "processing_error",
                "error": str(retry_gen_err),
                "answer": "Output validation retry failed.",
                "latency_seconds": round(latency, 3),
                "tokens_used": usage,
            }, passages

    enriched = enrich_evidence_items(validated.evidence, passages)
    return {
        **base_response,
        "status": validated.status,
        "answer": validated.answer,
        "evidence": enriched,
        "clarification_question": validated.clarification_question,
        "limitations": validated.limitations,
        "latency_seconds": round(latency, 3),
        "tokens_used": usage,
    }, passages


def execute_variant_b(
    question: str,
    question_id: str,
    groq_client: Optional[Groq] = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute Variant B: Hybrid retrieval (Dense + BM25 + RRF) + Guarded generation."""
    clean_q = " ".join((question or "").split())
    base_response = {
        "status": "processing_error",
        "answer": "",
        "evidence": [],
        "clarification_question": "",
        "limitations": [],
        "latency_seconds": 0.0,
        "tokens_used": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error": "",
    }

    # 1. Scope check
    scope_result = check_scope(clean_q)
    if scope_result is not None:
        return {**base_response, **scope_result}, []

    # 2. Drug lookup
    drug_res = resolve_drug(clean_q)
    if drug_res["status"] in ("clarify", "drug_not_covered"):
        return {
            **base_response,
            "status": drug_res["status"],
            "answer": drug_res.get("answer", ""),
            "clarification_question": drug_res.get("clarification_question", ""),
            "limitations": drug_res.get("limitations", []),
        }, []

    label_id = drug_res["label_id"]

    # 3. Hybrid retrieval (dense + BM25 + RRF)
    passages = retrieve_passages_hybrid(
        query_text=clean_q,
        label_id=label_id,
        dense_top_k=settings.retrieval_top_k,
        bm25_top_k=settings.retrieval_top_k,
        final_top_k=settings.retrieval_top_k,
        rrf_k=60,
    )

    if not passages:
        return {
            **base_response,
            "status": "insufficient_evidence",
            "answer": f"No relevant label passages were found for {drug_res['drug']} in the corpus.",
            "limitations": ["No passages retrieved matching query."],
        }, []

    # 4. Guarded generation using Groq
    t0 = time.perf_counter()
    try:
        generated_data, usage, latency = generate_grounded_response(
            question=clean_q,
            passages=passages,
            client=groq_client,
            question_id=question_id,
            variant="B",
            use_cache=use_cache,
        )
    except Exception as e:
        logger.exception("Generation error in Variant B for %s: %s", question_id, e)
        return {
            **base_response,
            "status": "processing_error",
            "error": str(e),
            "answer": "An error occurred while calling the language model provider.",
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }, passages

    # 5. Validation (with 1 retry on quote validation mismatch)
    try:
        validated = validate_generated_response(
            raw_output=generated_data,
            supplied_passages=passages,
        )
    except ResponseValidationError as e:
        logger.warning("Validation failed in Variant B for %s: %s. Attempting retry...", question_id, e)
        try:
            generated_data_retry, usage_retry, latency_retry = generate_grounded_response(
                question=clean_q,
                passages=passages,
                client=groq_client,
                question_id=question_id,
                variant="B",
                use_cache=False,
            )
            validated = validate_generated_response(
                raw_output=generated_data_retry,
                supplied_passages=passages,
            )
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0) + usage_retry.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0) + usage_retry.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0) + usage_retry.get("total_tokens", 0),
            }
            latency += latency_retry
        except ResponseValidationError as retry_err:
            logger.warning("Retry validation also failed in Variant B for %s: %s", question_id, retry_err)
            return {
                **base_response,
                "status": "processing_error",
                "error": retry_err.message,
                "answer": f"Output validation failed: {retry_err.message}",
                "latency_seconds": round(latency, 3),
                "tokens_used": usage,
            }, passages
        except Exception as retry_gen_err:
            return {
                **base_response,
                "status": "processing_error",
                "error": str(retry_gen_err),
                "answer": "Output validation retry failed.",
                "latency_seconds": round(latency, 3),
                "tokens_used": usage,
            }, passages

    enriched = enrich_evidence_items(validated.evidence, passages)
    return {
        **base_response,
        "status": validated.status,
        "answer": validated.answer,
        "evidence": enriched,
        "clarification_question": validated.clarification_question,
        "limitations": validated.limitations,
        "latency_seconds": round(latency, 3),
        "tokens_used": usage,
    }, passages


def execute_variant_c(
    question: str,
    question_id: str,
    groq_client: Optional[Groq] = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute Variant C: Dense retrieval + Local deterministic evidence gate + Guarded generation."""
    clean_q = " ".join((question or "").split())
    base_response = {
        "status": "processing_error",
        "answer": "",
        "evidence": [],
        "clarification_question": "",
        "limitations": [],
        "latency_seconds": 0.0,
        "tokens_used": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error": "",
    }

    # 1. Scope check
    scope_result = check_scope(clean_q)
    if scope_result is not None:
        return {**base_response, **scope_result}, []

    # 2. Drug lookup
    drug_res = resolve_drug(clean_q)
    if drug_res["status"] in ("clarify", "drug_not_covered"):
        return {
            **base_response,
            "status": drug_res["status"],
            "answer": drug_res.get("answer", ""),
            "clarification_question": drug_res.get("clarification_question", ""),
            "limitations": drug_res.get("limitations", []),
        }, []

    label_id = drug_res["label_id"]

    # 3. Dense retrieval
    passages = retrieve_passages(
        query_text=clean_q,
        n_results=settings.retrieval_top_k,
        label_id=label_id,
    )

    # 4. Local Deterministic Evidence Gate (G1, G2, G3)
    gate_decision = evaluate_evidence_gate(
        question=clean_q,
        retrieved_passages=passages,
        label_id=label_id,
    )

    if not gate_decision.passed:
        return {
            **base_response,
            "status": gate_decision.status,
            "answer": f"The indexed label for {drug_res['drug']} contains insufficient evidence to answer this question ({gate_decision.details}).",
            "limitations": [gate_decision.reason],
            "latency_seconds": 0.001,
        }, passages

    # 5. Guarded generation using Groq (only reached if gate passed)
    t0 = time.perf_counter()
    try:
        generated_data, usage, latency = generate_grounded_response(
            question=clean_q,
            passages=passages,
            client=groq_client,
            question_id=question_id,
            variant="C",
            use_cache=use_cache,
        )
    except Exception as e:
        logger.exception("Generation error in Variant C for %s: %s", question_id, e)
        return {
            **base_response,
            "status": "processing_error",
            "error": str(e),
            "answer": "An error occurred while calling the language model provider.",
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }, passages

    # 6. Validation (with 1 retry on quote validation mismatch)
    try:
        validated = validate_generated_response(
            raw_output=generated_data,
            supplied_passages=passages,
        )
    except ResponseValidationError as e:
        logger.warning("Validation failed in Variant C for %s: %s. Attempting retry...", question_id, e)
        try:
            generated_data_retry, usage_retry, latency_retry = generate_grounded_response(
                question=clean_q,
                passages=passages,
                client=groq_client,
                question_id=question_id,
                variant="C",
                use_cache=False,
            )
            validated = validate_generated_response(
                raw_output=generated_data_retry,
                supplied_passages=passages,
            )
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0) + usage_retry.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0) + usage_retry.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0) + usage_retry.get("total_tokens", 0),
            }
            latency += latency_retry
        except ResponseValidationError as retry_err:
            logger.warning("Retry validation also failed in Variant C for %s: %s", question_id, retry_err)
            return {
                **base_response,
                "status": "processing_error",
                "error": retry_err.message,
                "answer": f"Output validation failed: {retry_err.message}",
                "latency_seconds": round(latency, 3),
                "tokens_used": usage,
            }, passages
        except Exception as retry_gen_err:
            return {
                **base_response,
                "status": "processing_error",
                "error": str(retry_gen_err),
                "answer": "Output validation retry failed.",
                "latency_seconds": round(latency, 3),
                "tokens_used": usage,
            }, passages

    enriched = enrich_evidence_items(validated.evidence, passages)
    return {
        **base_response,
        "status": validated.status,
        "answer": validated.answer,
        "evidence": enriched,
        "clarification_question": validated.clarification_question,
        "limitations": validated.limitations,
        "latency_seconds": round(latency, 3),
        "tokens_used": usage,
    }, passages


def execute_variant_d(
    question: str,
    question_id: str,
    groq_client: Optional[Groq] = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute Variant D: Hybrid retrieval + Local deterministic evidence gate + Guarded generation."""
    clean_q = " ".join((question or "").split())
    base_response = {
        "status": "processing_error",
        "answer": "",
        "evidence": [],
        "clarification_question": "",
        "limitations": [],
        "latency_seconds": 0.0,
        "tokens_used": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error": "",
    }

    # 1. Scope check
    scope_result = check_scope(clean_q)
    if scope_result is not None:
        return {**base_response, **scope_result}, []

    # 2. Drug lookup
    drug_res = resolve_drug(clean_q)
    if drug_res["status"] in ("clarify", "drug_not_covered"):
        return {
            **base_response,
            "status": drug_res["status"],
            "answer": drug_res.get("answer", ""),
            "clarification_question": drug_res.get("clarification_question", ""),
            "limitations": drug_res.get("limitations", []),
        }, []

    label_id = drug_res["label_id"]

    # 3. Hybrid retrieval (dense + BM25 + RRF)
    passages = retrieve_passages_hybrid(
        query_text=clean_q,
        label_id=label_id,
        dense_top_k=settings.retrieval_top_k,
        bm25_top_k=settings.retrieval_top_k,
        final_top_k=settings.retrieval_top_k,
        rrf_k=60,
    )

    # 4. Local Deterministic Evidence Gate (G1, G2, G3)
    gate_decision = evaluate_evidence_gate(
        question=clean_q,
        retrieved_passages=passages,
        label_id=label_id,
    )

    if not gate_decision.passed:
        return {
            **base_response,
            "status": gate_decision.status,
            "answer": f"The indexed label for {drug_res['drug']} contains insufficient evidence to answer this question ({gate_decision.details}).",
            "limitations": [gate_decision.reason],
            "latency_seconds": 0.001,
        }, passages

    # 5. Guarded generation using Groq (only reached if gate passed)
    t0 = time.perf_counter()
    try:
        generated_data, usage, latency = generate_grounded_response(
            question=clean_q,
            passages=passages,
            client=groq_client,
            question_id=question_id,
            variant="D",
            use_cache=use_cache,
        )
    except Exception as e:
        logger.exception("Generation error in Variant D for %s: %s", question_id, e)
        return {
            **base_response,
            "status": "processing_error",
            "error": str(e),
            "answer": "An error occurred while calling the language model provider.",
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }, passages

    # 6. Validation (with 1 retry on quote validation mismatch)
    try:
        validated = validate_generated_response(
            raw_output=generated_data,
            supplied_passages=passages,
        )
    except ResponseValidationError as e:
        logger.warning("Validation failed in Variant D for %s: %s. Attempting retry...", question_id, e)
        try:
            generated_data_retry, usage_retry, latency_retry = generate_grounded_response(
                question=clean_q,
                passages=passages,
                client=groq_client,
                question_id=question_id,
                variant="D",
                use_cache=False,
            )
            validated = validate_generated_response(
                raw_output=generated_data_retry,
                supplied_passages=passages,
            )
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0) + usage_retry.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0) + usage_retry.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0) + usage_retry.get("total_tokens", 0),
            }
            latency += latency_retry
        except ResponseValidationError as retry_err:
            logger.warning("Retry validation also failed in Variant D for %s: %s", question_id, retry_err)
            return {
                **base_response,
                "status": "processing_error",
                "error": retry_err.message,
                "answer": f"Output validation failed: {retry_err.message}",
                "latency_seconds": round(latency, 3),
                "tokens_used": usage,
            }, passages
        except Exception as retry_gen_err:
            return {
                **base_response,
                "status": "processing_error",
                "error": str(retry_gen_err),
                "answer": "Output validation retry failed.",
                "latency_seconds": round(latency, 3),
                "tokens_used": usage,
            }, passages

    enriched = enrich_evidence_items(validated.evidence, passages)
    return {
        **base_response,
        "status": validated.status,
        "answer": validated.answer,
        "evidence": enriched,
        "clarification_question": validated.clarification_question,
        "limitations": validated.limitations,
        "latency_seconds": round(latency, 3),
        "tokens_used": usage,
    }, passages


def evaluate_benchmark(
    variant: str,
    engine=None,
    groq_client: Optional[Groq] = None,
    limit: Optional[int] = None,
    target_question_id: Optional[str] = None,
    benchmark_path: Optional[Path] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run evaluation for a specified variant across benchmark questions.

    Supported variants: 'A', 'B', 'C', 'D'.
    Resumable: Completed question results in the database are skipped.
    Errors on individual questions do not crash the run.
    """
    if variant not in ("A", "B", "C", "D"):
        raise ValueError(f"Invalid variant: {variant}. Supported: 'A', 'B', 'C', 'D'.")

    if engine is None:
        engine = get_engine()

    benchmark_file = benchmark_path or (settings.data_dir / "benchmark" / "dev_questions.jsonl")
    questions = load_benchmark_questions(benchmark_file)

    # Sync benchmark questions to DB
    sync_benchmark_questions_to_db(engine, questions)

    # Filter if target_question_id or limit specified
    if target_question_id:
        questions = [q for q in questions if q["question_id"] == target_question_id]
        if not questions:
            raise ValueError(f"Question ID {target_question_id} not found in benchmark.")
    elif limit is not None and limit > 0:
        questions = questions[:limit]

    # Compute hashes & run identity
    corpus_hash = compute_file_sha256(settings.documents_manifest_path)
    bench_hash = compute_file_sha256(benchmark_file)
    gen_config = {
        "temperature": settings.groq_temperature,
        "max_tokens": settings.groq_max_tokens,
        "reasoning_effort": settings.groq_reasoning_effort,
        "top_k": settings.retrieval_top_k,
        "retrieval_method": "hybrid" if variant in ("B", "D") else "dense",
        "evidence_gate": variant in ("C", "D"),
    }

    run_record = get_or_create_run(
        engine=engine,
        variant=variant,
        model_id=settings.groq_model,
        prompt_version=settings.prompt_version,
        corpus_manifest_hash=corpus_hash,
        benchmark_file_hash=bench_hash,
        config_dict=gen_config,
    )

    run_id = run_record.run_id
    logger.info("Starting evaluation for Run ID: %s (Variant %s, %d questions)", run_id, variant, len(questions))

    summary = {
        "run_id": run_id,
        "variant": variant,
        "total": len(questions),
        "executed": 0,
        "skipped": 0,
        "passed_status_match": 0,
        "errors": 0,
        "results": [],
    }

    for idx, q in enumerate(questions, start=1):
        qid = q["question_id"]
        q_text = q["question"]
        expected_action = q["expected_action"]

        # Check for existing result (Resumable!)
        with Session(engine) as session:
            existing_result = session.exec(
                select(Result).where(Result.run_id == run_id, Result.question_id == qid)
            ).first()

        if existing_result:
            summary["skipped"] += 1
            status_match = existing_result.status == expected_action
            if status_match:
                summary["passed_status_match"] += 1
            summary["results"].append({
                "question_id": qid,
                "status": existing_result.status,
                "expected": expected_action,
                "status_match": status_match,
                "latency_seconds": existing_result.latency_seconds,
                "cached_or_skipped": True,
            })
            continue

        # Execute selected variant
        try:
            if variant == "A":
                res, retrieved = execute_variant_a(
                    question=q_text,
                    question_id=qid,
                    groq_client=groq_client,
                    use_cache=use_cache,
                )
            elif variant == "B":
                res, retrieved = execute_variant_b(
                    question=q_text,
                    question_id=qid,
                    groq_client=groq_client,
                    use_cache=use_cache,
                )
            elif variant == "C":
                res, retrieved = execute_variant_c(
                    question=q_text,
                    question_id=qid,
                    groq_client=groq_client,
                    use_cache=use_cache,
                )
            elif variant == "D":
                res, retrieved = execute_variant_d(
                    question=q_text,
                    question_id=qid,
                    groq_client=groq_client,
                    use_cache=use_cache,
                )
        except Exception as e:
            logger.exception("Unexpected error executing %s on %s: %s", variant, qid, e)
            res = {
                "status": "processing_error",
                "answer": "",
                "error": str(e),
                "latency_seconds": 0.0,
                "tokens_used": {},
            }
            retrieved = []

        actual_status = res.get("status", "processing_error")
        retrieved_ids = [p["passage_id"] for p in retrieved]
        tokens = res.get("tokens_used", {})

        # Save result record to SQLite
        with Session(engine) as session:
            result_record = Result(
                run_id=run_id,
                question_id=qid,
                status=actual_status,
                answer=res.get("answer", ""),
                retrieved_ids_json=json.dumps(retrieved_ids),
                latency_seconds=float(res.get("latency_seconds", 0.0)),
                prompt_tokens=int(tokens.get("prompt_tokens", 0)),
                completion_tokens=int(tokens.get("completion_tokens", 0)),
                total_tokens=int(tokens.get("total_tokens", 0)),
                error=res.get("error", ""),
            )
            session.add(result_record)
            session.commit()

        summary["executed"] += 1
        status_match = actual_status == expected_action
        if status_match:
            summary["passed_status_match"] += 1
        if actual_status == "processing_error":
            summary["errors"] += 1

        summary["results"].append({
            "question_id": qid,
            "status": actual_status,
            "expected": expected_action,
            "status_match": status_match,
            "latency_seconds": res.get("latency_seconds", 0.0),
            "cached_or_skipped": False,
        })

    return summary
