"""Core question-answering workflow connecting retrieval, generation, and validation."""
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from groq import Groq

from app.config import settings
from app.services.generation import generate_grounded_response
from app.services.retrieval import retrieve_passages
from app.services.scope import check_scope, resolve_drug
from app.services.validation import ResponseValidationError, validate_generated_response

logger = logging.getLogger(__name__)

_DOCUMENTS_CACHE: Optional[Dict[str, Any]] = None


def get_documents_manifest(manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and cache the documents manifest."""
    global _DOCUMENTS_CACHE
    if _DOCUMENTS_CACHE is None:
        path = manifest_path or settings.documents_manifest_path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _DOCUMENTS_CACHE = json.load(f)
        else:
            _DOCUMENTS_CACHE = {}
    return _DOCUMENTS_CACHE


def enrich_evidence_items(
    evidence_items: List[Any],
    supplied_passages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach stored passage and document metadata to each evidence quote."""
    manifest = get_documents_manifest()
    passage_map = {p["passage_id"]: p for p in supplied_passages}

    enriched = []
    for item in evidence_items:
        pid = item.passage_id
        quote = item.quote
        p_data = passage_map.get(pid, {})
        doc_id = p_data.get("document_id", "")
        doc_info = manifest.get(doc_id, {})

        enriched.append(
            {
                "passage_id": pid,
                "quote": quote,
                "section_type": p_data.get("section_type", ""),
                "section_title": p_data.get("section_title", ""),
                "section_path": p_data.get("section_path", ""),
                "source_url": doc_info.get("source_url", ""),
                "label_version": str(doc_info.get("spl_version", "")),
                "effective_date": str(doc_info.get("effective_date", "")),
            }
        )
    return enriched


def answer_question(
    question: str,
    groq_client: Optional[Groq] = None,
) -> Dict[str, Any]:
    """Execute the clinical RAG question-answering workflow for a query.

    Flow:
      1. Input validation
      2. Scope check
      3. Drug lookup
      4. Retrieval filtered to label_id
      5. Guarded generation using Groq
      6. Validation
      7. Enriched response
    """
    req_id = str(uuid.uuid4())
    base_response = {
        "request_id": req_id,
        "model": settings.groq_model,
        "prompt_version": settings.prompt_version,
        "evidence": [],
        "clarification_question": "",
        "limitations": [],
        "latency_seconds": 0.0,
        "tokens_used": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    # 1. Input validation
    clean_q = " ".join((question or "").split())
    if not clean_q:
        return {
            **base_response,
            "status": "clarify",
            "answer": "",
            "clarification_question": "Please provide a clinical question regarding a covered medication.",
        }

    # 2. Scope check
    scope_result = check_scope(clean_q)
    if scope_result is not None:
        return {
            **base_response,
            **scope_result,
        }

    # 3. Drug lookup
    drug_res = resolve_drug(clean_q)
    if drug_res["status"] in ("clarify", "drug_not_covered"):
        return {
            **base_response,
            "status": drug_res["status"],
            "answer": drug_res.get("answer", ""),
            "clarification_question": drug_res.get("clarification_question", ""),
            "limitations": drug_res.get("limitations", []),
        }

    label_id = drug_res["label_id"]

    # 4. Retrieval filtered to applicable label
    try:
        passages = retrieve_passages(
            query_text=clean_q,
            n_results=settings.retrieval_top_k,
            label_id=label_id,
        )
    except Exception as e:
        logger.exception("Retrieval error during answer_question: %s", e)
        return {
            **base_response,
            "status": "processing_error",
            "error_category": "retrieval_error",
            "error": str(e),
            "answer": "An error occurred while retrieving relevant label passages.",
        }

    if not passages:
        return {
            **base_response,
            "status": "insufficient_evidence",
            "answer": f"No relevant label passages were found for {drug_res['drug']} in the corpus.",
            "limitations": ["No passages retrieved matching query."],
        }

    # 5. Guarded generation using Groq
    try:
        generated_data, usage, latency = generate_grounded_response(
            question=clean_q,
            passages=passages,
            client=groq_client,
        )
    except Exception as e:
        logger.exception("Generation error during answer_question: %s", e)
        return {
            **base_response,
            "status": "processing_error",
            "error_category": "provider_failure",
            "error": str(e),
            "answer": "An error occurred while calling the language model provider.",
        }

    # 6. Strict validation
    try:
        validated = validate_generated_response(
            raw_output=generated_data,
            supplied_passages=passages,
        )
    except ResponseValidationError as e:
        logger.warning("Response validation failed: %s (category: %s)", e.message, e.category)
        return {
            **base_response,
            "status": "processing_error",
            "error_category": e.category,
            "error": e.message,
            "answer": f"Output validation failed: {e.message}",
            "latency_seconds": latency,
            "tokens_used": usage,
        }

    # 7. Enrich evidence with stored metadata
    enriched_evidence = enrich_evidence_items(
        evidence_items=validated.evidence,
        supplied_passages=passages,
    )

    return {
        **base_response,
        "status": validated.status,
        "answer": validated.answer,
        "evidence": enriched_evidence,
        "clarification_question": validated.clarification_question,
        "limitations": validated.limitations,
        "latency_seconds": round(latency, 3),
        "tokens_used": usage,
    }
