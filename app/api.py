"""FastAPI router: /health, /sources, /ask endpoints."""
import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status

from app.config import settings
from app.schemas import AskRequest, AskResponse, HealthResponse, SourceMetadata, SourcesResponse
from app.services.qa import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()

# Statuses that map to a successful HTTP 200 response (workflow ran, may still be abstentions)
_OK_STATUSES = {"answer", "clarify", "insufficient_evidence", "out_of_scope", "drug_not_covered"}

# Statuses that map to a 502/503 upstream-error response
_PROVIDER_ERROR_STATUSES = {"processing_error"}


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return application health, model identifier, and covered corpus labels."""
    manifest = _load_manifest()
    corpus_labels = sorted(
        {entry.get("label_id", "") for entry in manifest.values() if entry.get("label_id")}
    )
    return HealthResponse(
        status="ok",
        app="clinical-rag-prototype",
        model=settings.groq_model,
        corpus_labels=corpus_labels,
    )


@router.get("/sources", response_model=SourcesResponse)
async def get_sources() -> SourcesResponse:
    """Return metadata for all verified corpus sources from the manifest."""
    manifest = _load_manifest()
    curation = _load_curation()

    sources: List[SourceMetadata] = []
    for doc_id, entry in manifest.items():
        label_id = entry.get("label_id") or curation.get(doc_id, {}).get("label_id", "")
        if not label_id:
            continue
        sources.append(
            SourceMetadata(
                label_id=label_id,
                document_id=doc_id,
                ingredient=entry.get("ingredient", ""),
                product_title=entry.get("product_title", "") or entry.get("xml_title_text", ""),
                formulation=entry.get("formulation", ""),
                route=entry.get("route", ""),
                spl_version=str(entry.get("spl_version", "")),
                effective_date=str(entry.get("effective_date", "")),
                source_url=entry.get("source_url", ""),
            )
        )
    return SourcesResponse(sources=sources)


@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(request: AskRequest) -> AskResponse:
    """Execute the clinical RAG question-answering workflow.

    Interactive questions are NOT persisted to the database by default.
    """
    try:
        result = answer_question(question=request.question)
    except Exception as exc:
        logger.exception("Unexpected error in /ask for question: %r", request.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred. Please try again.",
        ) from exc

    response_status = result.get("status", "processing_error")

    if response_status in _PROVIDER_ERROR_STATUSES:
        error_category = result.get("error_category", "unknown")
        if error_category in ("provider_failure", "retrieval_error"):
            http_code = status.HTTP_502_BAD_GATEWAY
        elif error_category in ("validation_failed",):
            http_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        else:
            http_code = status.HTTP_503_SERVICE_UNAVAILABLE

        raise HTTPException(
            status_code=http_code,
            detail={
                "status": response_status,
                "error_category": error_category,
                "message": result.get("answer", "Processing error occurred."),
                "request_id": result.get("request_id", ""),
            },
        )

    tokens = result.get("tokens_used", {})
    evidence = result.get("evidence", [])

    return AskResponse(
        status=response_status,
        answer=result.get("answer", ""),
        evidence=evidence,
        clarification_question=result.get("clarification_question", ""),
        limitations=result.get("limitations", []),
        request_id=result.get("request_id", ""),
        model=result.get("model", settings.groq_model),
        prompt_version=result.get("prompt_version", settings.prompt_version),
        latency_seconds=result.get("latency_seconds", 0.0),
        prompt_tokens=tokens.get("prompt_tokens", 0),
        completion_tokens=tokens.get("completion_tokens", 0),
        total_tokens=tokens.get("total_tokens", 0),
    )


# ---- helpers ----

def _load_manifest() -> Dict[str, Any]:
    path = settings.documents_manifest_path
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_curation() -> Dict[str, Any]:
    path = settings.manifest_dir / "curation.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
