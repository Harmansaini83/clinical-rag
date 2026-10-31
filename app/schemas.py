"""Pydantic request/response schemas for the FastAPI endpoints."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


MAX_QUESTION_LENGTH = 1000


class AskRequest(BaseModel):
    question: str = Field(..., description="Clinical question about a covered medication label.")

    @field_validator("question")
    @classmethod
    def question_must_be_nonempty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question must not be empty.")
        if len(stripped) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Question exceeds maximum allowed length of {MAX_QUESTION_LENGTH} characters."
            )
        return stripped


class EvidenceItemResponse(BaseModel):
    passage_id: str
    quote: str
    section_type: str = ""
    section_title: str = ""
    section_path: str = ""
    source_url: str = ""
    label_version: str = ""
    effective_date: str = ""


class AskResponse(BaseModel):
    status: str
    answer: str
    evidence: List[EvidenceItemResponse] = []
    clarification_question: str = ""
    limitations: List[str] = []
    request_id: str
    model: str
    prompt_version: str
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class SourceMetadata(BaseModel):
    label_id: str
    document_id: str
    ingredient: str
    product_title: str
    formulation: str
    route: str
    spl_version: str
    effective_date: str
    source_url: str


class HealthResponse(BaseModel):
    status: str
    app: str
    model: str
    corpus_labels: List[str]


class SourcesResponse(BaseModel):
    sources: List[SourceMetadata]
