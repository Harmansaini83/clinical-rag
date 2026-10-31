"""SQLModel table definitions for the Clinical RAG application."""
import json
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(SQLModel, table=True):
    """Stored DailyMed label document metadata (from documents.json manifest)."""
    __tablename__ = "documents"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: str = Field(index=True, unique=True)
    set_id: str = ""
    spl_version: str = ""
    effective_date: str = ""
    retrieved_at: str = ""
    source_url: str = ""
    source_hash: str = ""
    raw_file: str = ""
    label_id: str = Field(index=True, default="")
    ingredient: str = ""
    product_title: str = ""
    labeler: str = ""
    formulation: str = ""
    route: str = ""
    created_at: datetime = Field(default_factory=_now)


class Passage(SQLModel, table=True):
    """Extracted label passage record."""
    __tablename__ = "passages"

    id: Optional[int] = Field(default=None, primary_key=True)
    passage_id: str = Field(index=True, unique=True)
    document_id: str = Field(index=True, default="")
    label_id: str = Field(index=True, default="")
    section_type: str = ""
    section_title: str = ""
    section_path: str = ""
    text: str = ""
    word_count: int = 0
    flags_json: str = "[]"   # JSON array of flag strings
    created_at: datetime = Field(default_factory=_now)


class BenchmarkQuestion(SQLModel, table=True):
    """Benchmark evaluation question record."""
    __tablename__ = "benchmark_questions"

    id: Optional[int] = Field(default=None, primary_key=True)
    question_id: str = Field(index=True, unique=True)
    question: str = ""
    category: str = ""
    drug: str = ""
    expected_action: str = ""
    gold_answer_elements_json: str = "[]"
    gold_passage_ids_json: str = "[]"
    acceptable_alternative_passages_json: str = "[]"
    evidence_group_id: str = ""
    notes: str = ""
    split: str = ""   # 'dev' or 'held_out'
    created_at: datetime = Field(default_factory=_now)


class Run(SQLModel, table=True):
    """Experiment run record (variant A/B/C/D configuration)."""
    __tablename__ = "runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, unique=True)
    variant: str = ""          # 'A', 'B', 'C', 'D'
    model_id: str = ""
    prompt_version: str = ""
    corpus_manifest_hash: str = ""
    config_json: str = "{}"
    created_at: datetime = Field(default_factory=_now)


class Result(SQLModel, table=True):
    """Experiment result for one question in one run."""
    __tablename__ = "results"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, default="")
    question_id: str = Field(index=True, default="")
    status: str = ""
    answer: str = ""
    retrieved_ids_json: str = "[]"
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""
    created_at: datetime = Field(default_factory=_now)


class Judgment(SQLModel, table=True):
    """Manual reviewer judgment for one result."""
    __tablename__ = "judgments"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, default="")
    question_id: str = Field(index=True, default="")
    correct: Optional[bool] = None
    citation_supported: Optional[bool] = None
    over_refusal: Optional[bool] = None
    notes: str = ""
    reviewer: str = ""
    created_at: datetime = Field(default_factory=_now)


class LLMCache(SQLModel, table=True):
    """Cached Groq generation response for benchmark runs."""
    __tablename__ = "llm_cache"

    id: Optional[int] = Field(default=None, primary_key=True)
    cache_key: str = Field(index=True, unique=True)
    model: str = ""
    prompt_version: str = ""
    variant: str = ""
    question_id: str = ""
    passage_ids_json: str = "[]"
    settings_json: str = "{}"
    response_json: str = "{}"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    original_latency_seconds: float = 0.0
    created_at: datetime = Field(default_factory=_now)
