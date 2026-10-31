"""Strict citation and quote validation service with no silent repair."""
import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class EvidenceItem(BaseModel):
    passage_id: str
    quote: str


class GeneratedResponse(BaseModel):
    status: Literal["answer", "clarify", "insufficient_evidence", "out_of_scope"]
    answer: str
    evidence: List[EvidenceItem] = []
    clarification_question: str = ""
    limitations: List[str] = []


class ResponseValidationError(Exception):
    """Raised when generated response fails schema, citation, or quote validation."""
    def __init__(self, message: str, category: str = "validation_failed"):
        super().__init__(message)
        self.message = message
        self.category = category


def normalize_text_for_quote(text: str) -> str:
    """Normalize text for contiguous quote matching.

    Performs:
      - NFKC unicode normalization
      - Curly quote normalization to straight quotes
      - En/em dash and minus sign normalization to hyphens
      - Superscript rendering fix: insert space when a digit is
        immediately followed by a lowercase letter (DailyMed XML
        artifact, e.g. "1.73 m 2is" -> "1.73 m 2 is")
      - Whitespace collapsing
    Case is strictly preserved.
    Does NOT change letter case, remove meaningful content, or allow
    ellipses / partial quotes.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    # Curly quotes -> straight
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    # Dashes -> hyphen
    t = t.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
    # Superscript rendering artifact: digit immediately followed by a
    # lowercase letter (no space) -- insert a space.
    # Handles the common DailyMed pattern "1.73 m 2is" -> "1.73 m 2 is".
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
    # Collapse whitespace
    return " ".join(t.split())


def _strip_matched_parens(text: str) -> str:
    """Remove matched parentheses that wrap single words/abbreviations.

    This allows a quote like "eGFR below 30" to match against stored
    text "(eGFR) below 30" where the parentheses are purely
    typographic.  Only removes parens that wrap a single contiguous
    token (no spaces inside).
    """
    return re.sub(r"\((\S+)\)", r"\1", text)


def validate_generated_response(
    raw_output: Any,
    supplied_passages: List[Dict[str, Any]],
) -> GeneratedResponse:
    """Validate model output strictly with no silent repair.

    Checks:
      1. Schema is valid JSON matching GeneratedResponse.
      2. Status is allowed.
      3. Every cited passage_id exists and was in supplied_passages.
      4. Every quote is a non-empty contiguous substring of its cited passage after normalisation.
      5. An 'answer' response contains at least one valid evidence item and non-empty text.
      6. A 'clarify' response contains a non-empty clarification_question.
    """
    # 1. Parse JSON if string
    if isinstance(raw_output, str):
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error during validation: %s", e)
            raise ResponseValidationError(f"Invalid JSON output: {e}", category="invalid_json")
    elif isinstance(raw_output, dict):
        data = raw_output
    else:
        raise ResponseValidationError(
            f"Expected dict or JSON string, received {type(raw_output).__name__}",
            category="invalid_schema",
        )

    # 2. Pydantic schema validation
    try:
        parsed = GeneratedResponse.model_validate(data)
    except ValidationError as e:
        logger.error("Schema validation failed: %s", e)
        raise ResponseValidationError(f"Schema validation failed: {e}", category="invalid_schema")

    # Map of supplied passages by ID
    supplied_map = {p["passage_id"]: p for p in supplied_passages}

    # 3. Specific status rules
    if parsed.status == "answer":
        if not parsed.answer or not parsed.answer.strip():
            raise ResponseValidationError(
                "Answer status requires non-empty answer text.",
                category="empty_answer",
            )
        if not parsed.evidence:
            raise ResponseValidationError(
                "Answer status requires at least one valid supporting evidence item.",
                category="missing_evidence",
            )

    elif parsed.status == "clarify":
        if not parsed.clarification_question or not parsed.clarification_question.strip():
            raise ResponseValidationError(
                "Clarify status requires a non-empty clarification_question.",
                category="missing_clarification_question",
            )

    # 4. Citation and Quote validation
    for idx, item in enumerate(parsed.evidence, start=1):
        pid = item.passage_id
        quote = item.quote

        # Check passage_id was supplied
        if pid not in supplied_map:
            raise ResponseValidationError(
                f"Cited passage_id '{pid}' was not among the supplied passages to the model.",
                category="unsupplied_citation",
            )

        # Check quote is non-empty
        quote_clean = quote.strip() if quote else ""
        if not quote_clean:
            raise ResponseValidationError(
                f"Evidence item {idx} (passage '{pid}') has an empty quote.",
                category="empty_quote",
            )

        # Contiguous substring check after normalization
        stored_passage_text = supplied_map[pid]["text"]
        norm_passage = normalize_text_for_quote(stored_passage_text)
        norm_quote = normalize_text_for_quote(quote_clean)

        if norm_quote not in norm_passage:
            # Tolerant fallback: strip matched single-token parentheses
            # from both sides (handles "(eGFR)" vs "eGFR").
            norm_passage_stripped = _strip_matched_parens(norm_passage)
            norm_quote_stripped = _strip_matched_parens(norm_quote)
            if norm_quote_stripped not in norm_passage_stripped:
                raise ResponseValidationError(
                    f"Evidence quote for passage '{pid}' does not match stored passage text after normalization.",
                    category="quote_mismatch",
                )

    return parsed
