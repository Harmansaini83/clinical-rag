"""Unit tests for response, citation, and quote validation."""
import pytest
from app.services.validation import (
    ResponseValidationError,
    validate_generated_response,
)

SAMPLE_SUPPLIED_PASSAGES = [
    {
        "passage_id": "L001-4-p01",
        "text": (
            "Metformin hydrochloride tablets are contraindicated in patients with:\n"
            "- Severe renal impairment (eGFR below 30 mL/min/1.73 m 2) [ see Warnings and Precautions (5.1) ].\n"
            "- Hypersensitivity to metformin.\n"
            "- Acute or chronic metabolic acidosis, including diabetic ketoacidosis, with or without coma."
        ),
        "section_type": "contraindications",
        "section_title": "4 CONTRAINDICATIONS",
    },
    {
        "passage_id": "L001-5.2-p01",
        "text": (
            "In metformin hydrochloride tablets clinical trials of 29-week duration, a decrease to "
            "subnormal levels of previously normal serum vitamin B 12levels was observed in approximately 7% of patients."
        ),
        "section_type": "warnings_precautions",
        "section_title": "5.2 Vitamin B 12Deficiency",
    },
]


def test_invalid_json_rejected():
    with pytest.raises(ResponseValidationError) as exc:
        validate_generated_response("Not a JSON string {", SAMPLE_SUPPLIED_PASSAGES)
    assert exc.value.category == "invalid_json"


def test_invalid_schema_rejected():
    # Missing required field 'status'
    bad_data = {"answer": "Some answer without status", "evidence": []}
    with pytest.raises(ResponseValidationError) as exc:
        validate_generated_response(bad_data, SAMPLE_SUPPLIED_PASSAGES)
    assert exc.value.category == "invalid_schema"


def test_unknown_passage_id_rejected():
    data = {
        "status": "answer",
        "answer": "Contraindicated in severe renal impairment.",
        "evidence": [
            {
                "passage_id": "L999-NONEXISTENT",
                "quote": "Severe renal impairment",
            }
        ],
        "clarification_question": "",
        "limitations": [],
    }
    with pytest.raises(ResponseValidationError) as exc:
        validate_generated_response(data, SAMPLE_SUPPLIED_PASSAGES)
    assert exc.value.category == "unsupplied_citation"


def test_unsupplied_passage_id_rejected():
    # Passage ID exists in real world but was not supplied to this request
    data = {
        "status": "answer",
        "answer": "Contraindicated in severe renal impairment.",
        "evidence": [
            {
                "passage_id": "L001-8.6-p01",  # Not in SAMPLE_SUPPLIED_PASSAGES
                "quote": "Metformin is substantially excreted by the kidney",
            }
        ],
        "clarification_question": "",
        "limitations": [],
    }
    with pytest.raises(ResponseValidationError) as exc:
        validate_generated_response(data, SAMPLE_SUPPLIED_PASSAGES)
    assert exc.value.category == "unsupplied_citation"


def test_quote_mismatch_rejected():
    data = {
        "status": "answer",
        "answer": "Metformin is contraindicated in mild renal impairment.",
        "evidence": [
            {
                "passage_id": "L001-4-p01",
                "quote": "Mild renal impairment is completely prohibited",  # Hallucinated quote
            }
        ],
        "clarification_question": "",
        "limitations": [],
    }
    with pytest.raises(ResponseValidationError) as exc:
        validate_generated_response(data, SAMPLE_SUPPLIED_PASSAGES)
    assert exc.value.category == "quote_mismatch"


def test_answer_without_evidence_rejected():
    data = {
        "status": "answer",
        "answer": "Metformin should not be taken in acidosis.",
        "evidence": [],  # Empty evidence for an answer
        "clarification_question": "",
        "limitations": [],
    }
    with pytest.raises(ResponseValidationError) as exc:
        validate_generated_response(data, SAMPLE_SUPPLIED_PASSAGES)
    assert exc.value.category == "missing_evidence"


def test_valid_response_passes():
    data = {
        "status": "answer",
        "answer": (
            "Metformin hydrochloride tablets are contraindicated in patients with severe renal "
            "impairment (eGFR below 30 mL/min/1.73 m2), hypersensitivity, or metabolic acidosis."
        ),
        "evidence": [
            {
                "passage_id": "L001-4-p01",
                "quote": "Severe renal impairment (eGFR below 30 mL/min/1.73 m 2)",
            }
        ],
        "clarification_question": "",
        "limitations": [],
    }
    validated = validate_generated_response(data, SAMPLE_SUPPLIED_PASSAGES)
    assert validated.status == "answer"
    assert len(validated.evidence) == 1
    assert validated.evidence[0].passage_id == "L001-4-p01"


def test_insufficient_evidence_response_passes():
    data = {
        "status": "insufficient_evidence",
        "answer": "The supplied label passages do not contain information regarding pregnancy.",
        "evidence": [],
        "clarification_question": "",
        "limitations": ["Information not present in retrieved passages."],
    }
    validated = validate_generated_response(data, SAMPLE_SUPPLIED_PASSAGES)
    assert validated.status == "insufficient_evidence"
    assert len(validated.evidence) == 0


def test_clarify_response_requires_question():
    bad_clarify = {
        "status": "clarify",
        "answer": "",
        "evidence": [],
        "clarification_question": "",  # Empty
        "limitations": [],
    }
    with pytest.raises(ResponseValidationError) as exc:
        validate_generated_response(bad_clarify, SAMPLE_SUPPLIED_PASSAGES)
    assert exc.value.category == "missing_clarification_question"

    good_clarify = {
        "status": "clarify",
        "answer": "",
        "evidence": [],
        "clarification_question": "Which specific formulation are you referring to?",
        "limitations": [],
    }
    validated = validate_generated_response(good_clarify, SAMPLE_SUPPLIED_PASSAGES)
    assert validated.status == "clarify"
