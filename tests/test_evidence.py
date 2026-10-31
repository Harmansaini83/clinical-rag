"""Tests for deterministic evidence sufficiency gate (G1, G2, G3)."""
import pytest
from unittest.mock import patch

from app.services.evidence import evaluate_evidence_gate, GateDecision


def test_no_retrieved_passages_returns_insufficient_evidence():
    """G1 rule: No passages retrieved -> insufficient_evidence."""
    decision = evaluate_evidence_gate(
        question="What does the metformin label say about renal impairment?",
        retrieved_passages=[],
        label_id="L001",
    )
    assert not decision.passed
    assert decision.status == "insufficient_evidence"
    assert decision.reason == "no_passages_retrieved"


def test_relevant_section_missing_returns_insufficient_evidence():
    """G2 rule: Required section does not exist in label corpus -> relevant_section_not_found."""
    mock_passages = [
        {"passage_id": "L001-5.1-p01", "section_type": "warnings_precautions", "text": "Lactic acidosis warning"}
    ]
    # Label corpus only has warnings_precautions, but question asks about contraindications
    decision = evaluate_evidence_gate(
        question="What are the contraindications for this drug?",
        retrieved_passages=mock_passages,
        label_id="L001",
        available_sections={"warnings_precautions"},
    )
    assert not decision.passed
    assert decision.status == "insufficient_evidence"
    assert decision.reason == "relevant_section_not_found"


def test_relevant_section_not_retrieved_returns_insufficient_evidence():
    """G2 rule: Required section exists in corpus but was not retrieved -> relevant_section_not_retrieved."""
    # Label corpus has contraindications and warnings_precautions, but retrieval only returned warnings_precautions
    mock_passages = [
        {"passage_id": "L001-5.1-p01", "section_type": "warnings_precautions", "text": "Lactic acidosis warning"}
    ]
    decision = evaluate_evidence_gate(
        question="What are the contraindications for metformin?",
        retrieved_passages=mock_passages,
        label_id="L001",
        available_sections={"contraindications", "warnings_precautions"},
    )
    assert not decision.passed
    assert decision.status == "insufficient_evidence"
    assert decision.reason == "relevant_section_not_retrieved"


def test_required_qualifier_absent_returns_insufficient_evidence():
    """G3 rule: Question asks about qualifier (e.g. pregnancy) not present in passages -> required_qualifier_not_found."""
    mock_passages = [
        {
            "passage_id": "L001-2.3-p01",
            "section_type": "renal_impairment",
            "text": "Assess renal function before initiating metformin. Contraindicated if eGFR < 30.",
        }
    ]
    decision = evaluate_evidence_gate(
        question="What is the recommended metformin dosage during pregnancy?",
        retrieved_passages=mock_passages,
        label_id="L001",
        available_sections={"renal_impairment", "contraindications"},
    )
    assert not decision.passed
    assert decision.status == "insufficient_evidence"
    assert decision.reason == "required_qualifier_not_found"


def test_sufficient_renal_impairment_evidence_passes():
    """G1, G2, G3 all pass for renal impairment query with matching passage."""
    mock_passages = [
        {
            "passage_id": "L001-2.3-p01",
            "section_type": "renal_impairment",
            "text": "Metformin hydrochloride is contraindicated in patients with an eGFR below 30 mL/min/1.73 m2.",
        }
    ]
    decision = evaluate_evidence_gate(
        question="What are the renal impairment recommendations for metformin?",
        retrieved_passages=mock_passages,
        label_id="L001",
        available_sections={"renal_impairment", "contraindications", "warnings_precautions"},
    )
    assert decision.passed
    assert decision.status == "proceed"
    assert decision.reason == "passed"


def test_sufficient_interaction_evidence_passes():
    """G1, G2, G3 all pass for drug interaction query with matching passage."""
    mock_passages = [
        {
            "passage_id": "L001-7-p02",
            "section_type": "drug_interactions",
            "text": "Carbonic anhydrase inhibitors: Topiramate may increase the risk of lactic acidosis.",
        }
    ]
    decision = evaluate_evidence_gate(
        question="What is the drug interaction between metformin and topiramate?",
        retrieved_passages=mock_passages,
        label_id="L001",
        available_sections={"drug_interactions", "warnings_precautions"},
    )
    assert decision.passed
    assert decision.status == "proceed"
    assert decision.reason == "passed"


def test_gate_does_not_call_groq():
    """The evidence gate must be purely deterministic and never invoke Groq or any LLM."""
    with patch("app.services.generation.get_groq_client") as mock_groq, \
         patch("groq.Groq") as mock_groq_cls:
        
        mock_passages = [
            {"passage_id": "L001-2.3-p01", "section_type": "renal_impairment", "text": "eGFR below 30"}
        ]
        decision = evaluate_evidence_gate(
            question="What is the renal contraindication for metformin?",
            retrieved_passages=mock_passages,
            label_id="L001",
        )
        assert decision.passed
        mock_groq.assert_not_called()
        mock_groq_cls.assert_not_called()
