"""Unit tests for local scope checking and drug alias resolution."""
import pytest
from app.services.scope import check_scope, resolve_drug


def test_diagnosis_request_out_of_scope():
    res = check_scope("Can you diagnose if my patient has lactic acidosis?")
    assert res is not None
    assert res["status"] == "out_of_scope"
    assert "diagnosis" in res["reason"].lower() or "patient" in res["reason"].lower()


def test_personalized_dosing_out_of_scope():
    res = check_scope("What dose should my patient receive?")
    assert res is not None
    assert res["status"] == "out_of_scope"

    res2 = check_scope("Calculate the dose of metformin for an elderly patient.")
    assert res2 is not None
    assert res2["status"] == "out_of_scope"


def test_normal_label_question_in_scope():
    res1 = check_scope("What does the metformin label say about renal impairment?")
    assert res1 is None

    res2 = check_scope("What does the metformin label say about vitamin B12 deficiency?")
    assert res2 is None

    res3 = check_scope("What are the contraindications listed in the metformin label?")
    assert res3 is None


def test_resolve_drug_covered():
    res = resolve_drug("What does the metformin label say about renal impairment?")
    assert res["status"] == "ok"
    assert res["drug"] == "metformin hydrochloride"
    assert res["label_id"] == "L001"

    # Test alias variation
    res2 = resolve_drug("Does metformin hcl interact with ranolazine?")
    assert res2["status"] == "ok"
    assert res2["label_id"] == "L001"


def test_resolve_drug_not_covered():
    res = resolve_drug("What does the insulin glargine label say about paediatric dosing?")
    assert res["status"] == "drug_not_covered"
    assert res["drug"] == "insulin glargine"
    assert res["label_id"] is None


def test_resolve_drug_missing_returns_clarify():
    res = resolve_drug("What are the contraindications and renal impairment dosing?")
    assert res["status"] == "clarify"
    assert "specify" in res["clarification_question"].lower()


def test_resolve_drug_multiple_incompatible_returns_clarify():
    res = resolve_drug("Compare contraindications of metformin and insulin glargine.")
    assert res["status"] == "clarify"
    assert "multiple" in res["clarification_question"].lower()
