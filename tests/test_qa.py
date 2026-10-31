"""Unit tests for the QA pipeline using mocked Groq clients (no network calls)."""
import json
import pytest
from groq import APITimeoutError

from app.services.qa import answer_question


class FakeChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})()


class FakeUsage:
    prompt_tokens = 120
    completion_tokens = 45
    total_tokens = 165


class FakeCompletion:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class FakeCompletionsService:
    def __init__(self, content: str = "", side_effect: Exception = None):
        self.content = content
        self.side_effect = side_effect

    def create(self, *args, **kwargs):
        if self.side_effect:
            raise self.side_effect
        return FakeCompletion(self.content)


class FakeGroqClient:
    def __init__(self, content: str = "", side_effect: Exception = None):
        self.chat = type(
            "Chat",
            (),
            {"completions": FakeCompletionsService(content=content, side_effect=side_effect)},
        )()


def test_valid_generated_response_passes_workflow():
    valid_payload = json.dumps(
        {
            "status": "answer",
            "answer": "Metformin is contraindicated in patients with severe renal impairment.",
            "evidence": [
                {
                    "passage_id": "L001-4-p01",
                    "quote": "Severe renal impairment (eGFR below 30 mL/min/1.73 m 2)",
                }
            ],
            "clarification_question": "",
            "limitations": [],
        }
    )
    fake_client = FakeGroqClient(content=valid_payload)
    res = answer_question("What does the metformin label say about contraindications?", groq_client=fake_client)

    assert res["status"] == "answer"
    assert "severe renal impairment" in res["answer"].lower()
    assert len(res["evidence"]) == 1
    ev = res["evidence"][0]
    assert ev["passage_id"] == "L001-4-p01"
    assert ev["section_title"] == "4 CONTRAINDICATIONS"
    assert "source_url" in ev and len(ev["source_url"]) > 0


def test_unknown_drug_returns_drug_not_covered():
    # Never contacts Groq
    res = answer_question("What does the insulin glargine label say about paediatric dosing?", groq_client=None)
    assert res["status"] == "drug_not_covered"
    assert "insulin glargine" in res["answer"].lower()


def test_out_of_scope_returns_out_of_scope():
    # Never contacts Groq
    res = answer_question("What dose should my patient receive?", groq_client=None)
    assert res["status"] == "out_of_scope"
    assert "out of scope" in res["answer"].lower()


def test_provider_timeout_returns_processing_error():
    fake_client = FakeGroqClient(side_effect=APITimeoutError("Groq request timed out"))
    res = answer_question("What does the metformin label say about contraindications?", groq_client=fake_client)

    assert res["status"] == "processing_error"
    assert res["error_category"] == "provider_failure"
    assert "error" in res


def test_invalid_model_json_returns_processing_error():
    fake_client = FakeGroqClient(content="Not a valid JSON response from model")
    res = answer_question("What does the metformin label say about contraindications?", groq_client=fake_client)

    assert res["status"] == "processing_error"
    assert res["error_category"] == "provider_failure"
