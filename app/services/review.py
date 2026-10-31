"""Review export service for manual clinical assessment and validation."""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select
from app.db import get_engine
from app.models import BenchmarkQuestion, Result, Run

# Permitted values for reviewer fields
ALLOWED_CLAIM_SUPPORT = ["supported", "partial", "unsupported", "not_applicable"]
ALLOWED_CORRECT = ["yes", "no"]
ALLOWED_OVER_REFUSAL = ["yes", "no"]

REVIEW_COLUMNS = [
    "run_id",
    "variant",
    "question_id",
    "question",
    "category",
    "expected_action",
    "actual_status",
    "status_match",
    "answer",
    "retrieved_passage_ids",
    "gold_passage_ids",
    "evidence_hit",
    "required_evidence_covered",
    "cited_passage_ids",
    "quote_valid",
    "generated_claim_support",
    "correct",
    "over_refusal",
    "error_category",
    "reviewer_notes",
    "reviewer",
]


def generate_review_rows(run_id: str, engine=None) -> List[Dict[str, Any]]:
    """Build review data rows for a specified run."""
    if engine is None:
        engine = get_engine()

    with Session(engine) as session:
        run = session.exec(select(Run).where(Run.run_id == run_id)).first()
        if not run:
            raise ValueError(f"Run ID not found: {run_id}")

        results = session.exec(select(Result).where(Result.run_id == run_id)).all()
        questions = session.exec(select(BenchmarkQuestion)).all()

    if not results:
        raise ValueError(f"No results found for run ID: {run_id}")

    q_map = {q.question_id: q for q in questions}
    rows = []

    for r in results:
        q = q_map.get(r.question_id)
        if not q:
            continue

        gold_ids = json.loads(q.gold_passage_ids_json or "[]")
        retrieved_ids = json.loads(r.retrieved_ids_json or "[]")

        # Determine evidence hit and required coverage
        if gold_ids:
            gold_set = set(gold_ids)
            retrieved_set = set(retrieved_ids)
            evidence_hit = "yes" if any(gid in retrieved_set for gid in gold_ids) else "no"
            req_covered = "yes" if gold_set.issubset(retrieved_set) else "no"
        else:
            evidence_hit = "not_applicable"
            req_covered = "not_applicable"

        status_match = "yes" if r.status == q.expected_action else "no"

        # Auto-detect error_category for non-matches
        error_category = ""
        if r.status != q.expected_action:
            if r.status == "processing_error":
                error_category = "processing_error"
            elif q.expected_action == "answer" and r.status == "insufficient_evidence":
                error_category = "over_refusal"
            elif q.expected_action == "clarify" and r.status == "insufficient_evidence":
                error_category = "missed_clarification"
            elif q.expected_action == "insufficient_evidence" and r.status == "answer":
                error_category = "under_refusal"
            elif q.expected_action == "out_of_scope" and r.status != "out_of_scope":
                error_category = "incorrect_scope_refusal"
            elif evidence_hit == "no":
                error_category = "retrieval_miss"
            else:
                error_category = "other"

        row = {
            "run_id": run.run_id,
            "variant": run.variant,
            "question_id": r.question_id,
            "question": q.question,
            "category": q.category,
            "expected_action": q.expected_action,
            "actual_status": r.status,
            "status_match": status_match,
            "answer": r.answer,
            "retrieved_passage_ids": ";".join(retrieved_ids),
            "gold_passage_ids": ";".join(gold_ids),
            "evidence_hit": evidence_hit,
            "required_evidence_covered": req_covered,
            "cited_passage_ids": "",  # Populated from answer or left for reviewer inspection
            "quote_valid": "yes" if r.status == "answer" else ("not_applicable" if r.status != "processing_error" else "no"),
            "generated_claim_support": "",  # Blank for manual completion
            "correct": "",                  # Blank for manual completion
            "over_refusal": "",            # Blank for manual completion
            "error_category": error_category,
            "reviewer_notes": "",           # Blank for manual completion
            "reviewer": "",                 # Blank for manual completion
        }
        rows.append(row)

    return rows


def export_review_csv(run_ids: List[str], output_path: Path, engine=None) -> int:
    """Export benchmark evaluation results to CSV for manual blinded review."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for rid in run_ids:
        all_rows.extend(generate_review_rows(rid, engine=engine))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    return len(all_rows)
