"""Local explainable scope checker and drug alias resolver."""
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set

from app.config import settings


# Out-of-scope regex patterns with category reasons
OUT_OF_SCOPE_PATTERNS = [
    # Individualised prescribing / dosing
    (re.compile(r"\bwhat\s+dose\s+should\b", re.IGNORECASE), "Individualised dose calculation / recommendation"),
    (re.compile(r"\bdose\s+for\s+my\s+patient\b", re.IGNORECASE), "Patient-specific dosing request"),
    (re.compile(r"\bmy\s+patient\b", re.IGNORECASE), "Patient-specific clinical advice request"),
    (re.compile(r"\bcalculate\s+(?:the\s+)?dose\b", re.IGNORECASE), "Dose calculation request"),
    (re.compile(r"\bhow\s+much\s+(?:should|to)\s+(?:i|we|my\s+patient)\s+(?:take|give)\b", re.IGNORECASE), "Individualised dose recommendation"),
    (re.compile(r"\bprescribe\b", re.IGNORECASE), "Prescribing recommendation"),

    # Diagnosis requests
    (re.compile(r"\bdiagnos(?:e|is|ing|ed)\b", re.IGNORECASE), "Diagnosis request"),
    (re.compile(r"\bdo\s+i\s+have\b", re.IGNORECASE), "Personal diagnosis request"),
    (re.compile(r"\bdoes\s+(?:my|the)\s+patient\s+have\b", re.IGNORECASE), "Patient diagnosis request"),

    # Treatment selection
    (re.compile(r"\bwhich\s+drug\s+should\s+(?:i|we|my\s+patient)\s+(?:take|use)\b", re.IGNORECASE), "Treatment selection request"),
    (re.compile(r"\bwhat\s+should\s+(?:i|we)\s+prescribe\b", re.IGNORECASE), "Treatment selection request"),
    (re.compile(r"\bhow\s+should\s+i\s+treat\s+(?:my\s+patient|this\s+patient)\b", re.IGNORECASE), "Patient treatment request"),

    # Emergency or triage
    (re.compile(r"\bemergency\b", re.IGNORECASE), "Emergency assistance request"),
    (re.compile(r"\btriage\b", re.IGNORECASE), "Clinical triage request"),
]


def check_scope(question: str) -> Optional[Dict[str, Any]]:
    """Determine whether a question is out of scope.

    Returns an out_of_scope dictionary if triggered, otherwise None.
    """
    cleaned = " ".join(question.split())
    for pattern, reason in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(cleaned):
            return {
                "status": "out_of_scope",
                "answer": (
                    "This request is out of scope. The assistant provides evidence-based DailyMed "
                    "label information only and cannot provide individualized dosing, prescribing, "
                    "diagnosis, treatment selection, or emergency triage."
                ),
                "reason": reason,
                "evidence": [],
                "clarification_question": "",
                "limitations": [
                    "Individual patient management, dose calculation, diagnosis, and triage are strictly out of scope."
                ],
            }
    return None


def load_aliases(aliases_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the drug aliases registry from disk."""
    path = aliases_path or settings.aliases_path
    if not path.exists():
        raise FileNotFoundError(f"Aliases registry not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_drug(question: str, aliases_path: Optional[Path] = None) -> Dict[str, Any]:
    """Resolve drug mentioned in the query case-insensitively using aliases.json.

    Returns a dict with:
      - status: 'ok' | 'clarify' | 'drug_not_covered'
      - drug: canonical drug name or None
      - label_id: resolved label ID or None
      - clarification_question: text if clarify
      - answer: text if drug_not_covered
    """
    registry = load_aliases(aliases_path)
    normalized_q = " ".join(question.lower().split())

    # Find which canonical drug entries match the question
    matched_drug_keys: Set[str] = set()

    for drug_key, data in registry.items():
        aliases = data.get("aliases", [drug_key])
        # Sort aliases by length descending so longer phrases match first
        sorted_aliases = sorted(aliases, key=lambda a: len(a), reverse=True)
        for alias in sorted_aliases:
            alias_norm = " ".join(alias.lower().split())
            pattern = r"\b" + re.escape(alias_norm) + r"\b"
            if re.search(pattern, normalized_q):
                matched_drug_keys.add(drug_key)
                break

    if len(matched_drug_keys) == 0:
        return {
            "status": "clarify",
            "clarification_question": "Please specify which drug label you are inquiring about (e.g., metformin).",
            "drug": None,
            "label_id": None,
            "evidence": [],
            "limitations": [],
            "answer": "",
        }

    if len(matched_drug_keys) > 1:
        matched_names = [registry[k].get("canonical_name", k) for k in matched_drug_keys]
        return {
            "status": "clarify",
            "clarification_question": (
                f"Multiple drugs ({', '.join(matched_names)}) were identified in your query. "
                "Please ask about one specific drug at a time."
            ),
            "drug": None,
            "label_id": None,
            "evidence": [],
            "limitations": [],
            "answer": "",
        }

    key = next(iter(matched_drug_keys))
    drug_info = registry[key]
    canonical_name = drug_info.get("canonical_name", key)
    label_ids = drug_info.get("label_ids", [])

    if not label_ids:
        return {
            "status": "drug_not_covered",
            "answer": (
                f"The drug '{canonical_name}' is recognized, but its label is not currently "
                "included in the verified corpus."
            ),
            "drug": canonical_name,
            "label_id": None,
            "evidence": [],
            "clarification_question": "",
            "limitations": [
                f"Corpus limitation: {canonical_name} label is not ingested."
            ],
        }

    # For Milestone 1, we use the first matching label ID (e.g. L001)
    return {
        "status": "ok",
        "drug": canonical_name,
        "label_id": label_ids[0],
    }
