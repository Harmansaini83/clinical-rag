"""Deterministic evidence-sufficiency gate for clinical RAG.

Evaluates retrieval coverage, section relevance, and qualifier presence
before guarded generation, without calling any LLM.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from app.services.retrieval import load_passages_from_jsonl


@dataclass
class GateDecision:
    """Decision produced by the deterministic evidence gate."""
    passed: bool
    status: str      # 'proceed', 'insufficient_evidence', 'clarify'
    reason: str      # 'passed', 'no_passages_retrieved', 'relevant_section_not_found', etc.
    details: str = ""


# Section relevance mapping: keywords → target section_types
SECTION_TOPIC_MAPPING = {
    "contraindications": ["contraindications", "contraindicated"],
    "warnings_precautions": ["warning", "warnings", "precaution", "precautions"],
    "drug_interactions": ["interaction", "interactions", "interact", "interacts"],
    "renal_impairment": ["renal", "kidney", "egfr", "creatinine"],
}


# Recognized clinical qualifiers and their lexical patterns (normalized lowercase)
CLINICAL_QUALIFIERS = {
    "pregnancy": [r"\bpregnan", r"\bgestat", r"\btrimester", r"\bteratogen"],
    "pediatric": [r"\bpediatric", r"\bpaediatric", r"\bchild", r"\binfant", r"\badolescent"],
    "lactation": [r"\bbreastfeed", r"\bnursing\b", r"\blactat", r"\bbreast milk\b"],
    "hepatic": [r"\bhepatic", r"\bliver\b", r"\bcirrhosis"],
    "heart_failure": [r"\bheart failure\b", r"\bcongestive heart\b", r"\bchf\b"],
    "geriatric": [r"\bgeriatric", r"\baging\b", r"\b65 or greater\b", r"\belderly\b"],
}


def normalize_text(text: str) -> str:
    """Normalize text for consistent lexical matching."""
    t = text.lower()
    # Replace non-alphanumeric chars (except hyphens and periods inside numbers) with spaces
    t = re.sub(r"[^\w\s\.-]", " ", t)
    return " ".join(t.split())


def detect_relevant_sections(question: str) -> Set[str]:
    """Identify which section types are relevant to the question based on keywords."""
    norm_q = normalize_text(question)
    matched_sections: Set[str] = set()

    for sec_type, keywords in SECTION_TOPIC_MAPPING.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", norm_q):
                matched_sections.add(sec_type)
                break

    return matched_sections


def detect_question_qualifiers(question: str) -> List[str]:
    """Detect specific clinical qualifiers asked in the question."""
    norm_q = normalize_text(question)
    qualifiers_found = []

    for qual_name, patterns in CLINICAL_QUALIFIERS.items():
        for pat in patterns:
            if re.search(pat, norm_q):
                qualifiers_found.append(qual_name)
                break

    return qualifiers_found


def check_qualifier_in_text(qualifier: str, text: str) -> bool:
    """Check if a qualifier or its lexical variants appear in passage text."""
    norm_t = normalize_text(text)
    patterns = CLINICAL_QUALIFIERS.get(qualifier, [])
    for pat in patterns:
        if re.search(pat, norm_t):
            return True
    return False


def get_available_sections_for_label(label_id: str) -> Set[str]:
    """Return all section types present in the corpus for a given label_id."""
    try:
        passages = load_passages_from_jsonl(label_id=label_id)
        return {p.get("section_type", "") for p in passages if p.get("section_type")}
    except Exception:
        return set()


def evaluate_evidence_gate(
    question: str,
    retrieved_passages: List[Dict[str, Any]],
    label_id: str = "L001",
    available_sections: Optional[Set[str]] = None,
) -> GateDecision:
    """Evaluate deterministic evidence sufficiency gate (G1, G2, G3).

    Rules:
      G1 (retrieval coverage): If no passages retrieved -> insufficient_evidence
      G2 (section relevance): If question topic section is not in label corpus -> relevant_section_not_found
                              If question topic section was not retrieved -> relevant_section_not_retrieved
      G3 (qualifier presence): If a specific qualifier in question is absent from retrieved passages -> required_qualifier_not_found

    Returns:
      GateDecision(passed=True, status='proceed') or GateDecision(passed=False, status='insufficient_evidence')
    """
    # G1: Retrieval coverage
    if not retrieved_passages:
        return GateDecision(
            passed=False,
            status="insufficient_evidence",
            reason="no_passages_retrieved",
            details="No passages were retrieved for the specified drug query.",
        )

    # G2: Section relevance
    relevant_sections = detect_relevant_sections(question)
    if relevant_sections:
        if available_sections is None:
            available_sections = get_available_sections_for_label(label_id)

        # Check if the entire label corpus lacks all relevant section types
        if available_sections and not (relevant_sections & available_sections):
            return GateDecision(
                passed=False,
                status="insufficient_evidence",
                reason="relevant_section_not_found",
                details=f"Required section(s) {relevant_sections} do not exist in the indexed label for {label_id}.",
            )

        # Check if at least one retrieved passage matches a relevant section type
        retrieved_sections = {p.get("section_type", "") for p in retrieved_passages}
        if not (relevant_sections & retrieved_sections):
            return GateDecision(
                passed=False,
                status="insufficient_evidence",
                reason="relevant_section_not_retrieved",
                details=f"Retrieved passages contain sections {retrieved_sections}, none of which match question topic {relevant_sections}.",
            )

    # G3: Qualifier presence
    qualifiers = detect_question_qualifiers(question)
    combined_passage_text = " ".join(p.get("text", "") for p in retrieved_passages)

    for qual in qualifiers:
        if not check_qualifier_in_text(qual, combined_passage_text):
            return GateDecision(
                passed=False,
                status="insufficient_evidence",
                reason="required_qualifier_not_found",
                details=f"Question asked about qualifier '{qual}', but this topic is absent from all retrieved passages.",
            )

    return GateDecision(
        passed=True,
        status="proceed",
        reason="passed",
        details="All evidence gate rules (G1, G2, G3) satisfied.",
    )
