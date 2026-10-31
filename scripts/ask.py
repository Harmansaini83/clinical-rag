"""Terminal execution script for the clinical RAG question answering pipeline.

Usage:
  python scripts/ask.py "<clinical question>"
"""
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.services.qa import answer_question


def run_cli(question: str) -> None:
    print("=" * 80)
    print("CLINICAL AI ASSISTANT — RETRIEVAL-AUGMENTED GENERATION")
    print("Prototype using openai/gpt-oss-120b via Groq | Academic Evaluation Only")
    print("=" * 80)
    print(f"QUESTION: {question}\n")

    result = answer_question(question)

    status = result.get("status")
    print(f"STATUS: {status.upper()}")
    print("-" * 80)

    if status == "answer":
        print(f"ANSWER:\n{result.get('answer')}\n")
        evidence = result.get("evidence", [])
        print(f"SUPPORTING EVIDENCE ({len(evidence)} citation(s)):")
        for i, ev in enumerate(evidence, start=1):
            print(f"\n  [{i}] Passage ID:   {ev.get('passage_id')}")
            print(f"      Section:      {ev.get('section_title')} ({ev.get('section_type')})")
            print(f"      Path:         {ev.get('section_path')}")
            print(f"      Exact Quote:  \"{ev.get('quote')}\"")
            print(f"      Source URL:   {ev.get('source_url')}")
            print(f"      SPL Version:  v{ev.get('label_version')} (Effective: {ev.get('effective_date')})")

    elif status == "clarify":
        print(f"CLARIFICATION NEEDED:")
        print(f"  {result.get('clarification_question')}\n")

    elif status == "insufficient_evidence":
        print(f"INSUFFICIENT EVIDENCE IN CORPUS:")
        print(f"  {result.get('answer')}\n")

    elif status == "out_of_scope":
        print(f"OUT OF SCOPE REQUEST:")
        print(f"  {result.get('answer')}")
        if result.get("reason"):
            print(f"  Category: {result.get('reason')}\n")

    elif status == "drug_not_covered":
        print(f"DRUG NOT COVERED:")
        print(f"  {result.get('answer')}\n")

    elif status == "processing_error":
        print(f"PROCESSING ERROR:")
        print(f"  Category: {result.get('error_category')}")
        print(f"  Details:  {result.get('error')}\n")

    limitations = result.get("limitations")
    if limitations:
        print(f"LIMITATIONS / NOTES:")
        for lim in limitations:
            print(f"  - {lim}")

    print("-" * 80)
    tokens = result.get("tokens_used", {})
    prompt_toks = tokens.get("prompt_tokens", 0)
    comp_toks = tokens.get("completion_tokens", 0)
    tot_toks = tokens.get("total_tokens", 0)
    latency = result.get("latency_seconds", 0.0)
    print(f"PERFORMANCE & METADATA:")
    print(f"  Model:          {result.get('model')}")
    print(f"  Prompt Version: {result.get('prompt_version')}")
    print(f"  Request ID:     {result.get('request_id')}")
    print(f"  Latency:        {latency:.3f} s")
    print(f"  Tokens:         prompt={prompt_toks}, completion={comp_toks}, total={tot_toks}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ask.py \"<question>\"")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    run_cli(q)


if __name__ == "__main__":
    main()
