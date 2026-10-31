"""Ingest drug label passages into persistent ChromaDB and run test queries.

Usage (run from project root):
  python scripts/ingest.py                     # Index L001 and run 5 test queries
  python scripts/ingest.py index [L001]        # Index passages for a label
  python scripts/ingest.py test                # Run 5 standard retrieval test queries
  python scripts/ingest.py query "<text>"      # Run an ad-hoc test query
"""
import sys
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.config import settings
from app.services.retrieval import (
    get_chroma_client,
    get_or_create_collection,
    index_passages,
    load_passages_from_jsonl,
    retrieve_passages,
)

TEST_QUERIES = [
    "metformin contraindications",
    "lactic acidosis warning",
    "metformin vitamin B12 deficiency",
    "drug interactions with metformin",
    "renal impairment dosing recommendations",
]


def preview_text(text: str, max_chars: int = 140) -> str:
    """Return a single-line collapsed preview of text."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rstrip() + "..."


def run_indexing(label_id: str = "L001") -> int:
    """Index passages from data/passages/<label_id>.jsonl into persistent ChromaDB."""
    jsonl_path = settings.passages_dir / f"{label_id}.jsonl"
    print(f"Reading authoritative passages from: {jsonl_path.relative_to(ROOT)}")
    passages = load_passages_from_jsonl(jsonl_path)
    print(f"Loaded {len(passages)} passages.")

    client = get_chroma_client()
    collection = get_or_create_collection(client=client)

    print(f"Target ChromaDB directory: {settings.chroma_dir.relative_to(ROOT)}")
    print(f"Collection: {collection.name} (distance: {settings.chroma_distance_fn})")
    print("Computing embeddings and indexing records...")

    count = index_passages(passages, collection=collection)
    total_in_coll = collection.count()
    print(f"Successfully indexed {count} passages. Total records in collection: {total_in_coll}\n")
    return count


def run_retrieval_tests(label_id: str = "L001", top_k: int = 5) -> None:
    """Run the 5 required test queries and print formatted results."""
    print("=" * 80)
    print(f"RUNNING RETRIEVAL TEST QUERIES (Label: {label_id}, Top-{top_k})")
    print("=" * 80)

    client = get_chroma_client()
    collection = get_or_create_collection(client=client)

    for i, q in enumerate(TEST_QUERIES, start=1):
        print(f"\nQuery {i}: \"{q}\"")
        print("-" * 80)
        results = retrieve_passages(
            query_text=q,
            n_results=top_k,
            label_id=label_id,
            collection=collection,
        )

        if not results:
            print("  No matching passages retrieved.")
            continue

        for rank, r in enumerate(results, start=1):
            pid = r["passage_id"]
            stype = r["section_type"]
            stitle = r["section_title"]
            preview = preview_text(r["text"])
            dist = r["distance"]
            print(f"  [{rank}] passage_id:    {pid}")
            print(f"      section_type:  {stype}")
            print(f"      section_title: {stitle}")
            print(f"      distance:      {dist:.4f}")
            print(f"      preview:       {preview}")
    print("\n" + "=" * 80)


def run_adhoc_query(query: str, label_id: str = "L001", top_k: int = 5) -> None:
    """Execute a single ad-hoc query."""
    print(f"\nQuery: \"{query}\" (Label: {label_id}, Top-{top_k})")
    print("-" * 80)
    results = retrieve_passages(query_text=query, n_results=top_k, label_id=label_id)
    for rank, r in enumerate(results, start=1):
        print(f"  [{rank}] {r['passage_id']} | {r['section_type']} | {r['section_title']} (dist: {r['distance']:.4f})")
        print(f"      {preview_text(r['text'], 160)}")


def main(argv):
    if len(argv) == 1:
        run_indexing("L001")
        run_retrieval_tests("L001", top_k=5)
        return 0
    cmd = argv[1].lower()
    if cmd == "index":
        lid = argv[2] if len(argv) > 2 else "L001"
        run_indexing(lid)
    elif cmd == "test":
        lid = argv[2] if len(argv) > 2 else "L001"
        run_retrieval_tests(lid, top_k=5)
    elif cmd == "query" and len(argv) > 2:
        run_adhoc_query(" ".join(argv[2:]), label_id="L001", top_k=5)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
