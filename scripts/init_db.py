"""Initialise the SQLite database: create all seven tables and seed from manifests.

Usage (from project root):
  python scripts/init_db.py
  python scripts/init_db.py --seed    # also seed documents and passages tables
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sqlmodel import Session, select

from app.config import settings
from app.db import create_db_and_tables, get_engine
from app.models import Document, Passage


def init_tables() -> None:
    """Create all seven database tables (idempotent)."""
    engine = get_engine()
    create_db_and_tables(engine)
    print(f"Database: {settings.data_dir / 'app.db'}")
    print("Tables created (or already exist): documents, passages, benchmark_questions,")
    print("  runs, results, judgments, llm_cache")


def seed_documents(engine) -> int:
    """Seed the documents table from data/manifest/documents.json."""
    manifest_path = settings.documents_manifest_path
    if not manifest_path.exists():
        print(f"  WARNING: {manifest_path} not found — skipping document seed.")
        return 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    seeded = 0
    with Session(engine) as session:
        for doc_id, entry in manifest.items():
            existing = session.exec(
                select(Document).where(Document.document_id == doc_id)
            ).first()
            if existing:
                continue
            doc = Document(
                document_id=doc_id,
                set_id=entry.get("set_id", ""),
                spl_version=str(entry.get("spl_version", "")),
                effective_date=str(entry.get("effective_date", "")),
                retrieved_at=entry.get("retrieved_at", ""),
                source_url=entry.get("source_url", ""),
                source_hash=entry.get("source_hash", ""),
                raw_file=entry.get("raw_file", ""),
                label_id=entry.get("label_id", ""),
                ingredient=entry.get("ingredient", ""),
                product_title=entry.get("product_title", "") or entry.get("xml_title_text", ""),
                labeler=entry.get("labeler", ""),
                formulation=entry.get("formulation", ""),
                route=entry.get("route", ""),
            )
            session.add(doc)
            seeded += 1
        session.commit()
    print(f"  Seeded {seeded} document record(s).")
    return seeded


def seed_passages(engine) -> int:
    """Seed the passages table from all data/passages/*.jsonl files."""
    jsonl_files = list(settings.passages_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("  No .jsonl passage files found — skipping passage seed.")
        return 0

    seeded = 0
    with Session(engine) as session:
        for jsonl_path in jsonl_files:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = json.loads(line)
                    existing = session.exec(
                        select(Passage).where(Passage.passage_id == p["passage_id"])
                    ).first()
                    if existing:
                        continue
                    passage = Passage(
                        passage_id=p["passage_id"],
                        document_id=p.get("document_id", ""),
                        label_id=p.get("label_id", ""),
                        section_type=p.get("section_type", ""),
                        section_title=p.get("section_title", ""),
                        section_path=p.get("section_path", ""),
                        text=p.get("text", ""),
                        word_count=int(p.get("word_count", 0)),
                        flags_json=json.dumps(p.get("flags", [])),
                    )
                    session.add(passage)
                    seeded += 1
        session.commit()
    print(f"  Seeded {seeded} passage record(s) from {len(jsonl_files)} file(s).")
    return seeded


def main():
    seed = "--seed" in sys.argv

    print("=== init_db: Creating database tables ===")
    init_tables()

    if seed:
        print("\n=== Seeding database from manifests ===")
        engine = get_engine()
        seed_documents(engine)
        seed_passages(engine)

    print("\nDone. Run 'python scripts/init_db.py --seed' to also load documents and passages.")


if __name__ == "__main__":
    main()
