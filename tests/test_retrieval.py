"""Tests for persistent ChromaDB indexing, BM25 retrieval, and Hybrid RRF retrieval."""
import tempfile
from pathlib import Path

import pytest
import chromadb

from app.config import settings
from app.services.retrieval import (
    BM25Index,
    get_bm25_index,
    get_or_create_collection,
    index_passages,
    load_passages_from_jsonl,
    reciprocal_rank_fusion,
    retrieve_passages,
    retrieve_passages_bm25,
    retrieve_passages_hybrid,
)


@pytest.fixture(scope="module")
def passages():
    jsonl_path = settings.passages_dir / "L001.jsonl"
    assert jsonl_path.exists(), f"L001.jsonl does not exist at {jsonl_path}"
    data = load_passages_from_jsonl(jsonl_path)
    return data


def test_load_passages_count_and_fields(passages):
    assert len(passages) == 11
    required_keys = {
        "passage_id",
        "document_id",
        "label_id",
        "section_type",
        "section_title",
        "section_path",
        "word_count",
        "text",
    }
    for p in passages:
        assert required_keys.issubset(p.keys()), f"Missing keys in passage {p.get('passage_id')}"
        assert p["label_id"] == "L001"
        assert len(p["text"].strip()) > 0
        assert p["word_count"] > 0


def test_chroma_indexing_and_retrieval(passages, tmp_path):
    # Test indexing in an isolated temporary ChromaDB
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = get_or_create_collection(client=client, collection_name="test_passages")

    count = index_passages(passages, collection=collection)
    assert count == 11
    assert collection.count() == 11

    # Verify contraindications query retrieves L001-4-p01 as top result
    results = retrieve_passages(
        query_text="metformin contraindications",
        n_results=1,
        label_id="L001",
        collection=collection,
    )
    assert len(results) == 1
    assert results[0]["passage_id"] == "L001-4-p01"
    assert results[0]["section_type"] == "contraindications"


def test_persistent_chroma_index_query():
    # Verify retrieval against the actual persistent chroma_db
    assert settings.chroma_dir.exists(), "Persistent chroma_db directory must exist"
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_collection(settings.chroma_collection_name)
    assert collection.count() >= 11

    results = retrieve_passages(
        query_text="metformin vitamin B12 deficiency",
        n_results=1,
        label_id="L001",
        collection=collection,
    )
    assert len(results) == 1
    assert results[0]["passage_id"] == "L001-5.2-p01"
    assert results[0]["section_type"] == "warnings_precautions"


# ==============================================================================
# BM25 & HYBRID RRF TESTS
# ==============================================================================

def test_bm25_index_loads_all_passages(passages):
    index = BM25Index(passages)
    assert index.corpus_size == 11
    assert len(index.doc_lens) == 11
    assert index.avg_doc_len > 0


def test_bm25_returns_relevant_passages_for_renal_query(passages):
    index = BM25Index(passages)
    results = index.search(query_text="renal impairment eGFR cutoff", n_results=3, label_id="L001")
    assert len(results) > 0
    # Renal section passages should rank highly
    retrieved_sections = {r["section_type"] for r in results}
    assert "renal_impairment" in retrieved_sections or "contraindications" in retrieved_sections
    for r in results:
        assert r["retrieval_source"] == "bm25"
        assert r["bm25_score"] > 0


def test_bm25_respects_label_filtering(passages):
    index = BM25Index(passages)
    # Correct label
    res_correct = index.search(query_text="metformin", n_results=5, label_id="L001")
    assert len(res_correct) > 0

    # Non-existent label
    res_wrong = index.search(query_text="metformin", n_results=5, label_id="L999")
    assert len(res_wrong) == 0


def test_hybrid_fusion_deduplicates_passage_ids():
    dense_mock = [
        {"passage_id": "P1", "text": "Doc 1", "section_type": "s1"},
        {"passage_id": "P2", "text": "Doc 2", "section_type": "s2"},
    ]
    bm25_mock = [
        {"passage_id": "P2", "text": "Doc 2", "section_type": "s2"},
        {"passage_id": "P3", "text": "Doc 3", "section_type": "s3"},
    ]

    fused = reciprocal_rank_fusion(dense_mock, bm25_mock, k=60, top_n=5)
    fused_ids = [p["passage_id"] for p in fused]
    # P2 appears in both, deduplicated
    assert len(fused_ids) == len(set(fused_ids)) == 3
    assert set(fused_ids) == {"P1", "P2", "P3"}


def test_rrf_ranking_is_deterministic():
    dense_mock = [
        {"passage_id": "P1", "text": "Doc 1"},
        {"passage_id": "P2", "text": "Doc 2"},
    ]
    bm25_mock = [
        {"passage_id": "P2", "text": "Doc 2"},
        {"passage_id": "P3", "text": "Doc 3"},
    ]

    fused1 = reciprocal_rank_fusion(dense_mock, bm25_mock, k=60, top_n=5)
    fused2 = reciprocal_rank_fusion(dense_mock, bm25_mock, k=60, top_n=5)

    assert [p["passage_id"] for p in fused1] == [p["passage_id"] for p in fused2]
    assert [p["rrf_score"] for p in fused1] == [p["rrf_score"] for p in fused2]

    # P2 ranked 2 in dense (1/(60+2)) and 1 in bm25 (1/(60+1)), sum is highest
    assert fused1[0]["passage_id"] == "P2"


def test_hybrid_retrieval_returns_metadata_indicating_source():
    dense_mock = [
        {"passage_id": "P1", "text": "Doc 1"},
        {"passage_id": "P2", "text": "Doc 2"},
    ]
    bm25_mock = [
        {"passage_id": "P2", "text": "Doc 2"},
        {"passage_id": "P3", "text": "Doc 3"},
    ]

    fused = reciprocal_rank_fusion(dense_mock, bm25_mock, k=60, top_n=5)
    source_map = {p["passage_id"]: p["retrieval_source"] for p in fused}

    assert source_map["P1"] == "dense"
    assert source_map["P2"] == "both"
    assert source_map["P3"] == "bm25"


def test_hybrid_retrieval_does_not_return_passages_from_another_label(passages):
    # Verify searching with label_id L001 returns only L001
    results = retrieve_passages_hybrid(
        query_text="contraindications and warnings",
        label_id="L001",
        dense_top_k=5,
        bm25_top_k=5,
        final_top_k=5,
    )
    assert len(results) > 0
    for r in results:
        assert r["label_id"] == "L001"
