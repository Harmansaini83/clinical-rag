"""Dense and BM25 hybrid retrieval service using persistent ChromaDB and ONNX embeddings."""
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from app.config import settings


_EMBEDDING_FUNCTION: Optional[embedding_functions.EmbeddingFunction] = None


def get_embedding_function() -> embedding_functions.EmbeddingFunction:
    """Return the cached embedding function (ONNX all-MiniLM-L6-v2)."""
    global _EMBEDDING_FUNCTION
    if _EMBEDDING_FUNCTION is None:
        _EMBEDDING_FUNCTION = embedding_functions.DefaultEmbeddingFunction()
    return _EMBEDDING_FUNCTION


def get_chroma_client(persist_dir: Optional[Path] = None) -> ClientAPI:
    """Return a persistent ChromaDB client pointing to the configured directory."""
    path = str(persist_dir or settings.chroma_dir)
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(
    client: Optional[ClientAPI] = None,
    collection_name: Optional[str] = None,
) -> Collection:
    """Get or create the persistent passages collection with cosine distance."""
    if client is None:
        client = get_chroma_client()
    name = collection_name or settings.chroma_collection_name
    ef = get_embedding_function()
    return client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": settings.chroma_distance_fn},
    )


def load_passages_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load passage records from an authoritative JSONL file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Passage file not found: {file_path}")
    passages: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    return passages


def index_passages(
    passages: List[Dict[str, Any]],
    collection: Optional[Collection] = None,
) -> int:
    """Index passages into ChromaDB with explicit embeddings and required metadata.

    Required metadata stored per passage:
      - document_id
      - label_id
      - section_type
      - section_title
      - section_path
      - word_count
    """
    if not passages:
        return 0

    if collection is None:
        collection = get_or_create_collection()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for p in passages:
        ids.append(p["passage_id"])
        documents.append(p["text"])
        metadatas.append(
            {
                "document_id": str(p["document_id"]),
                "label_id": str(p["label_id"]),
                "section_type": str(p["section_type"]),
                "section_title": str(p["section_title"]),
                "section_path": str(p["section_path"]),
                "word_count": int(p["word_count"]),
            }
        )

    # Compute embeddings explicitly using the verified embedding function
    ef = get_embedding_function()
    embeddings = ef(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def retrieve_passages(
    query_text: str,
    n_results: int = 5,
    label_id: Optional[str] = None,
    collection: Optional[Collection] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-N passages using dense ChromaDB search with optional label filtering."""
    if collection is None:
        collection = get_or_create_collection()

    where_filter = {"label_id": label_id} if label_id else None

    # Compute query embedding explicitly for full deterministic traceability
    ef = get_embedding_function()
    query_embeddings = ef([query_text])

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where=where_filter,
    )

    retrieved: List[Dict[str, Any]] = []
    if not results or not results["ids"] or not results["ids"][0]:
        return retrieved

    ids = results["ids"][0]
    docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
    dists = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

    for pid, doc, meta, dist in zip(ids, docs, metas, dists):
        retrieved.append(
            {
                "passage_id": pid,
                "text": doc,
                "distance": dist,
                "document_id": meta.get("document_id", ""),
                "label_id": meta.get("label_id", ""),
                "section_type": meta.get("section_type", ""),
                "section_title": meta.get("section_title", ""),
                "section_path": meta.get("section_path", ""),
                "word_count": meta.get("word_count", 0),
                "retrieval_source": "dense",
            }
        )
    return retrieved


# ==============================================================================
# BM25 & HYBRID RETRIEVAL (RECIPROCAL RANK FUSION)
# ==============================================================================

def bm25_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25 matching."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25Okapi index built from authoritative passage documents."""

    def __init__(
        self,
        passages: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(passages)
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_term_freqs: List[Counter] = []
        self.df: Dict[str, int] = Counter()
        self.idf: Dict[str, float] = {}

        self._build_index()

    def _build_index(self) -> None:
        if not self.passages:
            return

        total_len = 0
        for p in self.passages:
            tokens = bm25_tokenize(p.get("text", ""))
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for term in tf.keys():
                self.df[term] += 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate Okapi IDF with floor threshold for negative IDFs
        negative_idfs = []
        for term, freq in self.df.items():
            # Standard Lucene/BM25Okapi IDF: ln(1 + (N - n + 0.5) / (n + 0.5))
            val = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[term] = val
            if val < 0:
                negative_idfs.append(term)

        # Handle negative IDFs if any
        if negative_idfs:
            average_idf = sum(v for v in self.idf.values() if v > 0) / max(1, len(self.idf) - len(negative_idfs))
            eps_idf = self.epsilon * average_idf
            for term in negative_idfs:
                self.idf[term] = eps_idf

    def search(
        self,
        query_text: str,
        n_results: int = 5,
        label_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-N scored passages for query_text with optional label_id filter."""
        query_tokens = bm25_tokenize(query_text)
        if not query_tokens or not self.passages:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, (p, tf, doc_len) in enumerate(zip(self.passages, self.doc_term_freqs, self.doc_lens)):
            if label_id and p.get("label_id") != label_id:
                continue

            score = 0.0
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0)
            for token in query_tokens:
                if token not in tf:
                    continue
                term_idf = self.idf.get(token, 0.0)
                freq = tf[token]
                term_score = term_idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm)
                score += term_score

            if score > 0.0:
                scores.append((idx, score))

        # Sort by score descending; break ties by passage_id
        scores.sort(key=lambda x: (x[1], -self.passages[x[0]].get("word_count", 0)), reverse=True)

        results: List[Dict[str, Any]] = []
        for doc_idx, score in scores[:n_results]:
            p = self.passages[doc_idx]
            results.append(
                {
                    "passage_id": p["passage_id"],
                    "text": p["text"],
                    "bm25_score": score,
                    "document_id": p.get("document_id", ""),
                    "label_id": p.get("label_id", ""),
                    "section_type": p.get("section_type", ""),
                    "section_title": p.get("section_title", ""),
                    "section_path": p.get("section_path", ""),
                    "word_count": p.get("word_count", 0),
                    "retrieval_source": "bm25",
                }
            )
        return results


# Cached singleton BM25 index per label
_BM25_INDICES: Dict[str, BM25Index] = {}


def get_bm25_index(label_id: str = "L001", passage_file: Optional[Path] = None) -> BM25Index:
    """Return or initialize cached BM25Index for a specific drug label."""
    global _BM25_INDICES
    if label_id not in _BM25_INDICES:
        path = passage_file or (settings.passages_dir / f"{label_id}.jsonl")
        passages = load_passages_from_jsonl(path)
        _BM25_INDICES[label_id] = BM25Index(passages)
    return _BM25_INDICES[label_id]


def retrieve_passages_bm25(
    query_text: str,
    n_results: int = 5,
    label_id: str = "L001",
    bm25_index: Optional[BM25Index] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-N passages using local BM25 ranking."""
    idx = bm25_index or get_bm25_index(label_id=label_id)
    return idx.search(query_text=query_text, n_results=n_results, label_id=label_id)


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Combine dense and BM25 candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum_{method} (1.0 / (k + rank))
        where rank is 1-indexed (1, 2, ...).

    Source tracking:
        Marks each passage's retrieval_source as 'dense', 'bm25', or 'both'.
    """
    rrf_scores: Dict[str, float] = {}
    passage_objects: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, Set[str]] = {}

    # 1. Process Dense candidates
    for rank, p in enumerate(dense_results, start=1):
        pid = p["passage_id"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + rank))
        if pid not in passage_objects:
            passage_objects[pid] = dict(p)
        sources.setdefault(pid, set()).add("dense")

    # 2. Process BM25 candidates
    for rank, p in enumerate(bm25_results, start=1):
        pid = p["passage_id"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + rank))
        if pid not in passage_objects:
            passage_objects[pid] = dict(p)
        sources.setdefault(pid, set()).add("bm25")

    # 3. Sort by RRF score descending
    sorted_pids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)

    fused_results: List[Dict[str, Any]] = []
    for pid in sorted_pids[:top_n]:
        item = passage_objects[pid]
        src_set = sources[pid]
        if "dense" in src_set and "bm25" in src_set:
            source_tag = "both"
        elif "dense" in src_set:
            source_tag = "dense"
        else:
            source_tag = "bm25"

        item["rrf_score"] = rrf_scores[pid]
        item["retrieval_source"] = source_tag
        fused_results.append(item)

    return fused_results


def retrieve_passages_hybrid(
    query_text: str,
    label_id: str = "L001",
    dense_top_k: int = 5,
    bm25_top_k: int = 5,
    final_top_k: int = 5,
    rrf_k: int = 60,
    collection: Optional[Collection] = None,
    bm25_index: Optional[BM25Index] = None,
) -> List[Dict[str, Any]]:
    """Execute Hybrid Retrieval combining Dense ChromaDB and Sparse BM25 via RRF.

    Parameters:
      - dense_top_k: 5
      - bm25_top_k: 5
      - final_top_k: 5
      - rrf_k: 60
    """
    dense_candidates = retrieve_passages(
        query_text=query_text,
        n_results=dense_top_k,
        label_id=label_id,
        collection=collection,
    )
    bm25_candidates = retrieve_passages_bm25(
        query_text=query_text,
        n_results=bm25_top_k,
        label_id=label_id,
        bm25_index=bm25_index,
    )
    return reciprocal_rank_fusion(
        dense_results=dense_candidates,
        bm25_results=bm25_candidates,
        k=rrf_k,
        top_n=final_top_k,
    )
