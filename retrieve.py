"""
retrieve.py — Query embedding → Chroma similarity search → optional reranking.

Architecture overview:
    This is the hot-path stage that runs at query time. Every question goes through
    here. Three retrieval modes:

    1. Dense-only (rerank=False):
       Query → bi-encoder embedding → Chroma cosine search → top-K chunks

    2. Dense + rerank (rerank=True):
       Query → bi-encoder → Chroma search (top_k * 3 candidates) →
       cross-encoder reranking → top-K chunks

    3. BM25 fallback (automatic):
       If the top dense score is below BM25_FALLBACK_THRESHOLD, the retriever
       automatically switches to BM25 keyword search and records a fallback event.
       WHY: dense retrieval underperforms on exact-keyword queries (e.g., looking
       for a specific acronym or API name). BM25 handles these better and is a
       fast, in-memory alternative.

    WHY two passes instead of cross-encoder over the full corpus:
        A cross-encoder scores (query, doc) pairs jointly — very accurate but
        O(corpus_size) inference per query. For a 10K-chunk corpus on CPU, that's
        10K forward passes per query (~seconds). Two-pass cuts this to ~15 passes
        (top_k * 3) with almost no quality loss, because the bi-encoder ANN search
        reliably surfaces relevant candidates in the top-3K.

Version-scoped filtering:
    Every Chroma query filters WHERE corpus_version = config.corpus.version before
    ANN search. This scopes retrieval to the current corpus version so bumping the
    version and re-ingesting doesn't mix old and new chunks in results. Callers can
    roll back by temporarily lowering CORPUS_VERSION without full re-ingest.

Startup validation:
    validate_collection() checks that the Chroma collection exists and is non-empty.
    Call it at application startup (e.g., in FastAPI lifespan) so the app warns
    early rather than failing silently on first query.

Usage:
    python retrieve.py --query "What is attention?" --top-k 5 --rerank
"""

import argparse
import logging
import time

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

# Import shared constants — these must match ingest.py exactly.
# Changing EMBED_MODEL without re-ingesting will silently return wrong results.
from config import (
    BM25_FALLBACK_THRESHOLD,
    COLLECTION_NAME,
    CORPUS_VERSION,
    DEFAULT_DB_PATH,
    DEFAULT_TOP_K,
    EMBED_MODEL,
    RERANK_MODEL,
    RETRIEVAL_LATENCY_SLA_MS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Module-level singletons — loaded once at import time, reused for every request.
# WHY: SentenceTransformer and CrossEncoder each take 2–5s to load from disk.
# Loading inside retrieve() adds that overhead to every API call. These are
# thread-safe for inference (no mutable state after __init__).
_embed_model: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def validate_collection(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Check that the Chroma collection exists and contains documents.

    Call this at startup (FastAPI lifespan, CLI entry point) so the app warns
    early rather than returning confusing empty results on the first query.

    Returns True if the collection is healthy, False otherwise. Non-fatal —
    the app continues but logs a warning so operators know to run ingest.py.
    """
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count == 0:
            logger.warning(
                "Chroma collection '%s' exists but is empty — run ingest.py first",
                COLLECTION_NAME,
            )
            return False
        logger.info("Chroma collection '%s' healthy (%d chunks)", COLLECTION_NAME, count)
        return True
    except Exception as e:
        logger.warning(
            "Chroma collection '%s' not found in '%s' — run ingest.py first. (%s)",
            COLLECTION_NAME, db_path, e,
        )
        return False


def _build_bm25_index(db_path: str = DEFAULT_DB_PATH) -> tuple[list[dict], object]:
    """
    Build an in-memory BM25 index from all chunks in the Chroma collection.

    WHY load from Chroma instead of a separate BM25 store:
        We already have all chunks in Chroma. Re-using them avoids maintaining a
        second index. For corpora larger than ~100K chunks, a dedicated Pyserini
        index would be more efficient (BM25 at 100K chunks saturates RAM here).

    Returns (chunks, bm25_index) where chunks[i] corresponds to bm25_index row i.
    """
    from rank_bm25 import BM25Okapi

    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' not found — run ingest.py first. ({e})"
        ) from e

    # Fetch all documents from the current corpus version for BM25 indexing.
    # WHY version-scoped: avoids indexing stale chunks from previous corpus runs.
    result = collection.get(
        where={"corpus_version": CORPUS_VERSION},
        include=["documents", "metadatas"],
    )
    docs = result["documents"]
    metas = result["metadatas"]

    if not docs:
        return [], None

    # Tokenize on whitespace — sufficient for keyword matching.
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    chunks = [{"text": doc, "source": meta["source"]} for doc, meta in zip(docs, metas)]
    return chunks, bm25


def retrieve_bm25(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Keyword-based BM25 retrieval as a fallback when dense scores are too low.

    Scores are normalized to [0, 1] by dividing by the max score in the result
    set so callers can compare them with dense cosine similarity scores.
    """
    chunks, bm25 = _build_bm25_index(db_path)
    if not chunks or bm25 is None:
        return []

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its BM25 score and sort descending.
    ranked = sorted(
        zip(chunks, scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    max_score = ranked[0][1] if ranked else 1.0
    return [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "score": float(score / max_score) if max_score > 0 else 0.0,
            "retrieval_method": "bm25",
        }
        for chunk, score in ranked
    ]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    db_path: str = DEFAULT_DB_PATH,
    rerank: bool = False,
    strategy: str = "auto",
) -> tuple[list[dict], dict]:
    """
    Retrieve the most relevant text chunks for a query.

    Uses a sentence-transformer to embed the query, then searches the Chroma
    vector store for nearest neighbors filtered to the current corpus_version.
    If rerank=True, a cross-encoder rescores the candidate pool.
    If the top dense score < BM25_FALLBACK_THRESHOLD, falls back to BM25.

    Returns (chunks, retrieval_info) where retrieval_info contains:
        strategy: "dense" | "dense_rerank" | "bm25_fallback"
        top_score: float
        embedding_model: str
        reranked: bool
        retrieval_latency_ms: int
        fallback_events: list[str]

    strategy controls retrieval behavior:
        "auto" — dense with automatic BM25 fallback when top_score < BM25_FALLBACK_THRESHOLD
        "bm25" — force BM25 keyword retrieval regardless of dense score
        "dense" — force dense-only; no BM25 fallback even if score is low

    Returns ([], retrieval_info) if the collection is empty.
    Raises RuntimeError if the collection doesn't exist (run ingest.py first).
    """
    t0 = time.time()
    fallback_events: list[str] = []

    # Reuse module-level singleton — loading SentenceTransformer takes 2-5s per call.
    embed_model = _get_embed_model()
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        # Collection doesn't exist — corpus hasn't been ingested yet.
        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' not found in '{db_path}'. "
            f"Run ingest.py first to build the index. (Original error: {e})"
        ) from e

    # Embed the query using the same model used to embed the corpus at ingest time.
    query_embedding = embed_model.encode(query).tolist()

    # WHY top_k * 3 candidates for reranking:
    #   The bi-encoder ANN search is fast but imprecise — it can miss relevant
    #   chunks that aren't closest in embedding space. Fetching 3x candidates
    #   gives the cross-encoder a larger pool to rerank, recovering chunks that
    #   the bi-encoder ranked 6th–15th. The multiplier 3 is empirical; see
    #   chunk_experiment.py for sensitivity analysis across different values.
    n_candidates = top_k * 3 if rerank else top_k
    count = collection.count()
    if count == 0:
        elapsed_ms = round((time.time() - t0) * 1000)
        return [], {
            "strategy": strategy,
            "top_score": 0.0,
            "embedding_model": EMBED_MODEL,
            "corpus_version": CORPUS_VERSION,
            "reranked": False,
            "retrieval_latency_ms": elapsed_ms,
            "fallback_events": [],
        }
    n_candidates = min(n_candidates, count)

    # WHY WHERE corpus_version filter:
    #   Scopes ANN search to the current corpus version so bumping the version
    #   in config.yaml and re-ingesting doesn't mix old and new chunks in results.
    #   This enables corpus rollback without full re-ingest.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
        where={"corpus_version": CORPUS_VERSION},
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Chroma uses cosine distance [0, 2], where 0 = identical vectors.
        # Convert to similarity [0, 1]: similarity = 1 - distance.
        # WHY 1 - dist instead of raw distance: higher score = more relevant
        # is the intuitive convention for callers (and for sorting descending).
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "score": 1 - dist,
            "retrieval_method": "dense",
        })

    # Determine effective retrieval strategy.
    # "bm25" forces BM25 regardless of dense score.
    # "dense" forces dense-only — no automatic fallback.
    # "auto" falls back to BM25 when top dense score < threshold.
    top_dense_score = chunks[0]["score"] if chunks else 0.0
    effective_strategy: str

    if strategy == "bm25":
        bm25_chunks = retrieve_bm25(query, top_k, db_path)
        if bm25_chunks:
            chunks = bm25_chunks
        effective_strategy = "bm25_fallback"
    elif strategy == "dense":
        effective_strategy = "dense"
    else:  # "auto"
        if top_dense_score < BM25_FALLBACK_THRESHOLD:
            logger.info(
                "Dense score %.3f < threshold %.3f — switching to BM25 fallback",
                top_dense_score, BM25_FALLBACK_THRESHOLD,
            )
            fallback_events.append("bm25_fallback_triggered")
            bm25_chunks = retrieve_bm25(query, top_k, db_path)
            if bm25_chunks:
                chunks = bm25_chunks
            effective_strategy = "bm25_fallback"
        else:
            effective_strategy = "dense"

    if rerank and chunks and effective_strategy != "bm25_fallback":
        # Cross-encoder scores (query, document) pairs jointly.
        # WHY cross-encoder for reranking: it attends to both texts simultaneously,
        # catching relevance signals that bi-encoder embedding similarity misses
        # (e.g., a chunk that uses different vocabulary but answers the question).
        reranker = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in chunks]
        rerank_scores = reranker.predict(pairs)
        for chunk, score in zip(chunks, rerank_scores):
            chunk["rerank_score"] = float(score)

        # Sort by cross-encoder score descending — highest relevance first.
        chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        effective_strategy = "dense_rerank"

    chunks = chunks[:top_k]

    # Latency SLA check.
    elapsed_ms = round((time.time() - t0) * 1000)
    if elapsed_ms > RETRIEVAL_LATENCY_SLA_MS:
        logger.warning(
            "Retrieval latency %dms exceeds SLA %dms — consider HNSW indexing "
            "or a smaller corpus. See docs/tradeoffs.md → Scale Boundaries.",
            elapsed_ms, RETRIEVAL_LATENCY_SLA_MS,
        )
        fallback_events.append(f"retrieval_sla_exceeded_{elapsed_ms}ms")

    top_score = 0.0
    if chunks:
        top_score = chunks[0].get("rerank_score", chunks[0]["score"])

    retrieval_info = {
        "strategy": effective_strategy,
        "top_score": round(top_score, 4),
        "embedding_model": EMBED_MODEL,
        "corpus_version": CORPUS_VERSION,
        "reranked": rerank and effective_strategy == "dense_rerank",
        "retrieval_latency_ms": elapsed_ms,
        "fallback_events": fallback_events,
    }

    return chunks, retrieval_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    chunks, info = retrieve(args.query, args.top_k, args.db_path, args.rerank)
    print(f"\nStrategy: {info['strategy']}  top_score={info['top_score']}  "
          f"latency={info['retrieval_latency_ms']}ms")
    if info["fallback_events"]:
        print(f"Fallback events: {info['fallback_events']}")
    for i, c in enumerate(chunks, 1):
        score = c.get("rerank_score", c["score"])
        print(f"\n[{i}] {c['source']} (score: {score:.4f})")
        print(c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"])
