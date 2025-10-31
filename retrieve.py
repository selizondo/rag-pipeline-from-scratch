"""
retrieve.py — Query embedding → Chroma similarity search → optional reranking.

Architecture overview:
    This is the hot-path stage that runs at query time. Every question goes through
    here. Two retrieval modes:

    1. Dense-only (rerank=False):
       Query → bi-encoder embedding → Chroma cosine search → top-K chunks

    2. Dense + rerank (rerank=True):
       Query → bi-encoder → Chroma search (top_k * 3 candidates) →
       cross-encoder reranking → top-K chunks

    WHY two passes instead of cross-encoder over the full corpus:
        A cross-encoder scores (query, doc) pairs jointly — very accurate but
        O(corpus_size) inference per query. For a 10K-chunk corpus on CPU, that's
        10K forward passes per query (~seconds). Two-pass cuts this to ~15 passes
        (top_k * 3) with almost no quality loss, because the bi-encoder ANN search
        reliably surfaces relevant candidates in the top-3K.

Usage:
    python retrieve.py --query "What is attention?" --top-k 5 --rerank
"""

import argparse

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

# Import shared constants — these must match ingest.py exactly.
# Changing EMBED_MODEL without re-ingesting will silently return wrong results.
from config import COLLECTION_NAME, DEFAULT_DB_PATH, EMBED_MODEL, RERANK_MODEL


def retrieve(
    query: str,
    top_k: int = 5,
    db_path: str = DEFAULT_DB_PATH,
    rerank: bool = False,
) -> list[dict]:
    """
    Retrieve the most relevant text chunks for a query.

    Uses a sentence-transformer to embed the query, then searches the Chroma
    vector store for nearest neighbors. If rerank=True, a cross-encoder rescores
    the candidate pool for more accurate relevance ordering.

    Returns a list of dicts with keys: text, source, score (and rerank_score if
    rerank=True). score is always the bi-encoder cosine similarity [0, 1].

    Returns [] if the collection is empty (corpus not yet ingested).
    Raises chromadb.errors.InvalidCollectionException if the collection doesn't
    exist at all — call ingest.py first.
    """
    # Load the same embedding model used during ingest — vector spaces must match.
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        # Collection doesn't exist — corpus hasn't been ingested yet.
        # Return empty with a clear message rather than crashing with a cryptic error.
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
        return []
    n_candidates = min(n_candidates, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
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
        })

    if rerank and chunks:
        # Cross-encoder scores (query, document) pairs jointly.
        # WHY cross-encoder for reranking: it attends to both texts simultaneously,
        # catching relevance signals that bi-encoder embedding similarity misses
        # (e.g., a chunk that uses different vocabulary but answers the question).
        reranker = CrossEncoder(RERANK_MODEL)
        pairs = [(query, c["text"]) for c in chunks]
        rerank_scores = reranker.predict(pairs)
        for chunk, score in zip(chunks, rerank_scores):
            chunk["rerank_score"] = float(score)

        # Sort by cross-encoder score descending — highest relevance first.
        chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return chunks[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    chunks = retrieve(args.query, args.top_k, args.db_path, args.rerank)
    for i, c in enumerate(chunks, 1):
        score = c.get("rerank_score", c["score"])
        print(f"\n[{i}] {c['source']} (score: {score:.4f})")
        print(c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"])
