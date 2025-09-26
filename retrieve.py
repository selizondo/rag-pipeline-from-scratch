"""
retrieve.py — Query embedding → Chroma similarity search → optional reranking.

Usage:
    python retrieve.py --query "What is attention?" --top-k 5 --rerank
"""

import argparse

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

COLLECTION_NAME = "study_notes"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def retrieve(
    query: str,
    top_k: int = 5,
    db_path: str = "./chroma_db",
    rerank: bool = False,
) -> list[dict]:
    """
    Retrieve the most relevant text chunks for a query.

    Uses a sentence-transformer to embed the query, then searches the Chroma
    vector store for nearest neighbors. If rerank=True, a cross-encoder is used
    to rescore the candidate chunks with a more expensive but more accurate
    relevance model.

    Returns a list of dicts with at least text, source, and score fields.
    """
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(COLLECTION_NAME)

    # Embed the query using the same model family used for the corpus.
    query_embedding = embed_model.encode(query).tolist()

    # When reranking, fetch more candidates than top_k so the cross-encoder can
    # choose the best ones from a larger pool. If rerank is disabled, only load
    # the exact number requested.
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
        # Chroma returns cosine distances; higher similarity means lower distance.
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "score": 1 - dist,
        })

    if rerank and chunks:
        # Cross-encoder models score query+document pairs directly, which can
        # improve relevance ordering compared to raw vector similarity.
        reranker = CrossEncoder(RERANK_MODEL)
        pairs = [(query, c["text"]) for c in chunks]
        rerank_scores = reranker.predict(pairs)
        for chunk, score in zip(chunks, rerank_scores):
            chunk["rerank_score"] = float(score)

        # Sort by the cross-encoder score descending, keeping highest relevance first.
        chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return chunks[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-path", default="./chroma_db")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    chunks = retrieve(args.query, args.top_k, args.db_path, args.rerank)
    for i, c in enumerate(chunks, 1):
        score = c.get("rerank_score", c["score"])
        print(f"\n[{i}] {c['source']} (score: {score:.4f})")
        print(c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"])
