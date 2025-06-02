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
    Retrieve top_k relevant chunks for a query.
    Returns list of {text, source, score}.
    """
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(COLLECTION_NAME)

    # Embed query and search
    query_embedding = embed_model.encode(query).tolist()
    # Fetch more candidates when reranking so the reranker has room to reorder
    n_candidates = top_k * 3 if rerank else top_k
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_candidates, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "score": 1 - dist,  # convert cosine distance → similarity
        })

    if rerank and chunks:
        reranker = CrossEncoder(RERANK_MODEL)
        pairs = [(query, c["text"]) for c in chunks]
        rerank_scores = reranker.predict(pairs)
        for chunk, score in zip(chunks, rerank_scores):
            chunk["rerank_score"] = float(score)
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
