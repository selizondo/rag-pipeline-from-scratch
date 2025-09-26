"""
pipeline.py — End-to-end RAG: question in, answer out.

Usage:
    # First ingest the corpus (one-time):
    python ingest.py --corpus /path/to/study_notes/aiml

    # Then ask questions:
    python pipeline.py --query "What is the difference between RAG and fine-tuning?"
    python pipeline.py --query "Explain attention mechanisms" --rerank --top-k 8
"""

import argparse
import time

from generate import generate
from ingest import ingest
from retrieve import retrieve


def run(
    query: str,
    top_k: int = 5,
    rerank: bool = False,
    model: str = "llama3.2",
    db_path: str = "./chroma_db",
) -> dict:
    """Run the full RAG pipeline and return result with timing metadata."""
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    t0 = time.time()

    # Retrieve relevant chunks first, optionally applying reranking.
    chunks = retrieve(query, top_k=top_k, db_path=db_path, rerank=rerank)
    t_retrieve = time.time()

    # Generate the answer from the retrieved context.
    answer = generate(query, chunks, model=model)
    t_generate = time.time()

    return {
        "query": query,
        "answer": answer,
        "chunks_used": len(chunks),
        "sources": list({c["source"] for c in chunks}),
        "latency": {
            "retrieve_ms": round((t_retrieve - t0) * 1000),
            "generate_ms": round((t_generate - t_retrieve) * 1000),
            "total_ms": round((t_generate - t0) * 1000),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline — question in, answer out")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--db-path", default="./chroma_db", help="Chroma DB path")
    parser.add_argument(
        "--ingest",
        metavar="CORPUS_DIR",
        help="(Re-)ingest corpus before querying",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    if args.ingest:
        ingest(args.ingest, args.chunk_size, args.overlap, args.db_path)

    result = run(
        query=args.query,
        top_k=args.top_k,
        rerank=args.rerank,
        model=args.model,
        db_path=args.db_path,
    )

    print(f"\nQ: {result['query']}\n")
    print(f"A: {result['answer']}\n")
    print(f"Sources: {', '.join(result['sources'])}")
    print(f"Latency: retrieve={result['latency']['retrieve_ms']}ms  "
          f"generate={result['latency']['generate_ms']}ms  "
          f"total={result['latency']['total_ms']}ms")
