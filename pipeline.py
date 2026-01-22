"""
pipeline.py — End-to-end RAG: question in, answer out.

This is the integration layer that wires retrieve.py and generate.py together.
It doesn't contain retrieval or generation logic — it just calls those modules
in order and assembles the combined response.

WHY a separate pipeline module instead of calling retrieve+generate inline:
    Separation of concerns: each module can be tested and benchmarked in isolation
    (retrieve.py for retrieval quality, generate.py for generation quality).
    pipeline.py only handles timing, response assembly, and the ingest flag.

Usage:
    # First ingest the corpus (one-time):
    python ingest.py --corpus /path/to/study_notes/aiml

    # Then ask questions:
    python pipeline.py --query "What is the difference between RAG and fine-tuning?"
    python pipeline.py --query "Explain attention mechanisms" --rerank --top-k 8
"""

import argparse
import time

from config import (
    CORPUS_VERSION,
    DEFAULT_DB_PATH,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K,
    EMBED_MODEL,
)
from generate import generate
from ingest import ingest
from retrieve import retrieve


def run(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    rerank: bool = False,
    model: str = DEFAULT_OLLAMA_MODEL,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """
    Run the full RAG pipeline and return a result dict with answer + metadata.

    The returned dict includes:
    - retrieval_metadata: strategy, top_score, embedding_model, corpus_version,
      reranked, retrieval_latency_ms
    - fallback_events: list of events (ollama_timeout, bm25_fallback, etc.)
    - embedding_model, corpus_version: top-level for artifact traceability

    WHY include these in the response instead of logs:
        Logs require a monitoring agent to parse. Response fields let the calling
        code (eval harness, integration tests, api.py) observe pipeline state
        directly without coupling to log format.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    t0 = time.time()

    # Retrieve relevant chunks. retrieve() now returns (chunks, retrieval_info)
    # with latency, strategy, fallback_events, and corpus_version already computed.
    chunks, retrieval_info = retrieve(query, top_k=top_k, db_path=db_path, rerank=rerank)
    t_retrieve = time.time()

    # Generate the answer using the retrieved context.
    # generate() returns (answer, fallback_events) — either the answer or a
    # best-effort fallback string. Generation fallback_events are merged with
    # retrieval fallback_events into a single list on the response.
    answer, gen_fallback_events = generate(query, chunks, model=model)
    t_generate = time.time()

    # Merge fallback events from both retrieval and generation stages.
    all_fallback_events = retrieval_info.get("fallback_events", []) + gen_fallback_events

    return {
        "query": query,
        "answer": answer,
        "chunks_used": len(chunks),
        "sources": list({c["source"] for c in chunks}),
        # Artifact traceability — lets eval harness and monitoring know exactly
        # which model and corpus version produced this answer.
        "embedding_model": EMBED_MODEL,
        "corpus_version": CORPUS_VERSION,
        # retrieval_metadata lets callers audit how the answer was produced
        # without re-running the pipeline or parsing logs.
        "retrieval_metadata": {
            "strategy": retrieval_info["strategy"],
            "top_score": retrieval_info["top_score"],
            "embedding_model": retrieval_info["embedding_model"],
            "corpus_version": retrieval_info["corpus_version"],
            "reranked": retrieval_info["reranked"],
            "retrieval_latency_ms": retrieval_info["retrieval_latency_ms"],
        },
        # fallback_events captures all degradation events across retrieval and
        # generation stages. Empty list means happy path. Populated list means
        # something fell back — check the strings for details.
        "fallback_events": all_fallback_events,
        "latency": {
            "retrieve_ms": round((t_retrieve - t0) * 1000),
            "generate_ms": round((t_generate - t_retrieve) * 1000),
            "total_ms": round((t_generate - t0) * 1000),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline — question in, answer out")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks to retrieve")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model to use")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Chroma DB path")
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
    print(f"Retrieval: strategy={result['retrieval_metadata']['strategy']}  "
          f"top_score={result['retrieval_metadata']['top_score']}  "
          f"latency={result['retrieval_metadata']['retrieval_latency_ms']}ms")
    print(f"Artifact: embedding_model={result['embedding_model']}  "
          f"corpus_version={result['corpus_version']}")
    if result["fallback_events"]:
        print(f"Fallback events: {result['fallback_events']}")
    print(f"Latency: retrieve={result['latency']['retrieve_ms']}ms  "
          f"generate={result['latency']['generate_ms']}ms  "
          f"total={result['latency']['total_ms']}ms")
