"""
evaluation/eval.py — Retrieval quality evaluation against a labeled test set.

Metrics computed:
    Precision@K: fraction of retrieved chunks from relevant sources
    Recall@K: fraction of relevant sources that appear in top-K results

Test set: data/test_set.json — 10 Q&A pairs with manually labeled relevant_sources.
Results: artifacts/eval/latest_run.json — overwritten each run.

Retrieval strategies compared:
    1. Dense: all-MiniLM-L6-v2 bi-encoder cosine search
    2. Dense + rerank: dense candidates → cross-encoder reranking
    3. BM25: rank_bm25 keyword search (baseline)

WHY evaluate retrieval separately from generation:
    Generation quality depends on retrieval quality. If the wrong chunks are
    retrieved, no generation model can produce a correct answer. Measuring
    retrieval precision/recall in isolation pinpoints whether the problem is
    in retrieval (wrong chunks) or generation (chunks retrieved but answer wrong).

Usage:
    python evaluation/eval.py
    python evaluation/eval.py --top-k 4 --strategy dense
    python evaluation/eval.py --all-strategies  # runs all three and compares

Output:
    artifacts/eval/latest_run.json — per-question results + aggregate metrics
    Prints summary table to stdout.

NOTE: This script does NOT call Ollama — it only evaluates retrieval quality.
Run it after any corpus update to detect retrieval regressions.
    make eval
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root or from evaluation/ subdirectory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CORPUS_VERSION, DEFAULT_DB_PATH, DEFAULT_TOP_K, EMBED_MODEL
from retrieve import retrieve, retrieve_bm25

TEST_SET_PATH = Path(__file__).parent.parent / "data" / "test_set.json"
RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "eval"


def precision_at_k(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """
    Precision@K: fraction of retrieved chunks whose source is in relevant_sources.

    WHY source-level instead of chunk-level precision:
        The test set labels relevant_sources (file names), not individual chunk IDs.
        Source-level precision is coarser but avoids the labeling cost of marking
        individual chunks. For a higher-fidelity eval, label chunk IDs directly.
    """
    if not retrieved_sources:
        return 0.0
    hits = sum(1 for s in retrieved_sources if s in relevant_sources)
    return hits / len(retrieved_sources)


def recall_at_k(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """
    Recall@K: fraction of relevant sources that appear in the retrieved set.

    Capped at 1.0 — if you retrieve the same relevant source multiple times,
    it counts once. Each relevant source is binary: found or not found.
    """
    if not relevant_sources:
        return 0.0
    retrieved_set = set(retrieved_sources)
    hits = sum(1 for s in relevant_sources if s in retrieved_set)
    return hits / len(relevant_sources)


def evaluate_strategy(
    test_cases: list[dict],
    strategy: str,
    top_k: int,
    db_path: str,
) -> dict:
    """
    Run retrieval for all test cases under a given strategy and compute metrics.

    strategy: "dense" | "dense_rerank" | "bm25"
    Returns a dict with per-question results and aggregate precision/recall.
    """
    per_question = []
    total_precision = 0.0
    total_recall = 0.0

    for case in test_cases:
        q = case["question"]
        relevant = case["relevant_sources"]
        t0 = time.time()

        if strategy == "bm25":
            chunks = retrieve_bm25(q, top_k, db_path)
            retrieval_info = {
                "strategy": "bm25",
                "top_score": chunks[0]["score"] if chunks else 0.0,
                "retrieval_latency_ms": round((time.time() - t0) * 1000),
                "fallback_events": [],
            }
        elif strategy == "dense_rerank":
            chunks, retrieval_info = retrieve(q, top_k, db_path, rerank=True)
        else:  # dense
            chunks, retrieval_info = retrieve(q, top_k, db_path, rerank=False)

        retrieved_sources = [c["source"] for c in chunks]
        prec = precision_at_k(retrieved_sources, relevant)
        rec = recall_at_k(retrieved_sources, relevant)
        total_precision += prec
        total_recall += rec

        per_question.append({
            "id": case["id"],
            "question": q,
            "relevant_sources": relevant,
            "retrieved_sources": retrieved_sources,
            "top_score": retrieval_info.get("top_score", 0.0),
            "latency_ms": retrieval_info.get("retrieval_latency_ms", 0),
            "precision_at_k": round(prec, 4),
            "recall_at_k": round(rec, 4),
        })

    n = len(test_cases)
    return {
        "strategy": strategy,
        "top_k": top_k,
        "avg_precision_at_k": round(total_precision / n, 4) if n else 0.0,
        "avg_recall_at_k": round(total_recall / n, 4) if n else 0.0,
        "per_question": per_question,
    }


def run_eval(
    top_k: int = DEFAULT_TOP_K,
    strategy: str = "dense",
    db_path: str = DEFAULT_DB_PATH,
    all_strategies: bool = False,
) -> dict:
    """
    Load the test set and run evaluation for one or all retrieval strategies.

    Saves results to artifacts/eval/latest_run.json.
    Returns the full results dict.
    """
    with open(TEST_SET_PATH) as f:
        test_cases = json.load(f)

    strategies_to_run = ["dense", "dense_rerank", "bm25"] if all_strategies else [strategy]

    results = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "corpus_version": CORPUS_VERSION,
        "embedding_model": EMBED_MODEL,
        "top_k": top_k,
        "db_path": db_path,
        "test_set": str(TEST_SET_PATH),
        "n_questions": len(test_cases),
        "strategies": {},
    }

    for strat in strategies_to_run:
        print(f"\nRunning strategy: {strat} (top_k={top_k})...")
        strat_results = evaluate_strategy(test_cases, strat, top_k, db_path)
        results["strategies"][strat] = strat_results
        print(f"  Precision@{top_k}: {strat_results['avg_precision_at_k']:.4f}  "
              f"Recall@{top_k}: {strat_results['avg_recall_at_k']:.4f}")

    # Save results to artifacts/eval/latest_run.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "latest_run.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'Strategy':<16} {'Precision@K':>12} {'Recall@K':>10}")
    print("-" * 42)
    for strat, res in results["strategies"].items():
        print(f"{strat:<16} {res['avg_precision_at_k']:>12.4f} {res['avg_recall_at_k']:>10.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks to retrieve per query")
    parser.add_argument("--strategy", choices=["dense", "dense_rerank", "bm25"], default="dense")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run dense, dense_rerank, and bm25 and compare",
    )
    args = parser.parse_args()

    run_eval(
        top_k=args.top_k,
        strategy=args.strategy,
        db_path=args.db_path,
        all_strategies=args.all_strategies,
    )
