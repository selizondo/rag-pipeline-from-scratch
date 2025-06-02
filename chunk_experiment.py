"""
chunk_experiment.py — Compare retrieval quality across chunk sizes.

Usage:
    python chunk_experiment.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from ingest import ingest
from retrieve import retrieve

CORPUS = "/Users/selizondo/Dropbox/projects/vscode/ml/ai_ml_study_notes/chats"
QUERIES = [
    "What is feature engineering and why does it matter?",
    "What are the most important feature transformation techniques?",
    "How do you handle missing values in feature engineering?",
]
CHUNK_SIZES = [256, 512, 1024]
OVERLAP_RATIO = 0.125  # 12.5% overlap (consistent ratio across sizes)
TOP_K = 3


def run_experiment():
    results = {}

    for size in CHUNK_SIZES:
        overlap = int(size * OVERLAP_RATIO)
        db_path = f"./chroma_db_exp_{size}"
        print(f"\n{'='*60}")
        print(f"Chunk size: {size} words  |  Overlap: {overlap} words")
        print(f"{'='*60}")

        total_chunks = ingest(CORPUS, size, overlap, db_path)
        results[size] = {"total_chunks": total_chunks, "queries": {}}

        for query in QUERIES:
            chunks = retrieve(query, top_k=TOP_K, db_path=db_path, rerank=False)
            avg_score = sum(c["score"] for c in chunks) / len(chunks) if chunks else 0
            top_score = chunks[0]["score"] if chunks else 0

            results[size]["queries"][query] = {
                "avg_score": round(avg_score, 4),
                "top_score": round(top_score, 4),
                "top_source": chunks[0]["source"] if chunks else "n/a",
                "top_snippet": chunks[0]["text"][:120].replace("\n", " ") if chunks else "",
            }

            print(f"\n  Q: {query[:60]}...")
            print(f"     top_score={top_score:.4f}  avg={avg_score:.4f}  source={chunks[0]['source'] if chunks else 'n/a'}")

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY — Average top_score per chunk size")
    print(f"{'='*60}")
    print(f"{'Size':>6}  {'Chunks':>7}  {'Avg top_score':>14}")
    for size in CHUNK_SIZES:
        scores = [v["top_score"] for v in results[size]["queries"].values()]
        avg = sum(scores) / len(scores)
        total = results[size]["total_chunks"]
        print(f"{size:>6}  {total:>7}  {avg:>14.4f}")

    with open("chunk_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to chunk_experiment_results.json")


if __name__ == "__main__":
    run_experiment()
