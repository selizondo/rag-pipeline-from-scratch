"""
ingest.py — Load markdown files → chunk → embed → store in Chroma.

Usage:
    python ingest.py --corpus /path/to/study_notes/aiml --chunk-size 512 --overlap 64
"""

import argparse
import os
import re

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "study_notes"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_markdown_files(corpus_dir: str) -> list[dict]:
    """Read all .md files from corpus_dir, return list of {source, text}."""
    docs = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(corpus_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append({"source": fname, "text": text})
    print(f"Loaded {len(docs)} files from {corpus_dir}")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    Word-based splitting preserves sentence boundaries better than character-based.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest(corpus_dir: str, chunk_size: int, overlap: int, db_path: str) -> int:
    docs = load_markdown_files(corpus_dir)

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=db_path)

    # Drop and recreate collection for clean re-ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{doc['source']}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks))]

        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"\nIngested {total_chunks} chunks from {len(docs)} files into {db_path}")
    return total_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to markdown files")
    parser.add_argument("--chunk-size", type=int, default=512, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap words between chunks")
    parser.add_argument("--db-path", default="./chroma_db", help="Chroma persistence path")
    args = parser.parse_args()

    ingest(args.corpus, args.chunk_size, args.overlap, args.db_path)
