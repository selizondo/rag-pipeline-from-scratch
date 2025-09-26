"""
ingest.py — Load markdown files → chunk → embed → store in Chroma.

Usage:
    python ingest.py --corpus /path/to/study_notes/aiml --chunk-size 512 --overlap 64
"""

import argparse
import os

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
    Split a document into overlapping chunks by word count.

    This is a simple way to create context windows for retrieval.
    We use words rather than characters so the chunks are less likely
    to break in the middle of a sentence, and we add overlap so that
    important context is preserved between adjacent chunks.
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
    """Ingest a markdown corpus into a persistent Chroma collection."""
    docs = load_markdown_files(corpus_dir)

    # Load the sentence-transformers model that converts text to vectors.
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=db_path)

    # Delete and recreate the collection so the ingestion is clean every time.
    # This avoids duplicated chunks when rerunning on the same corpus.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        # Use stable IDs and metadata so we can track which source file and
        # chunk index each vector belongs to after retrieval.
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
