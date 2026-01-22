"""
ingest.py — Load markdown files → chunk → embed → store in Chroma.

Architecture overview:
    This is the offline/batch stage of the pipeline. You run it once (or when
    the corpus changes), and it builds the vector index that retrieve.py searches
    at query time. Nothing here runs on the hot path.

    Flow: .md files → word-based chunking → sentence-transformer embeddings → Chroma

Version-scoped filtering:
    Every chunk is stored with a `corpus_version` metadata field (from
    config.yaml → corpus.version). Retrieve.py filters `WHERE corpus_version = "v1"`
    before ANN search, so bumping the version and re-ingesting doesn't mix old
    and new chunks — callers can roll back by querying an older version without
    full re-ingest. See docs/tradeoffs.md for details.

Usage:
    python ingest.py --corpus /path/to/study_notes/aiml --chunk-size 512 --overlap 64
"""

import argparse
import logging
import os

import chromadb
from sentence_transformers import SentenceTransformer

# Import shared constants — EMBED_MODEL must match retrieve.py exactly.
# WHY import instead of define locally: if ingest and retrieve use different
# embedding models, vectors are incompatible and retrieval silently returns wrong
# results. Sharing a constant from config.py makes that invariant enforced.
from config import (
    COLLECTION_NAME,
    CORPUS_VERSION,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_OVERLAP,
    EMBED_MODEL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
    logger.info("Loaded %d files from %s", len(docs), corpus_dir)
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a document into overlapping fixed-size chunks by word count.

    WHY word-based splitting instead of character-based:
        Words are natural language units — splitting on words is less likely to
        break at mid-token or mid-number than a character-based split. It also
        produces chunk sizes that are more stable across varied text density.

    WHY not sentence-boundary splitting:
        Sentence tokenization requires NLTK or spaCy and adds a dependency.
        For this research baseline, word-based splitting is sufficient and
        transparent. The known failure mode: if a sentence is longer than
        chunk_size words, it will be split mid-sentence. Overlap reduces but
        does not eliminate this problem.

    Known limitation — tokenizer mismatch:
        This function counts words, but the embedding model (all-MiniLM-L6-v2)
        tokenizes internally using WordPiece. One word ≈ 1.0–1.3 WordPiece tokens
        on average, but compound words and rare terms can be 2–4 tokens. The
        model has a hard 256-token limit — a 512-word chunk can be 400–650
        WordPiece tokens, and the tail is silently truncated. In practice,
        important content tends to be at the start of a chunk, so the impact is
        limited. But this is a known gap. See docs/tradeoffs.md.
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
    """
    Ingest a markdown corpus into a persistent Chroma collection.

    Every chunk is tagged with corpus_version from config.yaml so retrieve.py
    can filter to a specific corpus version without mixing old and new chunks.

    WHY delete-then-recreate instead of upsert:
        On a re-ingest (changed corpus or changed chunk_size), upsert would mix
        old and new chunks in the same collection. Chunks that no longer exist in
        the corpus would persist and pollute retrieval. Delete+recreate gives a
        clean slate every time — the cost is that re-ingest is all-or-nothing, but
        for a corpus this size (~MB range) that's acceptable.
    """
    docs = load_markdown_files(corpus_dir)

    # Load the sentence-transformer model that converts text to dense vectors.
    # This model must match EMBED_MODEL used in retrieve.py.
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=db_path)

    # Drop the collection if it exists, then recreate it clean.
    # The try/except handles the case where no collection exists yet (first run).
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, overlap)

        # Batch-encode all chunks for this document at once — sentence-transformers
        # is significantly faster encoding in batches than one chunk at a time.
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        # Use stable IDs so the same chunk always maps to the same vector row.
        # Format: "{filename}_{chunk_index}" — unique within the collection.
        ids = [f"{doc['source']}_{i}" for i in range(len(chunks))]

        # corpus_version enables version-scoped ANN filtering in retrieve.py.
        # WHY store it per-chunk: Chroma metadata filters are applied before ANN
        # search, so filtering WHERE corpus_version = "v1" scopes the search to
        # the current corpus without re-ingesting. Bump CORPUS_VERSION in
        # config.yaml when the corpus changes to isolate old and new chunks.
        metadatas = [
            {
                "source": doc["source"],
                "chunk_index": i,
                "corpus_version": CORPUS_VERSION,
            }
            for i in range(len(chunks))
        ]

        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
        logger.info("  %s: %d chunks (corpus_version=%s)", doc["source"], len(chunks), CORPUS_VERSION)

    logger.info(
        "Ingested %d chunks from %d files into %s (corpus_version=%s)",
        total_chunks, len(docs), db_path, CORPUS_VERSION,
    )
    return total_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to markdown files")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlap words between chunks")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Chroma persistence path")
    args = parser.parse_args()

    ingest(args.corpus, args.chunk_size, args.overlap, args.db_path)
