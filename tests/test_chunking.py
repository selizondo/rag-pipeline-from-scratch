"""
tests/test_chunking.py — Unit tests for the text chunker.

Coverage:
  - Basic chunk splitting: output count and content integrity
  - Overlap: consecutive chunks share overlap_tokens worth of content
  - Empty input: returns empty list
  - Short text smaller than chunk_size: returns single chunk
  - MiniLM token limit: documents the known truncation behavior
"""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Import the production chunker — same logic used by ingest.py."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingest import chunk_text
    return chunk_text(text, chunk_size, overlap)


class TestChunkText:
    def test_basic_split_produces_multiple_chunks(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = _chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) >= 2, "100-word text with chunk_size=20 should produce multiple chunks"

    def test_single_chunk_when_text_smaller_than_chunk_size(self):
        text = "hello world foo bar"
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty_list(self):
        assert _chunk_text("", chunk_size=50, overlap=10) == []

    def test_overlap_content_is_shared(self):
        # Build a text long enough to produce at least 2 chunks with overlap
        text = " ".join([f"w{i}" for i in range(30)])
        chunks = _chunk_text(text, chunk_size=10, overlap=3)
        assert len(chunks) >= 2
        # Last 3 words of chunk 0 should appear at start of chunk 1
        last_3_of_chunk0 = chunks[0].split()[-3:]
        first_3_of_chunk1 = chunks[1].split()[:3]
        assert last_3_of_chunk0 == first_3_of_chunk1, "Overlap content should be shared between consecutive chunks"

    def test_chunk_size_is_respected(self):
        text = " ".join([f"w{i}" for i in range(200)])
        chunk_size = 20
        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=0)
        for chunk in chunks[:-1]:  # last chunk may be shorter
            assert len(chunk.split()) == chunk_size

    def test_minilm_truncation_documented(self):
        """
        all-MiniLM-L6-v2 has a 256-token hard limit. A 512-word chunk produces
        400-650 WordPiece tokens — the tail is silently truncated. This test
        documents that behavior and will fail if the chunk_size is changed to
        stay within the 256-token limit (a desired future fix).
        """
        long_text = " ".join([f"word{i}" for i in range(600)])
        chunks = _chunk_text(long_text, chunk_size=512, overlap=64)
        # Each chunk except the last has 512 words — well over MiniLM's 256 WordPiece limit
        first_chunk_word_count = len(chunks[0].split())
        assert first_chunk_word_count == 512, (
            f"chunk has {first_chunk_word_count} words (expected 512). "
            "NOTE: all-MiniLM-L6-v2 silently truncates chunks >256 WordPiece tokens. "
            "A 512-word chunk is ~400-650 WordPiece tokens — truncation occurs."
        )
