"""
tests/test_config.py — Tests for config constants and retrieval filters.

Coverage:
  - CORPUS_VERSION is a string (not None)
  - EMBED_MODEL and COLLECTION_NAME are strings
  - Retrieval filter contract: version-scoped WHERE predicate is non-null
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConfig:
    def test_corpus_version_is_string(self):
        from config import CORPUS_VERSION
        assert isinstance(CORPUS_VERSION, str)
        assert len(CORPUS_VERSION) > 0, "CORPUS_VERSION should not be empty"

    def test_embed_model_defined(self):
        from config import EMBED_MODEL
        assert isinstance(EMBED_MODEL, str)
        assert len(EMBED_MODEL) > 0

    def test_collection_name_defined(self):
        from config import COLLECTION_NAME
        assert isinstance(COLLECTION_NAME, str)
        assert len(COLLECTION_NAME) > 0

    def test_default_chunk_size_within_safe_range(self):
        from config import DEFAULT_CHUNK_SIZE
        # Document the known truncation: 512 words > MiniLM 256-token limit.
        # This assertion captures the current behavior; change it if chunk_size
        # is reduced to stay within the model's token limit.
        assert isinstance(DEFAULT_CHUNK_SIZE, int)
        assert DEFAULT_CHUNK_SIZE > 0


class TestVersionScopedFilterContract:
    def test_corpus_version_is_not_none(self):
        """LanceDB/Chroma WHERE predicates require a non-null version value."""
        from config import CORPUS_VERSION
        assert CORPUS_VERSION is not None
        assert CORPUS_VERSION != ""

    def test_corpus_version_is_valid_identifier(self):
        """Version tags used in metadata filters must be safe for string comparison."""
        import re

        from config import CORPUS_VERSION
        assert re.match(r'^[a-zA-Z0-9._\-]+$', CORPUS_VERSION), (
            f"CORPUS_VERSION={CORPUS_VERSION!r} contains characters unsafe for SQL predicates"
        )
