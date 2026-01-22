"""
config.py — Load pipeline parameters from config.yaml and expose them as
module-level constants for backward compatibility.

WHY load from YAML instead of defining constants inline:
    EMBED_MODEL must be identical in ingest.py (builds the index) and
    retrieve.py (embeds queries). Different models produce incompatible vector
    spaces — a mismatch silently returns wrong results with no error. A single
    source of truth (config.yaml) enforces that invariant instead of relying
    on two files staying in sync manually.

    All other tuneable parameters (chunk_size, bm25_fallback_threshold, etc.)
    are co-located in config.yaml so operators can see the full parameter set
    in one place without reading Python source.

See config.yaml for parameter documentation and tuning guidance.
"""

import os

import yaml

# ---------------------------------------------------------------------------
# Load config.yaml from the same directory as this file
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Embedding / indexing
# ---------------------------------------------------------------------------

# Used in both ingest.py and retrieve.py — must stay in sync or retrieval
# breaks silently. See config.yaml → embedding.model for tuning guidance.
EMBED_MODEL: str = _cfg["embedding"]["model"]

# Cross-encoder used for the optional reranking pass.
# WHY cross-encoder instead of a second bi-encoder for reranking:
#   Bi-encoders encode query and document independently. Cross-encoders take
#   (query, document) as a pair and score them jointly — the model can attend
#   to both texts simultaneously, which gives more accurate relevance scores.
#   Tradeoff: ~5–10× slower. So we run bi-encoder over the full corpus first,
#   take top_k * 3 candidates, then rerank that smaller pool.
RERANK_MODEL: str = _cfg["retrieval"]["rerank_model"]

# Default chunking parameters — re-ingest required if you change these.
DEFAULT_CHUNK_SIZE: int = _cfg["embedding"]["chunk_size"]
DEFAULT_OVERLAP: int = _cfg["embedding"]["chunk_overlap"]

# WHY 64/512 = 12.5% overlap:
#   Overlap ensures context spanning a chunk boundary is captured by at least
#   one chunk on each side. Too little overlap (<5%) misses boundary context;
#   too much (>30%) bloats the index without adding retrieval benefit.
#   12.5% is a common starting point — see chunk_experiment.py for empirical
#   comparison across chunk sizes.

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: int = _cfg["retrieval"]["top_k"]
BM25_FALLBACK_THRESHOLD: float = _cfg["retrieval"]["bm25_fallback_threshold"]
RETRIEVAL_LATENCY_SLA_MS: int = _cfg["retrieval"]["latency_sla_ms"]

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_MODEL: str = _cfg["generation"]["model"]
MAX_CONTEXT_WORDS: int = _cfg["generation"]["context_budget_words"]
OLLAMA_TIMEOUT_SECONDS: int = _cfg["generation"]["timeout_seconds"]

# ---------------------------------------------------------------------------
# Corpus versioning
# ---------------------------------------------------------------------------

CORPUS_VERSION: str = _cfg["corpus"]["version"]

# ---------------------------------------------------------------------------
# Chroma constants — must match between ingest.py and retrieve.py
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = _cfg["chroma"]["collection_name"]
DEFAULT_DB_PATH: str = _cfg["chroma"]["db_path"]
