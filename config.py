"""
config.py — Single source of truth for model names and shared pipeline constants.

WHY a shared config module instead of inline constants:
    EMBED_MODEL is defined in both ingest.py (builds the vector index) and
    retrieve.py (embeds the query). They MUST use the same model — different
    embedding models produce incompatible vector spaces, so a mismatch silently
    returns wrong results with no error. Defining it once here enforces that
    invariant instead of relying on two files staying in sync manually.
"""

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

# Used in both ingest.py and retrieve.py — must stay in sync or retrieval breaks.
# WHY all-MiniLM-L6-v2:
#   Fast on CPU (~80ms per batch), compact 384-dim vectors, strong on semantic
#   similarity tasks (STS Benchmark). Known limitation: 256 WordPiece token input
#   limit. A 512-word chunk can be 400–650 WordPiece tokens — the tail is silently
#   truncated during embedding. In practice, the most relevant content tends to be
#   at the start of a chunk, so the impact is limited. But it means large chunks
#   (chunk_size > ~190 words) are partially ignored by the embedding model.
#   See docs/tradeoffs.md for the full analysis.
EMBED_MODEL = "all-MiniLM-L6-v2"

# Cross-encoder used for the optional reranking pass.
# WHY cross-encoder instead of a second bi-encoder for reranking:
#   Bi-encoders (like all-MiniLM) encode query and document independently, then
#   compare their vectors. Cross-encoders take (query, document) as a pair and
#   score them jointly — the model can attend to both texts simultaneously, which
#   gives more accurate relevance scores. The tradeoff is speed: cross-encoders
#   are ~5–10x slower. So we run the bi-encoder over the full corpus first,
#   take the top_k * 3 candidates, then rerank that smaller pool.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Default generation model (Ollama).
DEFAULT_OLLAMA_MODEL = "llama3.2"

# ---------------------------------------------------------------------------
# Chroma constants — must match between ingest.py and retrieve.py
# ---------------------------------------------------------------------------

COLLECTION_NAME = "study_notes"
DEFAULT_DB_PATH = "./chroma_db"

# ---------------------------------------------------------------------------
# Default chunking parameters
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 512   # words per chunk
DEFAULT_OVERLAP = 64       # overlap words shared between adjacent chunks
# WHY 64/512 = 12.5% overlap:
#   Overlap ensures that context spanning a chunk boundary is captured by at
#   least one chunk on each side. Too little overlap (< 5%) misses boundary
#   context; too much (> 30%) bloats the index without adding retrieval benefit.
#   12.5% is a common starting point — see chunk_experiment.py for empirical
#   comparison across chunk sizes.
