"""
api.py — FastAPI wrapper for the RAG pipeline.

Provides a single POST /query endpoint that accepts a question and returns
the answer with full observability metadata (retrieval strategy, fallback
events, artifact versions).

Pydantic schemas enforce a stable request/response contract so callers
(eval harness, integration tests, frontend) can depend on field presence.

Startup:
    uvicorn api:app --reload

Or via Makefile:
    make serve
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import (
    CORPUS_VERSION,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K,
    EMBED_MODEL,
)
from generate import generate_async
from retrieve import retrieve, validate_collection

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Validated incoming query parameters."""

    query: str = Field(..., min_length=1, description="The question to answer.")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Chunks to retrieve.")
    rerank: bool = Field(False, description="Apply cross-encoder reranking.")
    strategy: Literal["dense", "bm25", "auto"] = Field(
        "auto",
        description=(
            "'auto' uses dense retrieval with automatic BM25 fallback when the "
            "top dense score is below config.retrieval.bm25_fallback_threshold. "
            "'bm25' forces BM25 keyword retrieval. 'dense' forces dense-only."
        ),
    )


class RetrievalMetadata(BaseModel):
    strategy: str
    top_score: float
    embedding_model: str
    corpus_version: str
    reranked: bool
    retrieval_latency_ms: int


class QueryResponse(BaseModel):
    """Stable response contract for the /query endpoint."""

    query: str
    answer: str
    chunks_used: int
    sources: list[str]

    # Artifact traceability — lets downstream know exactly which model and
    # corpus version produced this answer. Useful for debugging regressions.
    embedding_model: str
    corpus_version: str

    # Retrieval observability — what strategy was used, the top confidence
    # score, and how long retrieval took.
    retrieval_metadata: RetrievalMetadata

    # All degradation events across retrieval + generation stages.
    # Empty list = happy path. Populated = something fell back.
    # Example values: "bm25_fallback_triggered", "ollama_timeout_30s",
    # "retrieval_sla_exceeded_600ms"
    fallback_events: list[str]

    latency: dict


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks before the app starts serving requests."""
    # Non-fatal: app starts even if collection is missing, but logs a clear
    # warning so operators know to run ingest.py before querying.
    validate_collection()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Pipeline — Baseline",
    description=(
        "Research-quality RAG pipeline: dense retrieval (all-MiniLM-L6-v2) + "
        "optional cross-encoder reranking + BM25 fallback + Ollama generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Answer a question using the RAG pipeline.

    Retrieval strategy:
    - 'auto' (default): dense retrieval with automatic BM25 fallback if top
      dense score < config.retrieval.bm25_fallback_threshold
    - 'bm25': force BM25 keyword retrieval
    - 'dense': force dense-only (no BM25 fallback)

    Response includes full observability metadata: retrieval strategy, top
    confidence score, fallback events, artifact versions, and latency breakdown.
    """
    import time
    t0 = time.time()

    try:
        # Retrieval — returns (chunks, retrieval_info)
        chunks, retrieval_info = retrieve(
            query=req.query,
            top_k=req.top_k,
            rerank=req.rerank,
            strategy=req.strategy,
        )
    except RuntimeError as e:
        # Collection not found — ingest.py hasn't been run yet.
        raise HTTPException(status_code=503, detail=str(e))

    t_retrieve = time.time()

    # Async generation so the event loop isn't blocked by the Ollama HTTP call.
    # WHY asyncio.to_thread: Ollama is single-threaded; blocking here starves
    # all concurrent requests. generate_async() wraps the blocking call.
    answer, gen_fallback_events = await generate_async(req.query, chunks, DEFAULT_OLLAMA_MODEL)
    t_generate = time.time()

    all_fallback_events = retrieval_info.get("fallback_events", []) + gen_fallback_events

    return QueryResponse(
        query=req.query,
        answer=answer,
        chunks_used=len(chunks),
        sources=list({c["source"] for c in chunks}),
        embedding_model=EMBED_MODEL,
        corpus_version=CORPUS_VERSION,
        retrieval_metadata=RetrievalMetadata(
            strategy=retrieval_info["strategy"],
            top_score=retrieval_info["top_score"],
            embedding_model=retrieval_info["embedding_model"],
            corpus_version=retrieval_info["corpus_version"],
            reranked=retrieval_info["reranked"],
            retrieval_latency_ms=retrieval_info["retrieval_latency_ms"],
        ),
        fallback_events=all_fallback_events,
        latency={
            "retrieve_ms": round((t_retrieve - t0) * 1000),
            "generate_ms": round((t_generate - t_retrieve) * 1000),
            "total_ms": round((t_generate - t0) * 1000),
        },
    )


@app.get("/health")
def health() -> dict:
    """Liveness check. Returns 200 if the app is running."""
    return {"status": "ok", "corpus_version": CORPUS_VERSION, "embedding_model": EMBED_MODEL}
