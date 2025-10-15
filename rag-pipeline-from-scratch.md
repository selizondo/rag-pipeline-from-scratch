---

## Staff-Level Review: rag-pipeline-from-scratch

### Executive Summary

**Project Scope:** Intentionally minimal, research-quality RAG baseline  
**Intent:** Transparent learning tool (not production-ready)  
**Architecture:** Ingest → Retrieve (Chroma + optional rerank) → Generate (Ollama)  
**Corpus:** 2,700+ Q&A from HuggingFace + Kaggle  
**Stack:** Python, sentence-transformers, Chroma, cross-encoder, Ollama  

---

### **1. Architecture & Dataflow**

The pipeline follows a classic three-stage RAG structure:

```
[Markdown corpus] → [word-chunking + embedding] → [Chroma vector store]
                                                           ↓
                          [Query embedding] → [cosine search] → [rerank?] → [top-K chunks]
                                                                                  ↓
                          [Context window trim] → [Ollama generation] → [answer + latency]
```

**Positives:**
- Clean module separation: [ingest.py](c:\Users\selizondo\projects\selizondo\coding_labs\rag-pipeline-from-scratch\ingest.py) → [retrieve.py](c:\Users\selizondo\projects\selizondo\coding_labs\rag-pipeline-from-scratch\retrieve.py) → [generate.py](c:\Users\selizondo\projects\selizondo\coding_labs\rag-pipeline-from-scratch\generate.py)
- Transparent component boundaries (no framework abstractions)
- Reranking integrated as optional pipeline stage (2-pass retrieval: dense → optional cross-encoder)

**Gaps:**
- No API boundary or request-response contract (entrypoint is CLI script)
- No version constants or schema for reproducibility across runs
- No fallback orchestration (if one stage fails, entire pipeline fails)

---

### **2. Production-Readiness Assessment**

Per the README: **"intentionally minimal and not meant to be production-ready."** This is honest scoping, but here's what would be needed to cross that line:

| Pattern | Status | Gap |
|---------|--------|-----|
| **Non-Fatal Degradation** | ❌ Missing | Ollama down crashes; Chroma missing returns empty silently; embedding model not cached |
| **Contract-First Design** | ❌ Missing | Model names + chunk params inline in code; no schema validation |
| **Observability Fields** | ⚠️ Partial | Latency present, but missing `retrieval_source`, `embedding_version`, `fallback_reason` |
| **Failure Modes in Design** | ❌ Missing | No skew detection, stale feature detection, or adaptive thresholds |
| **Explicit Scale Boundaries** | ❌ Missing | Chroma performance untested; no documented max corpus size or latency SLA |
| **Baselines + Evaluation** | ⚠️ Partial | Chunk experiment exists but isolated; no baseline (random chunking, popularity, BM25) |

---

### **3. Error Handling & Degradation**

**Current State:**
```python
# generate.py — Hard failure if Ollama unreachable
response = requests.post(OLLAMA_URL, ...)
response.raise_for_status()  # Throws HTTPError if Ollama returns 5xx
```

```python
# retrieve.py — Silent empty return if collection missing
collection = client.get_collection(COLLECTION_NAME)  # Raises if not found
# No try/except
```

```python
# ingest.py — Catches collection drop but no logging
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass  # Silently ignores if collection doesn't exist
```

**Recommendation:**
- Wrap `requests.post()` in try/except; fall back to stub answer (e.g., "Unable to generate; here are the sources: [list]")
- Validate Chroma collection exists at startup; warn if not, suggest re-ingest
- Log all fallback events to response field: `{"fallback_reason": "ollama_timeout"}`
- Document timeouts and retry budgets

---

### **4. Chunking Strategy & Trade-Offs**

**Implementation: [ingest.py](c:\Users\selizondo\projects\selizondo\coding_labs\rag-pipeline-from-scratch\ingest.py#L24-L36)**
```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
```

**Decision Rationale:** Word-based splitting "preserves sentence boundaries better than character-based."

**Assessment:**
- ✅ **Predictable**: Fixed word count ensures consistent chunk size
- ⚠️ **Incomplete preservation**: Word boundaries ≠ sentence boundaries. A word-based split can still break mid-sentence if sentence length doesn't align with chunk size
- ❌ **No empirical tuning**: Overlap ratio (12.5% = 64 words / 512 base) derived from defaults, not from retrieval quality experiments

**What [chunk_experiment.py](c:\Users\selizondo\projects\selizondo\coding_labs\rag-pipeline-from-scratch\chunk_experiment.py) measures:**
- Compares chunk sizes [256, 512, 1024] on 3 hardcoded queries
- Metric: average retrieval score (1 - cosine distance)
- **Gap**: No ground truth labels. Cannot measure if top-K chunks actually answer the query.

**Recommendations:**
1. **Measure faithfulness, not just retrieval score**: For each query, label top-K chunks manually as "answers query" / "doesn't answer" → compute precision@K as ground truth
2. **Test sentence-boundary-aware chunking**: Use `nltk.sent_tokenize()` + resolution-content promotion (known RAG pattern)
3. **Automate chunk_experiment.py into CI**: Run on corpus updates to detect degradation
4. **Document scale boundaries**: e.g., "word-based chunking tested up to 50K chunks; performance untested at 1M+"

---

### **5. ML Data Quality Lens**

**Corpus Construction:**
- Sources: Shlok307 AI Q&A (~1.8k), mjphayes ML Q&A (~600), Kaggle Q&A (~500)
- Quality filtering: Length check (answer ≥ 40 chars) and keyword filtering (ML keywords for mjphayes)
- **Issue**: Downloaded dynamically from HuggingFace/Kaggle. No frozen baseline → corpus changes on reruns

**Train/Val/Test Split:**
- ❌ **None**. Chunk experiment queries (3 hardcoded examples) used both for parameter tuning and evaluation
- **Label leakage risk**: If chunk_experiment results informed chunk size choice, evaluation is biased

**Baseline Comparisons:**
- ❌ **None**. No measurement of what random chunking, popularity-only retrieval, or BM25 alone would achieve
- Example: Does dense retrieval + cross-encoder rerank actually beat BM25? Unknown.

**Evaluation Reproducibility:**
- ✅ Partial: [chunk_experiment_results.json](c:\Users\selizondo\projects\selizondo\coding_labs\rag-pipeline-from-scratch\chunk_experiment_results.json) persisted, but:
  - Corpus version not recorded (hardcoded path: `/Users/selizondo/Dropbox/...`)
  - Model versions not in artifact (embedding + reranking models inline in code)
  - Queries hardcoded (cannot generalize to new queries)

**Recommendations:**
1. **Freeze corpus**: Store canonical corpus in repo or Docker image; version it (e.g., `corpus_v1`)
2. **Create test split**: Hold out 10–20% of Q&A pairs for evaluation; use only during final reporting
3. **Add baseline evaluation**: Measure BM25 alone, random chunk ordering, popularity
4. **Version model choices**: Store in config file (not inline):
   ```yaml
   embedding_model: all-MiniLM-L6-v2
   rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
   chunk_size: 512
   overlap_words: 64
   corpus_version: v1
   ```
5. **Compute retrieval@K metrics**: For each query, manually label top-5 as "helpful" / "not helpful" → report precision@5, recall@5

---

### **6. LLM/RAG-Specific Patterns**

**Faithfulness vs. Relevancy Tradeoff:**
- **Current**: Dense (all-MiniLM) + optional cross-encoder reranking (ms-marco)
- **What's missing**: BM25 baseline (typically achieves higher faithfulness)
- **Pattern gap**: No measurement of whether retrieved context is verbatim-grounded vs. semantically drifted

**Recommendation:**
```python
# Add to retrieve.py
def retrieve_hybrid(query: str, top_k: int, db_path: str) -> list[dict]:
    """Return chunks scored on both dense + BM25 metrics."""
    dense_chunks = retrieve_dense(query, top_k * 2, db_path)
    bm25_chunks = retrieve_bm25(query, top_k * 2)  # requires BM25 index
    
    # Hybrid: favor chunks in both dense AND BM25 top-K
    # Return combined with reason: "dense_only", "bm25_only", "hybrid"
```

**Version-Scoped Filtering:**
- ❌ **Missing**: No metadata filtering before ANN ranking
- **Current pattern**: Store all chunks, rank by similarity, trim context
- **Problem**: Cannot update corpus without re-ingesting; no rollback story
- **Fix**: Add `corpus_version` metadata; filter `WHERE corpus_version = "v1"` before ANN

**Context Window Budget:**
```python
MAX_CONTEXT_WORDS = 1500  # "stay well within Ollama context window"
```
- ❌ **Arbitrary**: No calculation shown
- **Fix**: 
  ```python
  # With llama3.2, typical context window is 8k tokens
  # Prompt overhead: ~200 tokens (system + query)
  # per-word ≈ 1.3 tokens, so 1500 words ≈ 1950 tokens
  # Safe margin: 8000 - 200 - 1950 = 5850 tokens spare (good)
  OLLAMA_CONTEXT_TOKENS = 8000
  PROMPT_OVERHEAD_TOKENS = 200
  MAX_CONTEXT_TOKENS = OLLAMA_CONTEXT_TOKENS - PROMPT_OVERHEAD_TOKENS
  WORDS_PER_TOKEN = 1.3
  MAX_CONTEXT_WORDS = int(MAX_CONTEXT_TOKENS / WORDS_PER_TOKEN)
  ```

**Judge Model Calibration:**
- ❌ **Not applicable**: No judge model used for evaluation (only for generation)
- **If you add evaluation**: Use llama3.2 as judge? Document that absolute scores shift with stronger models; report as deltas

**Adaptive Fallback Threshold:**
- ❌ **Missing**: No automatic BM25 fallback based on dense score threshold
- **Pattern**: If `dense_score < 0.3`, switch retrieval strategy (common in production)
- **Recommendation**: Add config param + document tuning range

---

### **7. Observability & Response Contract**

**Current Output:**
```python
return {
    "query": query,
    "answer": answer,
    "chunks_used": len(chunks),
    "sources": list({c["source"] for c in chunks}),
    "latency": {
        "retrieve_ms": round((t_retrieve - t0) * 1000),
        "generate_ms": round((t_generate - t_retrieve) * 1000),
        "total_ms": round((t_generate - t0) * 1000),
    },
}
```

**Strengths:**
- ✅ Latency breakdown (retrieve vs. generate)
- ✅ Source tracking (caller can audit grounding)

**Missing Observability Fields:**
- `embedding_model`, `chunk_version`, `rerank_applied`
- `retrieval_source` (dense? bm25? hybrid?)
- `top_k_scores` (caller can see if answer is high-confidence)
- `fallback_reason` (if Ollama timed out, BM25 used, etc.)

**Recommendation:**
```python
return {
    "query": query,
    "answer": answer,
    "chunks_used": len(chunks),
    "sources": list({c["source"] for c in chunks}),
    "retrieval_metadata": {
        "strategy": "dense_rerank",  # or "bm25_only", "hybrid", etc.
        "top_score": chunks[0]["score"] if chunks else 0,
        "embedding_model": "all-MiniLM-L6-v2",
        "corpus_version": "v1",
    },
    "latency": {...},
    "fallback_events": [],  # e.g., ["ollama_timeout_retry_1", "bm25_fallback"]
}
```

---

### **8. Anti-Patterns & Concrete Recommendations**

| Anti-Pattern | Current | Fix |
|--------------|---------|-----|
| **Magic numbers in thresholds** | `MAX_CONTEXT_WORDS = 1500` (no justification) | Document calculation: `(context_tokens - overhead) / words_per_token` |
| **Model version constants scattered** | `EMBED_MODEL = "all-MiniLM-L6-v2"` in both [retrieve.py](c:\Users\selizondo\projects\selizondo\coding_labs\rag-pipeline-from-scratch\retrieve.py#L11) and [ingest.py](c:\Users\selizondo\projects\selizondo\coding_labs\rag-pipeline-from-scratch\ingest.py#L14) | Create `config.yaml` with single source of truth |
| **Observability added post-incident** | No observability at all (this is baseline, so OK) | Design in: latency, retrieval_source, fallback_reason from day 1 |
| **Tests only cover happy path** | 0 tests | Add: test Ollama down → graceful degradation, test empty corpus → sensible error |
| **Non-reproducible evaluation** | chunk_experiment hardcoded corpus path, no artifact versioning | Freeze corpus + versions in artifact, commit results to repo |
| **Hard dependencies that crash API** | `requests.raise_for_status()` throws if Ollama unreachable | Wrap in try/except, fall back to "Unable to generate" |

---

### **9. Recommended Roadmap**

**Phase 1: Reproducibility (foundation)**
- [ ] Move magic numbers to [config.yaml](config.yaml): embedding model, chunk size, overlap, context budget
- [ ] Freeze corpus (pin HuggingFace/Kaggle versions or download to repo)
- [ ] Version all artifacts: `embedding_model`, `corpus_version` in response
- [ ] Add request-response schemas (e.g., Pydantic)

**Phase 2: Observability (operational)**
- [ ] Add fallback_events to response (track Ollama timeouts, BM25 switches)
- [ ] Add retrieval_metadata: strategy, top_score, embedding_model
- [ ] Add latency SLA checks (e.g., warn if retrieve > 500ms)

**Phase 3: Graceful Degradation (resilience)**
- [ ] Wrap Ollama call in try/except; return best-effort answer (e.g., "Sources: [list]")
- [ ] Validate Chroma collection at startup; warn if missing
- [ ] Add BM25 fallback: if dense_score < threshold, switch retrieval strategy

**Phase 4: Evaluation (rigor)**
- [ ] Freeze test set (10% of corpus, held out)
- [ ] Compute retrieval@K metrics (precision, recall) against labeled ground truth
- [ ] Add baseline measurements (BM25 alone, random chunking, popularity)
- [ ] Integrate chunk_experiment into CI; auto-run on corpus updates

**Phase 5: Production-Ready (scaling)**
- [ ] Add API boundary (FastAPI wrapper)
- [ ] Implement version-scoped filtering (can roll back corpus without re-ingest)
- [ ] Document scale boundaries: max corpus size, latency SLAs at scale
- [ ] Add async generation (Ollama is single-threaded)

---

### **10. Summary**

**What This Project Does Well:**
- Clear, learnable architecture (no framework abstractions)
- Transparent chunking strategy with empirical comparison
- Clean module boundaries
- Honest about scope ("research-quality baseline")

**What's Missing (Intentionally or Not):**
- **Non-fatal degradation**: Any dependency failure crashes the pipeline
- **Versioning**: No way to track reproducibility across runs or corpus updates
- **Evaluation rigor**: No test split, no baselines, no ground truth labels
- **Observability**: Response lacks retrieval strategy, model versions, fallback events
- **RAG-specific patterns**: No BM25 fallback, no adaptive thresholds, no judge model

**Positioning:**
- ✅ **Excellent** as a learning/teaching tool (transparent, minimal, no abstractions)
- ❌ **Not ready** as production service (no error handling, no observability, no evaluation framework)
- 🟡 **With Phase 1–2 work**, could become a solid open-source baseline for the community

**Effort to Production-Ready:** ~2–3 weeks (Phases 1–4)