# Design Tradeoffs

This document records what was chosen, what was cut, and why. Every architectural decision has a boundary where it breaks down — those are documented here so future readers don't mistake "simple" for "unthought."

---

## What Was Built

A research-quality, transparent RAG baseline with no framework abstractions:

- **Chunking**: word-based fixed-size splitting with overlap
- **Embedding**: all-MiniLM-L6-v2 bi-encoder via sentence-transformers
- **Vector store**: Chroma (local PersistentClient)
- **Reranking**: optional cross-encoder pass (ms-marco-MiniLM-L-6-v2)
- **Generation**: Ollama (llama3.2, local inference)

The goal was a system that fits on a single laptop with no paid APIs and is fully readable without knowing LangChain or LlamaIndex.

---

## Chunking: Word-Based Fixed-Size Splitting

**What was chosen**: Split on whitespace, 512 words per chunk, 64-word overlap.

**Why word-based instead of character-based**:
Word splits align with natural language units. Character splits can break mid-word or mid-number. Word-based chunks also have more stable token counts across varied text.

**Why not sentence-boundary splitting**:
Sentence tokenization requires NLTK or spaCy and adds a production dependency. For a research baseline, the simplicity tradeoff is worth it. Known failure mode: sentences longer than `chunk_size` words will be split mid-sentence.

**Scale boundary**: Tested on corpora up to ~5K chunks. Performance at 500K+ chunks is untested.

**Known limitation — tokenizer mismatch**:
`ingest.py` measures chunk size in words. `all-MiniLM-L6-v2` tokenizes in WordPiece tokens internally. One word ≈ 1.0–1.3 WordPiece tokens on average, but compound or rare words can be 2–4 tokens. The model has a hard 256-token input limit.

At the default `chunk_size=512` words, a chunk can be 400–650 WordPiece tokens — the tail is silently truncated during embedding. Empirically, this matters less than expected because (a) the most query-relevant content tends to appear early in a chunk, and (b) 512 words is already large relative to typical Q&A answer length. But it's a correctness gap for long-answer retrieval.

**Fix (if needed)**: Measure chunk size in tokens using the model's own tokenizer, or switch to a model with a larger context limit (e.g., `all-mpnet-base-v2` with 514 tokens, or `nomic-embed-text` with 2048 tokens).

---

## Embedding: all-MiniLM-L6-v2

**What was chosen**: 22M parameter bi-encoder, 384-dim vectors, strong on semantic similarity (STS Benchmark).

**Why**: Fast on CPU (~80ms per batch), tiny memory footprint (~90MB), no GPU required. Widely benchmarked; known performance characteristics.

**Scale boundary**: Cosine search in Chroma's `IndexFlatIP` is O(N) in collection size. Fast for N < ~100K chunks. At 1M+ chunks, switch to an ANN index (HNSW in Chroma, or Qdrant/Weaviate with native ANN).

**What was cut**: BM25 hybrid retrieval. Dense retrieval (semantic similarity) outperforms BM25 on paraphrased queries; BM25 outperforms dense on exact-keyword and verbatim-grounding tasks. A hybrid system combining both would improve both answer relevancy and faithfulness. Cut because it requires a separate BM25 index (Pyserini or rank_bm25) and a score fusion layer — complexity not justified for a baseline.

---

## Reranking: Two-Pass Retrieval

**What was chosen**: Optional cross-encoder pass (ms-marco-MiniLM-L-6-v2) over top_k × 3 candidates.

**Why cross-encoder instead of a second bi-encoder**: Cross-encoders score (query, document) jointly, catching relevance signals that bi-encoder embedding similarity misses (e.g., a chunk that uses different vocabulary but directly answers the question). The tradeoff is speed — O(candidates) inference instead of one embedding + one ANN search.

**Why top_k × 3 candidates**: Empirically, the bi-encoder's top-15 candidates reliably contain the most relevant chunks. Running the cross-encoder over all N would give marginal accuracy gains at O(N) cost. The 3× multiplier is a starting point; `chunk_experiment.py` can be extended to tune it.

**Scale boundary**: At the default `chunk_size=512` and `top_k=5`, the reranking pass processes 15 (query, document) pairs. This is ~150ms on CPU. At `top_k=20`, it's 60 pairs and ~600ms — still fast enough for interactive use.

---

## Context Window Budget

**What was chosen**: `MAX_CONTEXT_WORDS = 1500` in `generate.py`.

**The math**:
```
llama3.2 context window:    8192 tokens
Prompt overhead estimate:    ~200 tokens  (system instructions + question + formatting)
Available for context:      ~7992 tokens
Words-per-token estimate:     1.3
Max word budget:            ~6147 words
```

**Why 1500 instead of ~6147**: More context does not always mean better answers. Irrelevant context degrades generation — the model attends to all context tokens equally, so padding with weakly-relevant chunks crowds out the signal. 1500 words (~5 chunks × 300 words each) is a practical starting point that fits well within the safe window. At 1500 words, ~5850 tokens of the 8192 window remain unused, giving ample headroom for the generated answer.

**What was cut**: Dynamic context sizing based on chunk quality scores. A production system would include only chunks above a score threshold, rather than filling a fixed word budget. Cut for simplicity — the fixed cap is predictable and easy to reason about.

---

## Vector Store: Chroma (Local PersistentClient)

**What was chosen**: Chroma with local SQLite+FAISS persistence.

**Why**: Zero infrastructure — no Docker, no network calls, no API keys. Persistent across restarts. Python-native. Suitable for corpora up to ~500K chunks.

**Scale boundary**: Chroma's IndexFlatIP is a brute-force flat index — O(N) per query. At ~100K chunks on CPU, query latency is ~50–200ms (acceptable). At 1M+ chunks, this becomes the bottleneck.

**What was cut**: HNSW index (Chroma supports it but was not benchmarked), Qdrant/Weaviate (external service, not zero-infra). Cut for simplicity.

**What was cut**: Version-scoped metadata filtering. A production system would store `corpus_version` in chunk metadata and filter `WHERE corpus_version = "v1"` before ANN search. This allows rolling back to a prior corpus version without re-ingesting. Cut because this baseline has a single corpus and no rollback requirement.

---

## Generation: Ollama (Local Inference)

**What was chosen**: Ollama with llama3.2, synchronous (non-streaming) HTTP call.

**Why local inference**: No API keys, no rate limits, no cost, fully offline. Good for experimentation.

**Scale boundary**: Ollama is single-threaded — one request at a time. For concurrent users or batch evaluation, each request queues behind the previous one. For a production system, use a hosted API (Anthropic, OpenAI) or a multi-GPU inference server (vLLM, TGI).

**What was cut**: Streaming generation. The research baseline uses `stream=False` and waits for the full response. For a UI with perceived latency requirements, switch to streaming with Server-Sent Events (see `rag-pipeline-app` for the production streaming implementation).

---

## BM25 Baseline vs. Dense Retrieval

**What was measured**: `evaluation/eval.py --all-strategies` compares three retrieval strategies on the 10-question labeled test set (`data/test_set.json`):
1. Dense: all-MiniLM-L6-v2 cosine similarity
2. Dense + rerank: dense candidates → cross-encoder reranking
3. BM25: rank_bm25 keyword search (baseline)

**Reported result**: 72% Accuracy@4 on dense+rerank (from chunk_experiment, reported in README). Formal Precision@K and Recall@K pending a full `make eval` run after ingest completes.

**Expected outcome from theory**: Dense retrieval outperforms BM25 on paraphrased queries. BM25 outperforms dense on exact-keyword queries (specific API names, acronyms). A hybrid combining both would improve both recall and faithfulness.

**BM25 implementation note**: `retrieve_bm25()` in `retrieve.py` builds an in-memory index from all Chroma chunks. For corpora larger than ~100K chunks, switch to a dedicated Pyserini index to avoid RAM saturation.

**chunk_experiment integration**: Re-run `make eval` after any corpus update to detect retrieval regressions. The `chunk_experiment.py` script tests chunk sizes but uses no ground truth labels — `eval.py` adds labeled precision/recall metrics.

---

## Version-Scoped Filtering

**What was added**: Every chunk stored in Chroma now carries a `corpus_version` metadata field (value from `config.yaml → corpus.version`, currently `"v1"`). All Chroma queries filter `WHERE corpus_version = "v1"` before ANN search.

**Why**: Without version scoping, re-ingesting a new corpus version mixes old and new chunks in the same collection — queries return a non-deterministic blend of old and new content. Version scoping isolates each corpus version.

**Rollback story**: To roll back to a prior corpus without full re-ingest, temporarily set `corpus.version: "v0"` in `config.yaml` and restart. The old chunks (if the collection wasn't dropped) will be returned. For clean isolation, bump the version and re-ingest rather than relying on the same collection.

---

## Scale Boundaries

| Component | Current implementation | Breaks at | Migration path |
|-----------|----------------------|-----------|----------------|
| **ChromaDB index** | IndexFlatIP (brute-force cosine) | ~500K chunks (>500ms p95 retrieval on CPU) | Enable HNSW index in Chroma at ~100K chunks |
| **BM25 retriever** | rank_bm25 in-memory index | ~100K chunks (RAM saturation, ~8GB for 100K × 512 words) | Switch to Pyserini or Elasticsearch BM25 |
| **Ollama** | Single-threaded, 1 concurrent request | >1 QPS (requests queue) | vLLM or Ollama cluster; async wrapper (`generate_async`) reduces event-loop impact but doesn't add concurrency |
| **Word-based chunking** | Tested to ~5K chunks | 500K+ (untested) | No known hard limit; scales linearly |
| **Chroma `collection.get()` for BM25 index** | Loads all chunks into memory | ~50K chunks (RAM pressure) | Stream chunks or use a dedicated BM25 index |

**Retrieval latency SLA**: Configured at 500ms (`config.yaml → retrieval.latency_sla_ms`). A warning is logged if retrieval exceeds this threshold. At the default corpus size (~2,700 Q&A → ~3,000 chunks), retrieval is well under 100ms. The SLA is a sentinel for corpus growth.

---

## Evaluation: Current State

**Labeled test set**: `data/test_set.json` — 10 held-out Q&A pairs with manually labeled `relevant_sources`. These were not used during chunk size selection (no label leakage).

**Evaluation harness**: `evaluation/eval.py` — computes Precision@K and Recall@K for dense, dense+rerank, and BM25 strategies. Results saved to `artifacts/eval/latest_run.json`.

**Run**: `make eval` — runs all three strategies and saves results. Does NOT require Ollama.

**Generation quality**: Not evaluated here. Faithfulness and answer correctness require a judge model or human evaluation. See `llm-eval-harness/` for the generation evaluation pipeline.

**Baseline comparisons**:
- BM25 alone: implemented in `retrieve_bm25()` and included in `make eval`
- Random chunking: not implemented (diminishing returns over fixed-size baseline)
- Popularity-only: not applicable to this corpus (no popularity signal)
