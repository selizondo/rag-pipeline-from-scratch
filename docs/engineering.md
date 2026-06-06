# Design and Tradeoffs

Every architectural decision has a boundary where it breaks down. This document records what was chosen, what was cut, and why, so future readers do not mistake "simple" for "unthought."

---

## What Was Built

A research-quality, transparent RAG baseline with no framework abstractions:

- **Chunking:** word-based fixed-size splitting with overlap
- **Embedding:** all-MiniLM-L6-v2 bi-encoder via sentence-transformers
- **Vector store:** Chroma (local PersistentClient)
- **Reranking:** optional cross-encoder pass (ms-marco-MiniLM-L-6-v2)
- **Generation:** Ollama (llama3.2, local inference)

The goal: a system that fits on a single laptop with no paid APIs and is fully readable without knowing LangChain or LlamaIndex.

---

## Chunking: Word-Based Fixed-Size Splitting

**What was chosen:** Split on whitespace, 256 words per chunk, 32-word overlap.

**Why word-based instead of character-based:** Word splits align with natural language units. Character splits can break mid-word or mid-number. Word-based chunks also have more stable token counts across varied text.

**Why not sentence-boundary splitting:** Sentence tokenization requires NLTK or spaCy and adds a production dependency. For a research baseline, the simplicity tradeoff is worth it. Known failure mode: sentences longer than `chunk_size` words will be split mid-sentence.

**Tokenizer mismatch gap:** `ingest.py` measures chunk size in words. `all-MiniLM-L6-v2` tokenizes in WordPiece tokens internally. One word averages 1.0 to 1.3 WordPiece tokens, but compound or rare words can be 2 to 4 tokens. The model has a hard 256-token input limit. At the default `chunk_size=256` words, chunks stay safely below the ceiling. At 512 words, tail content is silently truncated during embedding. Fix: measure chunk size in tokens using the model's own tokenizer, or switch to a model with a larger context limit.

**Scale boundary:** Tested on corpora up to ~5K chunks. No known hard limit; scales linearly.

---

## Embedding: all-MiniLM-L6-v2

**What was chosen:** 22M parameter bi-encoder, 384-dim vectors.

**Why:** Fast on CPU (~80ms per batch), tiny memory footprint (~90MB), no GPU required. Widely benchmarked with known performance characteristics on semantic similarity tasks.

**What was cut:** BM25 hybrid retrieval. Dense retrieval outperforms BM25 on paraphrased queries; BM25 outperforms dense on exact-keyword and verbatim-grounding tasks. A hybrid system would improve both answer relevancy and faithfulness. Cut because it requires a separate BM25 index and a score fusion layer, complexity not justified for a baseline. `rag-pipeline-app` implements hybrid search with alpha-weighted score fusion.

**Scale boundary:** Cosine search in Chroma's `IndexFlatIP` is O(N) in collection size. Fast for N below ~100K chunks. At 1M+ chunks, switch to an ANN index.

---

## Reranking: Two-Pass Retrieval

**What was chosen:** Optional cross-encoder pass (ms-marco-MiniLM-L-6-v2) over top_k x 3 candidates.

**Why cross-encoder instead of a second bi-encoder:** Cross-encoders score (query, document) jointly, catching relevance signals that bi-encoder similarity misses. The tradeoff is speed: O(candidates) inference instead of one embedding plus one ANN search.

**Why top_k x 3 candidates:** Empirically, the bi-encoder's top-15 candidates reliably contain the most relevant chunks. Running the cross-encoder over all N gives marginal accuracy gains at O(N) cost. The 3x multiplier is a starting point; `chunk_experiment.py` can be extended to tune it.

**Latency at default settings:** At `chunk_size=256` and `top_k=5`, the reranking pass processes 15 (query, document) pairs, approximately 150ms on CPU. At `top_k=20`, it is 60 pairs and approximately 600ms, still fast enough for interactive use.

---

## Context Window Budget

**What was chosen:** `MAX_CONTEXT_WORDS = 1500` in `generate.py`.

**The math:**

```
llama3.2 context window:     8192 tokens
Prompt overhead estimate:     ~200 tokens  (system instructions + question + formatting)
Available for context:       ~7992 tokens
Words-per-token estimate:      1.3
Max word budget:             ~6147 words
```

**Why 1500 instead of ~6147:** More context does not always mean better answers. Irrelevant context degrades generation because the model attends to all context tokens equally, so padding with weakly-relevant chunks crowds out the signal. 1500 words (~5 chunks x 300 words each) is a practical starting point that fits well within the safe window.

**What was cut:** Dynamic context sizing based on chunk quality scores. A production system would include only chunks above a score threshold rather than filling a fixed word budget. Cut for simplicity: the fixed cap is predictable and easy to reason about.

---

## Vector Store: Chroma (Local PersistentClient)

**What was chosen:** Chroma with local SQLite+FAISS persistence.

**Why:** Zero infrastructure. No Docker, no network calls, no API keys. Persistent across restarts. Python-native. Suitable for corpora up to ~500K chunks.

**What was cut:** HNSW index (Chroma supports it but was not benchmarked), Qdrant/Weaviate (external service, not zero-infra).

**What was cut:** Version-scoped metadata filtering in the initial design. Now added: every chunk carries a `corpus_version` metadata field. All Chroma queries filter `WHERE corpus_version = "v1"` before ANN search. This prevents re-ingesting a new corpus version from mixing old and new chunks in the same collection.

**Rollback story:** To roll back to a prior corpus without full re-ingest, set `corpus.version: "v0"` in `config.yaml` and restart. The old chunks remain in the collection and will be returned. For clean isolation, bump the version and re-ingest rather than relying on the same collection.

---

## Generation: Ollama (Local Inference)

**What was chosen:** Ollama with llama3.2, synchronous (non-streaming) HTTP call.

**Why local inference:** No API keys, no rate limits, no cost, fully offline. Right for experimentation.

**Scale boundary:** Ollama is single-threaded: one request at a time. For concurrent users or batch evaluation, each request queues behind the previous one. For a production system, use a hosted API or a multi-GPU inference server (vLLM, TGI).

**What was cut:** Streaming generation. The research baseline uses `stream=False` and waits for the full response. For a UI with perceived latency requirements, switch to streaming with Server-Sent Events. See `rag-pipeline-app` for the production streaming implementation.

---

## Scale Boundaries

| Component | Current implementation | Breaks at | Migration path |
|-----------|----------------------|-----------|----------------|
| ChromaDB index | IndexFlatIP (brute-force cosine) | ~500K chunks (>500ms p95 retrieval on CPU) | Enable HNSW index in Chroma at ~100K chunks |
| BM25 retriever | rank_bm25 in-memory index | ~100K chunks (RAM saturation, ~8GB for 100K x 512 words) | Switch to Pyserini or Elasticsearch BM25 |
| Ollama | Single-threaded, 1 concurrent request | >1 QPS (requests queue) | vLLM or Ollama cluster |
| Word-based chunking | Tested to ~5K chunks | 500K+ (untested) | No known hard limit; scales linearly |
| Chroma `collection.get()` for BM25 | Loads all chunks into memory | ~50K chunks (RAM pressure) | Stream chunks or use a dedicated BM25 index |

**Retrieval latency SLA:** Configured at 500ms (`config.yaml → retrieval.latency_sla_ms`). A warning is logged if retrieval exceeds this threshold. At the default corpus size (~2,700 Q&A, ~3,000 chunks), retrieval is well under 100ms. The SLA is a sentinel for corpus growth.

---

## Evaluation Design

**Labeled test set:** `data/test_set.json` contains 10 held-out Q&A pairs with manually labeled `relevant_sources`. These were not used during chunk size selection (no label leakage).

**Evaluation harness:** `evaluation/eval.py` computes Precision@K and Recall@K for dense, dense+rerank, and BM25 strategies. Results saved to `artifacts/eval/latest_run.json`.

**Run:** `make eval` runs all three strategies and saves results. Does not require Ollama.

**Generation quality:** Not evaluated here. Faithfulness and answer correctness require a judge model or human evaluation. See [llm-eval-harness](https://github.com/selizondo/llm-eval-harness) for the generation evaluation pipeline.

---

## Architectural Standard

The 72% Accuracy@4 baseline is not the deliverable: the measurement discipline is. You cannot improve what you cannot measure, and this repo establishes the reference point. Vector-only retrieval, chunk=256, no BM25, no reranking, no framework. Every improvement claim in downstream projects ([rag-pipeline-app](https://github.com/selizondo/rag-pipeline-app): +11pp with hybrid search, [rag-ragas-eval](https://github.com/selizondo/rag-ragas-eval): +75% faithfulness with BM25) has this number to beat.

The chunk size experiment methodology transfers directly to any new corpus: define the metric, vary one parameter, hold everything else constant, measure. Any team starting a RAG project can run `python chunk_experiment.py` on their own data and answer the chunk size question with evidence rather than convention.
