# Product and Cost

This document frames the project for a technical business reviewer: what organizational problem it addresses, how it earns trust, what it costs to operate, and when a team should build something like this versus buying a managed solution.

---

## The Business Problem

AI teams routinely ship RAG systems tuned by intuition and tested by demo. The demo works. The product degrades. A corpus update changes retrieval behavior silently. A model swap shifts answer quality with no measurement to detect it. The team has no baseline, so every fix is a guess.

This is an organizational risk problem, not a retrieval problem. Teams that cannot measure their RAG system cannot defend its behavior to stakeholders, cannot detect regressions before users do, and cannot justify engineering investment in improvements. The baseline is the precondition for everything else.

---

## Trust Surface

**What can go wrong:**

- Retrieval returns irrelevant chunks, and the LLM generates a plausible-sounding wrong answer with no signal that retrieval failed
- A corpus update mixes old and new content in the same index, producing non-deterministic answers across queries
- The embedding model and the ingest pipeline use different chunk size measurements (words vs tokens), causing silent truncation on longer documents

**How this system addresses each:**

- Grounded prompt design forces the model to say "I don't know" when context is insufficient, rather than hallucinating. Wrong answers are distinguishable from confident fabrications.
- Version-scoped metadata (`corpus_version` field on every chunk) isolates corpus versions in the index. Re-ingesting a new corpus does not corrupt existing queries.
- The tokenizer mismatch between word-count chunking and WordPiece tokenization is documented explicitly with the failure boundary: 256 words stays safely below the 256-token model limit; 512 words silently truncates tail content.

**What is not addressed here:** There is no query audit log, no per-query retrieval trace, and no production alerting. These are the observability gaps [rag-pipeline-app](https://github.com/selizondo/rag-pipeline-app) closes.

---

## Cost Model

This pipeline is designed to run with zero recurring cost. All components run locally on a laptop CPU.

| Scale | API cost | Infra cost | Notes |
|-------|----------|-----------|-------|
| Prototype (< 5K chunks) | $0 | $0 | Ollama local, Chroma file-backed, no GPU |
| Small team (10K queries/day) | $0 (local) or ~$10/day (OpenAI API swap) | $0 or ~$50/month VPS | Ollama bottlenecks at >1 QPS; hosted API removes the bottleneck |
| Production (100K queries/day) | ~$100/day (OpenAI API) | ~$200/month (managed vector DB) | ChromaDB brute-force search breaks above ~500K chunks; requires HNSW or a managed service |

**Inflection points:**
- Above ~1 QPS: Ollama's single-threaded model becomes the bottleneck. Swap to a hosted LLM API.
- Above ~100K chunks: Chroma's flat index latency degrades. Enable HNSW or migrate to Qdrant/Weaviate.
- Above ~50K chunks: BM25 in-memory index saturates RAM. Migrate to Pyserini or Elasticsearch.

---

## Market Context

Most teams adopting RAG in 2024 to 2026 skip the baseline. They adopt a framework (LangChain, LlamaIndex), wire it to an LLM, and ship. The framework hides the mechanics, so when retrieval degrades, the team has no handle on which parameter to adjust.

The pattern that breaks at scale is: no measurement before optimization. Teams spend weeks tuning prompts and switching embedding models without a controlled reference point. The improvement they see (or don't see) is uninterpretable.

This project is the reference point. It exists because the AI infrastructure market has strong tooling for building RAG but weak tooling for measuring it. The eval harness, the baseline number, and the chunk size experiment are the measurement layer that the frameworks do not provide.

---

## Deployment Constraints

**Latency:** At the default corpus size (~3,000 chunks), retrieval is under 100ms. Total pipeline latency (retrieve + generate) is 3 to 100 seconds depending on the model and whether reranking is enabled. Generation latency is dominated by Ollama local inference on CPU: ~80 to 100 seconds for llama3.2. Swap to a hosted API to bring generation to 2 to 5 seconds.

**Concurrency:** Ollama is single-threaded. One concurrent user. Not suitable for multi-user deployment without an inference server upgrade.

**Corpus updates:** Re-ingest with a new `corpus_version` value. Old chunks remain in the collection and can be queried by rolling back the version in `config.yaml`. For clean separation, drop the collection and re-ingest.

**Observability:** None at this layer. There is no record of which chunks were retrieved for a given query, no latency percentile tracking beyond the single-request log line, and no alert on retrieval degradation. The retrieval SLA warning (500ms threshold in `config.yaml`) is the only production signal.

**On-call implications:** If Ollama is unavailable, the pipeline returns a typed error string rather than raising. If Chroma is missing, the pipeline raises `RuntimeError` with an actionable message. Neither is silent.

---

## Build vs Buy

**Build (this approach) when:**

- The team needs a controlled baseline before committing to a framework
- The corpus is small enough that a local vector store is sufficient (under ~100K chunks)
- Cost is the primary constraint and local inference is acceptable
- The team wants to understand what the framework is doing before adopting one

**Buy or adopt a framework when:**

- The team is past the baseline phase and needs multi-tenancy, streaming, or managed observability
- The corpus exceeds 500K chunks (managed vector DBs handle scaling automatically)
- Latency requirements are under 1 second (requires a hosted LLM API, not local Ollama)
- Multiple teams need to share the same retrieval infrastructure

**The hand-off point:** This pipeline is the right starting point for any RAG project. The framework adoption decision belongs after the baseline is established and the team has a measurement to compare against. Frameworks bought before a baseline exists cannot be evaluated; they can only be trusted.
