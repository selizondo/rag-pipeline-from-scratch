# RAG Pipeline From Scratch

A minimal, framework-free Retrieval-Augmented Generation pipeline built to understand every layer — chunking, embedding, vector search, reranking, and generation — without abstractions hiding the mechanics.

**Stack:** Python · Chroma · sentence-transformers · Ollama · cross-encoder reranking

---

## Architecture

```
Query
  │
  ▼
[Embed query]  ←── all-MiniLM-L6-v2
  │
  ▼
[Chroma similarity search]  ←── cosine distance over stored embeddings
  │
  ▼
[Optional: cross-encoder rerank]  ←── ms-marco-MiniLM-L-6-v2
  │
  ▼
[Build grounded prompt]
  │
  ▼
[Ollama LLM]  ←── llama3.2 (local)
  │
  ▼
Answer + sources + latency
```

**Ingest (one-time):**
```
Markdown files → word-based chunking (with overlap) → embed chunks → store in Chroma
```

---

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Load `.md` files → chunk → embed → store in Chroma |
| `retrieve.py` | Embed query → Chroma search → optional cross-encoder rerank |
| `generate.py` | Build grounded prompt from chunks → call Ollama |
| `pipeline.py` | End-to-end entrypoint with latency metadata |
| `chunk_experiment.py` | Compare retrieval quality across chunk sizes |

---

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai) running locally with a model pulled.

```bash
# Pull a model
ollama pull llama3.2

# Create venv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

**1. Ingest your corpus** (point at a folder of `.md` files):

```bash
python ingest.py --corpus /path/to/your/notes --chunk-size 256 --overlap 32
```

**2. Ask questions:**

```bash
python pipeline.py --query "What is feature engineering?"
python pipeline.py --query "Explain attention mechanisms" --rerank --top-k 8
python pipeline.py --query "What is RAG?" --model mistral
```

**3. One-shot ingest + query:**

```bash
python pipeline.py --ingest /path/to/notes --query "What is overfitting?"
```

**Example output:**
```
Q: What is feature engineering and why does it matter?

A: Feature engineering is the process of transforming raw data into meaningful
representations that improve a model's ability to learn patterns and generalize.
Good features often outperform complex models.

Sources: feature_engineering_interview_cheat_sheet.md, feature_engineering_techniques.md
Latency: retrieve=2905ms  generate=102502ms  total=105407ms
```

---

## Design Decisions

**Word-based chunking over character-based**
Words are the natural unit of meaning. Character splits can break mid-word and produce semantically incoherent chunks. Word-based splitting also makes `chunk_size` human-interpretable.

**`all-MiniLM-L6-v2` for embeddings**
Fast, small (22M params), strong on semantic similarity benchmarks. No GPU required. Right tradeoff for a local pipeline where latency matters.

**Two-stage retrieval: ANN + cross-encoder reranking**
Approximate nearest-neighbor search (Chroma/HNSW) is fast but uses bi-encoder embeddings that compress both query and document independently — losing interaction signals. The cross-encoder reranker sees the full (query, chunk) pair and scores relevance more precisely. Cost: ~3x more chunks retrieved in stage 1, then reranked and trimmed.

**`--rerank` is opt-in**
Reranking adds ~1-2s of CPU inference per query. For most use cases, the bi-encoder retrieval is good enough. The flag lets you pay only when you need it.

**Ollama for generation**
Keeps the pipeline fully local — no API keys, no rate limits, no cost. Swap to any model with `--model`.

---

## Chunking Experiment

Tested 256 / 512 / 1024 word chunk sizes on 5 source documents, 3 queries:

| Chunk size | Chunks | Avg top retrieval score |
|-----------|--------|------------------------|
| **256** | 13 | **0.1881** |
| 512 | 8 | 0.1009 |
| 1024 | 5 | 0.1009 |

**Finding:** Smaller chunks (256 words) produce more topically focused embeddings, leading to higher retrieval precision. At 1024 words, entire documents collapse into single vectors that dilute query relevance. **Default recommendation: 256 words with ~12% overlap.**

> Note: absolute cosine similarity scores are low here because the corpus is small (5 files). On a larger corpus, scores will be higher and the gap between chunk sizes more pronounced.

---

## Limitations & What's Next

- `ingest.py` reads a single flat directory — no recursive crawl
- No chunking on semantic boundaries (sentences, paragraphs) — purely word-count based
- No streaming responses from Ollama
- No persistence of query history or caching

Project 05 in this series ([Polished RAG App](../roadmap/projects/05_rag_app_portfolio.md)) builds on this foundation with FastAPI, streaming, hybrid search, and observability.
