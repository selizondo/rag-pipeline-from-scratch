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
| `seed_corpus.py` | Download ML Q&A datasets from HuggingFace + Kaggle → write `.md` files |
| `ingest.py` | Load `.md` files → chunk → embed → store in Chroma |
| `retrieve.py` | Embed query → Chroma search → optional cross-encoder rerank |
| `generate.py` | Build grounded prompt from chunks → call Ollama |
| `pipeline.py` | End-to-end entrypoint with latency metadata |
| `chunk_experiment.py` | Compare retrieval quality across chunk sizes |

---

## Limitations
This repo is intentionally minimal and is not meant to be production-ready. It lacks an API boundary, query audit logging, and explicit fallback behavior, so it is best used as a research-quality baseline rather than a deployable service.

---

## Quick Start

**Runs fully locally — no GPU, no API key required.**

```bash
# 1. One-time: copy and fill the workspace master env
cp ../career/.env.example ../career/.env   # add keys if needed (not required for this project)

# 2. Activate shared venv
source ~/.venvs/newline/bin/activate
# or create a project venv:
#   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 3. Pull an Ollama model (if not already pulled)
ollama pull llama3.2   # or: qwen2.5-coder:7b, mistral

# 4. Seed + ingest corpus (one-time, ~2 min)
python seed_corpus.py --out ./corpus        # HuggingFace sources only
# python seed_corpus.py --out ./corpus --skip-kaggle  # skip if no Kaggle credentials

python ingest.py --corpus ./corpus --chunk-size 256 --overlap 32

# 5. Query
python pipeline.py --query "What is the attention mechanism in transformers?"
python pipeline.py --query "How do you prevent overfitting?" --rerank --top-k 8
```

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) running locally.
**For Kaggle corpus:** `~/.kaggle/kaggle.json` credentials ([get token](https://www.kaggle.com/settings/account)).

---

## Corpus

The study corpus is seeded from three ML Q&A datasets via `seed_corpus.py`:

| Source | Type | Examples | Content |
|--------|------|----------|---------|
| `Shlok307/Interview_questions` (HuggingFace) | AI-domain Q&A | 1,819 | Deep learning, NLP, transformers, interview-style |
| `mjphayes/machine_learning_questions` (HuggingFace) | ML interview Q&A | 390 | Algorithms, cross-validation, fundamentals |
| `lorenzoscaturchio/ml-interview-qa` (Kaggle) | Q&A with difficulty + company tags | 502 | System design, CV, ML, tagged by Amazon/OpenAI/Meta |

```bash
# Download and write corpus .md files (requires Kaggle credentials for full corpus)
python seed_corpus.py --out ./corpus

# HuggingFace only (no Kaggle credentials needed)
python seed_corpus.py --out ./corpus --skip-kaggle
```

---

## Usage

**1. Seed and ingest the corpus:**

```bash
python seed_corpus.py --out ./corpus
python ingest.py --corpus ./corpus --chunk-size 256 --overlap 32
```

**2. Ask questions:**

```bash
python pipeline.py --query "What is the attention mechanism in transformers?"
python pipeline.py --query "How do you handle overfitting?" --rerank --top-k 8
python pipeline.py --query "When should you fine-tune vs use RAG?" --model mistral
```

**3. One-shot ingest + query:**

```bash
python pipeline.py --ingest ./corpus --query "What is gradient descent?"
```

**Example output:**
```
Q: What is the attention mechanism in transformers?

A: The attention mechanism allows transformers to weigh the relevance of different
tokens when encoding a sequence. Each token attends to all others via learned
query, key, and value projections — enabling long-range dependencies that RNNs
struggle to capture.

Sources: ml_interview_qa_kaggle.md, ai_interview_qa_shlok.md
Latency: retrieve=2936ms  generate=96082ms  total=99018ms
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

Tested 256 / 512 / 1024 word chunk sizes across 3 queries:

| Chunk size | Chunks | Avg top retrieval score |
|-----------|--------|------------------------|
| **256** | 13 | **0.1881** |
| 512 | 8 | 0.1009 |
| 1024 | 5 | 0.1009 |

**Finding:** Smaller chunks (256 words) produce more topically focused embeddings, leading to higher retrieval precision. At 1024 words, entire documents collapse into single vectors that dilute query relevance. **Default: 256 words with ~12% overlap.**

**Answer quality:** Evaluated on 20 held-out questions using the [llm-eval-harness](../llm-eval-harness) judge — 72% Accuracy@4 with chunk=256, vector search only. This is the baseline that `rag-pipeline-app` improves on: adding hybrid BM25+vector search raises it to 83% (see [rag-pipeline-app Decision 1](../rag-pipeline-app/README.md#decision-1-hybrid-bm25--vector-search)).

Run the experiment yourself against the seeded corpus:

```bash
python chunk_experiment.py
```

---

## Where This Breaks

**At 100k documents:** The embedding step in `ingest.py` is synchronous — all chunks embedded in a single loop. At ~20k chunks (256 words each from a 100k-doc corpus), ingest takes 45–60 minutes. Fix: batch with `asyncio` or use a dedicated embedding service.

**Chroma file-backed storage:** The default Chroma setup writes to a local directory. It has no concurrency guarantees — two simultaneous writes corrupt the index. For multi-user or multi-process access, switch to Chroma's HTTP server mode.

**No observability:** There's no record of which chunks were retrieved for any given query or how long retrieval took. You can't tell whether a wrong answer is a retrieval failure or a generation failure. `rag-pipeline-app` adds SQLite logging for this — see [its observability section](../rag-pipeline-app/README.md#decision-3-observability-from-day-one).

---

## Limitations & What's Next

- `ingest.py` reads a single flat directory — no recursive crawl
- No chunking on semantic boundaries (sentences, paragraphs) — purely word-count based
- No streaming responses from Ollama
- No persistence of query history or caching
- `seed_corpus.py` pulls full Q&A text but doesn't filter by topic cluster — a larger seeded corpus would benefit from category-based file splitting

---

## Architectural Standard

The 72% Accuracy@4 baseline is not the deliverable — the measurement discipline is. You can't improve what you can't measure, and this repo establishes the reference point: vector-only retrieval, chunk=256, no BM25, no reranking, no framework. Every improvement claim in downstream projects ([rag-pipeline-app](../rag-pipeline-app): +4pp with hybrid search, [rag-ragas-eval](../rag-ragas-eval): +75% faithfulness with BM25) has this number to beat.

The chunk size experiment methodology transfers directly to any new corpus: define the metric, vary one parameter, hold everything else constant, measure. That's what makes retrieval improvements defensible. Any team starting a RAG project can run `python chunk_experiment.py` on their own data and answer the chunk size question with evidence rather than convention.

