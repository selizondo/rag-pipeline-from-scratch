# Setup and Usage

## Key Concepts

**Chunking:** Splitting documents into overlapping segments (256 words default, 32-word overlap). Overlap ensures concepts split across chunk boundaries don't lose context. The tradeoff: smaller chunks = better retrieval precision but more chunks to embed; larger chunks = fewer embeddings but diluted signal. This project measures the tradeoff (chunk size 128/256/512) before choosing the default.

**Embedding:** Converting text to dense vectors (384-dim with `all-MiniLM-L6-v2`). Embeddings capture semantic meaning — "transformer architecture" and "attention mechanism" are close in embedding space. Stored in Chroma, indexed for fast similarity search. Key insight: embeddings compress meaning but lose exact keyword matching — solved by reranking.

**Vector similarity search:** Finding chunks closest to the query in embedding space (Chroma uses HNSW indexing + cosine distance). Fast but biased toward semantic similarity. Fails on exact keyword queries: "What is LoRA?" — the embedding doesn't match "low-rank adaptation" well enough to surface the definition. Fallback: cross-encoder reranking.

**Cross-encoder reranking:** A 2-stage retrieval. Stage 1: retrieve top-100 chunks via vector similarity (fast, broad). Stage 2: re-score the top-100 using a cross-encoder model that sees the full (query, chunk) pair (slower, precise). Reranking adds ~3x latency but surfaces the right chunks for keyword queries. Optional — use `--rerank` if quality matters more than speed.

**Grounding and hallucination prevention:** Building a prompt with retrieved chunks as context, then asking the LLM to answer using only that context. Reduces hallucinations — the model can't confidently invent facts if it's forced to cite retrieved context. This project explicitly instructs the LLM: "Answer using ONLY the context provided. If the context is insufficient, say so."

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- Kaggle credentials at `~/.kaggle/kaggle.json` (optional, for full corpus)

## Quick Start

Runs fully locally. No GPU, no API key required.

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pull an Ollama model
ollama pull llama3.2

# 3. Seed and ingest the corpus (one-time, ~2 min)
python seed_corpus.py --out ./corpus
python ingest.py --corpus ./corpus --chunk-size 256 --overlap 32

# 4. Query
python pipeline.py --query "What is the attention mechanism in transformers?"
python pipeline.py --query "How do you prevent overfitting?" --rerank --top-k 8
```

To skip Kaggle (no credentials needed):
```bash
python seed_corpus.py --out ./corpus --skip-kaggle
```

## Commands

```bash
# One-shot ingest + query
python pipeline.py --ingest ./corpus --query "What is gradient descent?"

# Run the chunk size experiment
python chunk_experiment.py

# Run retrieval evaluation (Precision@K, Recall@K across strategies)
make eval
```

## Corpus

The study corpus is seeded from three ML Q&A datasets via `seed_corpus.py`:

| Source | Type | Size | Content |
|--------|------|------|---------|
| `Shlok307/Interview_questions` (HuggingFace) | AI-domain Q&A | 1,819 | Deep learning, NLP, transformers |
| `mjphayes/machine_learning_questions` (HuggingFace) | ML interview Q&A | 390 | Algorithms, cross-validation, fundamentals |
| `lorenzoscaturchio/ml-interview-qa` (Kaggle) | Q&A with difficulty tags | 502 | System design, CV, ML, tagged by company |

## Example Output

```
Q: What is the attention mechanism in transformers?

A: The attention mechanism allows transformers to weigh the relevance of different
tokens when encoding a sequence. Each token attends to all others via learned
query, key, and value projections — enabling long-range dependencies that RNNs
struggle to capture.

Sources: ml_interview_qa_kaggle.md, ai_interview_qa_shlok.md
Latency: retrieve=2936ms  generate=96082ms  total=99018ms
```

## Code Layout

| File | Purpose |
|------|---------|
| `seed_corpus.py` | Download ML Q&A datasets from HuggingFace and Kaggle, write to `.md` files |
| `ingest.py` | Load `.md` files, chunk, embed, store in Chroma |
| `retrieve.py` | Embed query, Chroma search, optional cross-encoder rerank |
| `generate.py` | Build grounded prompt from chunks, call Ollama |
| `pipeline.py` | End-to-end entrypoint with latency metadata |
| `chunk_experiment.py` | Compare retrieval quality across chunk sizes |
| `evaluation/eval.py` | Precision@K and Recall@K across dense, rerank, and BM25 strategies |
| `data/test_set.json` | 10 held-out Q&A pairs with labeled relevant sources |
