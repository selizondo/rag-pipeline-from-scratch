# Setup and Usage

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
