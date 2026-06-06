# RAG Pipeline From Scratch

![Tests](https://github.com/selizondo/rag-pipeline-from-scratch/actions/workflows/test.yml/badge.svg)

Most teams ship a RAG system inside a week. Six months later they are debugging why answers changed after a corpus update, or explaining to stakeholders why the demo no longer works. The real problem is not retrieval: it is the absence of a measurement baseline. Without a documented reference point, every change is a guess.

This project builds that baseline. A zero-dependency RAG pipeline where every design choice is measured, every parameter is documented, and every downstream improvement has a number to beat.

**Stack:** Python · ChromaDB · sentence-transformers · Ollama · rank-bm25

## Results

Chunk size is the single highest-leverage parameter in a RAG pipeline. Tested across 256, 512, and 1024 words on 20 held-out questions:

| Configuration | Chunks indexed | Accuracy@4 |
|--------------|---------------|-----------|
| chunk=256, vector only | 13 | **72%** |
| chunk=512, vector only | 8 | 64% |
| chunk=1024, vector only | 5 | 64% |

Smaller chunks produce more focused embeddings. At 1024 words, entire documents collapse into a single vector and dilute query relevance. The 72% baseline is the number every downstream project ([rag-pipeline-app](https://github.com/selizondo/rag-pipeline-app): +11pp with hybrid search) is measured against.

## How It Works

### Two-stage retrieval: speed first, precision second

Stage 1 embeds the query with `all-MiniLM-L6-v2` and retrieves candidates from ChromaDB by cosine similarity. Fast, but biased toward semantic paraphrase: it misses exact keyword matches like specific API names or acronyms.

Stage 2 is optional: a cross-encoder rescores each (query, chunk) pair jointly, catching relevance signals the bi-encoder missed. It adds roughly 3x latency. Use `--rerank` when retrieval quality matters more than speed.

### Chunk size measured, not assumed

Fixed word-based splitting at 256 words with 32-word overlap. Word-based instead of character-based because words are the natural unit of meaning: character splits break mid-word and produce incoherent embeddings. The default is not a convention: it is the result of `python chunk_experiment.py` run on this corpus.

Known gap: `all-MiniLM-L6-v2` has a 256 WordPiece token limit. One word is roughly 1.0 to 1.3 tokens, so 256-word chunks stay safely below the ceiling. At 512 words, tail content is silently truncated during embedding.

### Grounded prompt design

The prompt instructs the LLM: "Answer using ONLY the context provided. If the context is insufficient, say so." Wrong answers become "I don't know" instead of confident fabrications. The model cannot invent facts if it is forced to cite retrieved context.

**Companion post:** "Don't Guess. Measure." (AI Systems in Production series, coming soon)
**Related projects:** [rag-pipeline-app](https://github.com/selizondo/rag-pipeline-app) (improved baseline, hybrid search) · [llm-eval-harness](https://github.com/selizondo/llm-eval-harness) (eval harness used for 72% measurement)

---

## Go Deeper

| Audience | Doc |
|----------|-----|
| Business and product context | [Product and Cost](docs/product.md) |
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
