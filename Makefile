.PHONY: bootstrap test ingest query query-rerank chunk-experiment eval serve clean

bootstrap:
	UV_PROJECT_ENVIRONMENT=.venv uv sync

test:
	uv run pytest

# One-time: seed corpus from HuggingFace + ingest into Chroma
ingest:
	uv run python seed_corpus.py --out ./corpus --skip-kaggle
	uv run python ingest.py --corpus ./corpus --chunk-size 256 --overlap 32

# Query the pipeline (CLI)
query:
	uv run python pipeline.py --query "What is the attention mechanism in transformers?"

# Query with cross-encoder reranking (adds ~1-2s)
query-rerank:
	uv run python pipeline.py --query "What is the attention mechanism in transformers?" --rerank --top-k 8

# Chunk size experiment (256 / 512 / 1024 comparison)
chunk-experiment:
	uv run python chunk_experiment.py

# Evaluate retrieval quality against the labeled test set.
# Run after any corpus update to verify retrieval quality hasn't regressed.
# Compares dense, dense+rerank, and BM25 baseline strategies.
# Results saved to artifacts/eval/latest_run.json.
# NOTE: Requires ingest to have been run first. Does NOT require Ollama.
eval:
	uv run python evaluation/eval.py --all-strategies --top-k 5
	@echo "Results in artifacts/eval/latest_run.json"

# Start the FastAPI server (requires uvicorn)
serve:
	uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf chroma_db chroma_db_hf corpus __pycache__ **/__pycache__
