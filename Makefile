.PHONY: install ingest query chunk-experiment clean

install:
	pip install -r requirements.txt

# One-time: seed corpus from HuggingFace + ingest into Chroma
ingest:
	python seed_corpus.py --out ./corpus --skip-kaggle
	python ingest.py --corpus ./corpus --chunk-size 256 --overlap 32

# Query the pipeline
query:
	python pipeline.py --query "What is the attention mechanism in transformers?"

# Query with cross-encoder reranking (adds ~1-2s)
query-rerank:
	python pipeline.py --query "What is the attention mechanism in transformers?" --rerank --top-k 8

# Chunk size experiment (256 / 512 / 1024 comparison)
chunk-experiment:
	python chunk_experiment.py

clean:
	rm -rf chroma_db chroma_db_hf corpus __pycache__ **/__pycache__
