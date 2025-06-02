"""
generate.py — Build prompt from retrieved chunks + call Ollama for an answer.

Usage:
    python generate.py --query "What is attention?" --top-k 5
"""

import argparse

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"
MAX_CONTEXT_WORDS = 1500  # stay well within Ollama context window


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Construct a grounded prompt from retrieved chunks."""
    # Trim context to avoid exceeding context window
    context_parts = []
    word_count = 0
    for chunk in chunks:
        words = chunk["text"].split()
        if word_count + len(words) > MAX_CONTEXT_WORDS:
            break
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
        word_count += len(words)

    context = "\n\n---\n\n".join(context_parts)
    return f"""You are a helpful AI assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say "I don't have enough information in the provided context."
Do not add information beyond what is in the context.

Context:
{context}

Question: {query}

Answer:"""


def generate(query: str, chunks: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Send prompt to Ollama and return the response."""
    prompt = build_prompt(query, chunks)
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


if __name__ == "__main__":
    from retrieve import retrieve

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--db-path", default="./chroma_db")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    chunks = retrieve(args.query, args.top_k, args.db_path, args.rerank)
    answer = generate(args.query, chunks, args.model)
    print(f"\nQ: {args.query}\n")
    print(f"A: {answer}")
