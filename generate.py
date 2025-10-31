"""
generate.py — Build a grounded prompt from retrieved chunks and call Ollama.

Architecture overview:
    This is the final stage of the RAG pipeline. It takes the chunks from
    retrieve.py, builds a prompt that constrains the model to answer from context
    only, and calls Ollama's local inference endpoint.

    WHY context-grounded prompting ("answer ONLY from context"):
        Without this constraint, LLMs will supplement retrieved context with
        parametric knowledge — producing answers that sound confident but aren't
        grounded in the retrieved documents. This is the "hallucination" failure
        mode. The constraint forces the model to admit uncertainty ("I don't have
        enough information") rather than fabricate plausible-sounding answers.

Error handling:
    generate() returns a plain string in all cases. On Ollama errors, it returns
    a "[ERROR: ...]" string rather than raising. WHY: callers (like pipeline.py)
    can include the error string in their response without catching exceptions,
    and the user sees a useful message instead of a stack trace.

Usage:
    python generate.py --query "What is attention?" --top-k 5
"""

import argparse

import requests

from config import DEFAULT_DB_PATH, DEFAULT_OLLAMA_MODEL

OLLAMA_URL = "http://localhost:11434/api/generate"

# Context window budget for llama3.2.
# llama3.2 has an 8192-token context window (Ollama default).
# Prompt overhead (system instructions + question + formatting): ~200 tokens.
# English text averages ~1.3 tokens per word.
# Max safe word budget = (8192 - 200) / 1.3 ≈ 6147 words.
#
# WHY 1500 words instead of the theoretical max of ~6147:
#   More context does not always mean better answers. Irrelevant context degrades
#   generation quality — the model attends to all context tokens, so padding the
#   prompt with weakly-related chunks can crowd out the relevant ones. 1500 words
#   (≈5 chunks × 300 words each) is a practical starting point. The calculation
#   above confirms 1500 is well within the safe window, not an arbitrary cap.
MAX_CONTEXT_WORDS = 1500


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Assemble a grounded RAG prompt from retrieved chunks.

    WHY iterate chunks and trim by word count instead of passing all chunks:
        Different corpora and top_k values can produce very long context sections.
        Trimming at MAX_CONTEXT_WORDS prevents the prompt from exceeding the
        model's context window, which would cause Ollama to silently truncate the
        input or return an error.
    """
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


def generate(query: str, chunks: list[dict], model: str = DEFAULT_OLLAMA_MODEL) -> str:
    """
    Send the assembled prompt to a local Ollama model and return the answer.

    WHY stream=False instead of streaming:
        This is a research pipeline, not a production API. Blocking until we
        have the full response is simpler. For a production UI, switch to
        stream=True and yield tokens as they arrive (see rag-pipeline-app for
        the streaming implementation with SSE).

    On Ollama errors, returns an "[ERROR: ...]" string rather than raising.
    WHY: callers can include the error in their response without extra exception
    handling. The user sees a clear message instead of a stack trace.
    """
    prompt = build_prompt(query, chunks)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return f"[ERROR: Ollama not reachable at {OLLAMA_URL}. Is Ollama running? Try: ollama serve]"
    except requests.exceptions.Timeout:
        return "[ERROR: Ollama request timed out after 120s. Try a shorter query or a smaller model.]"
    except requests.exceptions.HTTPError as e:
        return f"[ERROR: Ollama returned HTTP {e.response.status_code}: {e.response.text[:200]}]"
    except Exception as e:
        return f"[ERROR: Generation failed: {e}]"


if __name__ == "__main__":
    from retrieve import retrieve

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    chunks = retrieve(args.query, args.top_k, args.db_path, args.rerank)
    answer = generate(args.query, chunks, args.model)
    print(f"\nQ: {args.query}\n")
    print(f"A: {answer}")
