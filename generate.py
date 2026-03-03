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
    generate() returns (answer_str, fallback_events) in all cases. On Ollama
    errors, answer is a best-effort "Sources: [list]" string and fallback_events
    records what went wrong (e.g., "ollama_timeout_30s"). WHY return instead of
    raise: callers (like pipeline.py, api.py) can include the fallback in their
    response without catching exceptions, and the user sees a useful message.

Async usage:
    generate_async() wraps the synchronous Ollama call in asyncio.to_thread() so
    FastAPI endpoints don't block the event loop. WHY: Ollama is single-threaded;
    blocking the event loop starves all concurrent requests while one generation
    runs. asyncio.to_thread() offloads the blocking HTTP call to a thread pool.

Usage:
    python generate.py --query "What is attention?" --top-k 5
"""

import asyncio
import argparse
import logging

import requests

from config import CORPUS_VERSION, DEFAULT_DB_PATH, DEFAULT_OLLAMA_MODEL, MAX_CONTEXT_WORDS, OLLAMA_TIMEOUT_SECONDS, OLLAMA_URL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Assemble a grounded RAG prompt from retrieved chunks.

    WHY iterate chunks and trim by word count instead of passing all chunks:
        Different corpora and top_k values can produce very long context sections.
        Trimming at MAX_CONTEXT_WORDS (from config.yaml → generation.context_budget_words)
        prevents the prompt from exceeding the model's context window, which would
        cause Ollama to silently truncate the input or return an error.
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


def _call_ollama(prompt: str, model: str) -> tuple[str, list[str]]:
    """
    Synchronous Ollama HTTP call with timeout and structured error handling.

    Returns (answer, fallback_events). On failure, answer is a best-effort
    "Generation unavailable. Sources: [list]" message so the caller still has
    something useful to return to the user.

    WHY return fallback_events instead of raising:
        Callers (pipeline.py, api.py) include fallback_events in the response
        so downstream observers (eval harness, monitoring) can audit what
        degradation occurred without parsing log files.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["response"].strip(), []
    except requests.exceptions.ConnectionError:
        msg = (
            f"Generation unavailable — Ollama not reachable at {OLLAMA_URL}. "
            f"Try: ollama serve"
        )
        return msg, ["ollama_connection_error"]
    except requests.exceptions.Timeout:
        msg = (
            f"Generation unavailable — Ollama timed out after {OLLAMA_TIMEOUT_SECONDS}s. "
            f"Try a shorter query or a smaller model."
        )
        return msg, [f"ollama_timeout_{OLLAMA_TIMEOUT_SECONDS}s"]
    except requests.exceptions.HTTPError as e:
        msg = f"Generation unavailable — Ollama returned HTTP {e.response.status_code}."
        return msg, [f"ollama_http_{e.response.status_code}"]
    except Exception as e:
        msg = f"Generation unavailable — unexpected error: {e}"
        return msg, ["ollama_error_unexpected"]


def generate(query: str, chunks: list[dict], model: str = DEFAULT_OLLAMA_MODEL) -> tuple[str, list[str]]:
    """
    Send the assembled prompt to a local Ollama model and return the answer.

    Returns (answer, fallback_events). On Ollama failure, answer is a
    best-effort string listing the sources so the caller still has something
    useful to return.

    WHY stream=False instead of streaming:
        This is a research pipeline, not a production API. Blocking until we
        have the full response is simpler. For a production UI, switch to
        stream=True and yield tokens as they arrive (see rag-pipeline-app for
        the streaming implementation with SSE).
    """
    if not chunks:
        sources = []
    else:
        sources = list({c["source"] for c in chunks})

    prompt = build_prompt(query, chunks)
    answer, fallback_events = _call_ollama(prompt, model)

    # If generation failed, append source list so the user has grounding context.
    if fallback_events:
        if sources:
            answer += f" Sources: {', '.join(sources)}"
        logger.warning("Generation fallback: %s", fallback_events)

    return answer, fallback_events


async def generate_async(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_OLLAMA_MODEL,
) -> tuple[str, list[str]]:
    """
    Async wrapper for generate() using asyncio.to_thread().

    WHY to_thread instead of native async:
        Ollama's HTTP endpoint is a blocking call — there is no native async
        client for it. asyncio.to_thread() offloads the blocking I/O to a
        thread pool so the FastAPI event loop remains free to handle other
        requests while generation runs. Without this, one slow Ollama call
        blocks all concurrent requests in the same process.
    """
    return await asyncio.to_thread(generate, query, chunks, model)


if __name__ == "__main__":
    from retrieve import retrieve

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    chunks, _ = retrieve(args.query, args.top_k, args.db_path, args.rerank)
    answer, fallback_events = generate(args.query, chunks, args.model)
    print(f"\nQ: {args.query}\n")
    print(f"A: {answer}")
    if fallback_events:
        print(f"Fallback events: {fallback_events}")
