# Failure Scenarios

Failure modes for the pipeline. "Handled" means a try/except with a non-fatal path exists. "Documented gap" means the failure is understood but detection is not yet implemented.

---

## Failure 1: Ollama Unavailable

**What breaks:** `generate()` calls the Ollama HTTP endpoint. If Ollama isn't running, the request fails and the pipeline returns no answer.

**Status:** Handled — `generate()` wraps the HTTP call in typed try/except blocks; returns `"[ERROR: ...]"` strings rather than raising. The error message includes the specific failure type and a remediation hint.

**Error types caught:**
- `ConnectionError` → `"[ERROR: Ollama not reachable at {url}. Is Ollama running? Try: ollama serve]"`
- `Timeout` → `"[ERROR: Ollama request timed out after 120s. Try a shorter query or a smaller model.]"`
- `HTTPError` → `"[ERROR: Ollama returned HTTP {code}: {message}]"`
- Generic `Exception` → `"[ERROR: Generation failed: {e}]"`

**Observable:** Error string is returned as the `answer` field in the pipeline response. Callers can detect `answer.startswith("[ERROR:")` to distinguish generation failures from valid answers.

---

## Failure 2: Chroma Collection Missing

**What breaks:** `retrieve()` calls `client.get_collection(COLLECTION_NAME)`. If the collection doesn't exist (ingest hasn't been run), `chromadb` raises `ValueError`.

**Status:** Handled — wrapped in try/except; raises `RuntimeError` with a clear message: `"Collection 'study_notes' not found. Run ingest.py first to build the index."` Caller gets an actionable error rather than a raw chromadb traceback.

---

## Failure 3: Embedding Model Not Downloaded

**What breaks:** `retrieve()` calls `SentenceTransformer(EMBED_MODEL)`. On first run, `sentence-transformers` downloads the model from HuggingFace. If the download fails (no network, HuggingFace unavailable), the pipeline fails with an unhandled connection error.

**Status:** Documented gap — no retry or fallback.

**Detection (planned):** Wrap model load in try/except; provide a clear error message with a local cache fallback hint.

---

## Failure 4: Model Mismatch Between Ingest and Retrieve

**What breaks:** `EMBED_MODEL` was previously defined independently in both `ingest.py` and `retrieve.py`. If they diverge, the query vector is built with a different model than the index — results are silently wrong (vectors are incompatible across embedding models).

**Status:** Fixed — both files now import `EMBED_MODEL` from `config.py`. Mismatch is now a compile-time import error, not a silent runtime failure.
