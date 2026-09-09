"""Central configuration for the Security Requirements Assistant.

Every constant that used to be scattered across `rag.py`, `ingest.py`, and
`database.py` (model name, storage paths, LLM generation settings, retrieval/
chunking sizes) lives here instead, so a new developer has one place to look
when tuning behavior instead of grepping through multiple modules.

Storage paths are kept as plain relative strings (not `os.path.join(os.getcwd(), ...)`)
on purpose: this matches the app's original behavior of resolving everything
relative to the working directory the server was started from, and changing
that would be an unrelated behavior change.
"""
import os

# sentence-transformers checks the HuggingFace Hub for the embedding model's metadata on
# every startup, even though it's already cached locally after the first run - the HF server
# then replies with a rate-limit reminder for unauthenticated requests, which huggingface_hub
# logs as a warning. This only turns down that library's own log verbosity (errors still show);
# it doesn't disable the metadata check or the first-run download itself.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# --- Storage locations (relative to the working directory the app is started from) ---
DB_NAME = "chat_history.db"
CHROMA_DIR = "chroma_db"
KNOWLEDGE_BASE_DIR = "knowledge_base"
UPLOAD_DIR = "uploaded_docs"

# --- LLM model & hardware ---
MODEL_NAME = "Llama-3.2-3B-Instruct-Q4_0.gguf"
N_CTX = 4096

# Leave one core free for the OS/UI so the machine doesn't feel frozen during generation -
# important on weaker proband hardware where an "auto" (all-cores) thread count can stall everything else.
N_THREADS = max(1, (os.cpu_count() or 4) - 1)

# Larger prompt-processing batch size speeds up prefill (system prompt + retrieved context +
# conversation history), at the cost of a bit more RAM. GPT4All's own default here is only 8.
N_BATCH = 256

# --- Generation temperature ---
# Audit is a deterministic scoring tool and stays low-temperature for consistent, repeatable
# scores. Elicitation gets a much higher temperature so replies don't all collapse onto the
# same phrasing turn after turn - the 3B model at temp=0.1 was observed to reuse near-identical
# sentence openers regardless of topic, which reads as "the bot always says the same thing".
AUDIT_TEMP = 0.1
ELICITATION_TEMP = 0.65

# --- Prompt-size safety net ---
# Rough char-based budget to keep the assembled prompt inside n_ctx tokens (reserving room
# for the completion). ~1 token ≈ 4 chars for English text.
PROMPT_CHAR_BUDGET = 12000
MAX_USER_QUERY_CHARS = 4000

# --- Retrieval & document ingestion ---
RETRIEVAL_N_RESULTS = 2
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PDF_CHUNK_SIZE = 800
PDF_CHUNK_OVERLAP = 100
