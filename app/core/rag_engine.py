"""The RAG (Retrieval-Augmented Generation) engine: loads the local GPT4All model
once at startup and turns (user query + retrieved context + approved artifacts +
conversation history) into a model response.

Two response-generating entry points are used by the API layer:
- `generate_response`         - a normal chat turn (Elicitation guidance or Audit scoring)
- `generate_finalize_summary`  - the explicit "Finalize {Step}" action, which asks the
                                   model to summarize a finished step into one clean artifact
                                   instead of relying on the model to naturally arrive at a
                                   clean closing message (which was observed to instead
                                   produce content-free small talk and drift to the next step)

`scan_full_document` is a separate, cheaper extraction pass used when a PDF is
uploaded, to auto-suggest artifacts from its text.
"""
import sys
import time
import json
import ast
import re
import random

from gpt4all import GPT4All

from app.config import (
    MODEL_NAME, N_CTX, N_THREADS, N_BATCH,
    AUDIT_TEMP, ELICITATION_TEMP,
    PROMPT_CHAR_BUDGET, MAX_USER_QUERY_CHARS,
)
from app.core.prompts import AUDIT_PROMPT, ELICITATION_PROMPT, FINALIZE_PROMPT, STEP_GUIDANCE, STYLE_HINTS
from app.core.database import db
from app.core.ingestion import ingestor

# gpt4all==2.8.2's GPT4All.generate() has no `stop` parameter; the only way to
# halt generation early on a special-token sequence is via the per-token callback
# used in `_make_stop_callback`.
STOP_SEQUENCES = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>"]


class RagEngine:
    """Owns the loaded GPT4All model and all prompt-building/generation logic."""

    def __init__(self):
        self.n_threads = N_THREADS
        self.n_batch = N_BATCH
        self.llm = self._load_model()

    def _load_model(self):
        """Loads the GGUF model, preferring GPU acceleration when available and
        falling back to CPU on any failure (e.g. insufficient VRAM)."""
        print("--- LOADING AI MODEL (GPT4All) ---")
        gpu_device = self._pick_gpu_device()
        llm = None
        if gpu_device:
            try:
                llm = GPT4All(MODEL_NAME, device=gpu_device, n_threads=self.n_threads, n_ctx=N_CTX)
                print(f"--- MODEL LOADED on GPU (device={llm.device or gpu_device}) ---")
            except Exception as e:
                # e.g. insufficient VRAM for this model - GPT4All docs warn the instance is unusable after this
                print(f"--- GPU init failed ({e}) - falling back to CPU ---")
                llm = None

        if llm is None:
            llm = GPT4All(MODEL_NAME, device="cpu", n_threads=self.n_threads, n_ctx=N_CTX)
            print("--- MODEL LOADED on CPU ---")
        return llm

    def _pick_gpu_device(self):
        """Returns a GPT4All `device` string to attempt GPU acceleration with, or
        None if no GPU-capable backend is available/detectable.

        On non-macOS, `list_gpus()` can report the same physical GPU under multiple
        backends (e.g. both "kompute:..." and "cuda:..." for an NVIDIA card once the
        CUDA runtime is available - see nvidia-cuda-runtime-cu11/nvidia-cublas-cu11,
        which GPT4All auto-loads if installed). Native CUDA is meaningfully faster
        than the Kompute/Vulkan fallback for NVIDIA cards, so it's preferred whenever
        listed; Kompute remains the broad-compatibility fallback for AMD/Intel GPUs
        or NVIDIA cards without the matching CUDA runtime installed."""
        try:
            gpus = GPT4All.list_gpus()
        except Exception as e:
            print(f"--- GPU detection failed ({e}) - will use CPU ---")
            return None
        if not gpus:
            print("--- No GPU detected - will use CPU ---")
            return None
        print(f"--- GPU(s) detected: {gpus} ---")

        if sys.platform == "darwin":
            return "gpu"  # Metal
        if any(g.startswith("cuda:") for g in gpus):
            return "cuda"
        return "kompute"

    def _make_stop_callback(self):
        """Builds a fresh per-token callback that halts generation as soon as a
        Llama-3 special token sequence starts appearing in the tail of the
        streamed output (see STOP_SEQUENCES)."""
        buffer = {"text": ""}

        def callback(token_id, response):
            buffer["text"] = (buffer["text"] + response)[-64:]
            if any(marker in buffer["text"] for marker in STOP_SEQUENCES):
                return False
            return True

        return callback

    def _clean_response(self, text: str) -> str:
        """Strips any special-token remnants that slipped past the stop callback,
        and replaces a raw backend overflow error with a friendly message instead
        of showing internal error text to the user."""
        for marker in STOP_SEQUENCES:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        text = text.strip()
        if text.startswith("ERROR:") or "context window" in text.lower():
            return ("⚠️ Your message plus the current conversation history was too large for the model "
                     "to process. Please shorten your message, or start a new chat to reset the context.")
        return text

    def _format_history(self, history, max_turns=8, max_chars_per_msg=500):
        """Formats prior conversation turns as Llama-3 chat-template blocks."""
        if not history:
            return ""
        blocks = []
        for msg in history[-max_turns:]:
            role = "assistant" if msg.get("role") == "bot" else "user"
            content = (msg.get("content") or "")[:max_chars_per_msg]
            blocks.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>\n")
        return "".join(blocks)

    def _build_full_prompt(self, active_prompt, state_instruction, context_text, history_text, user_query):
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{active_prompt}\n\n"
            f"{state_instruction}\n\n"
            f"CONTEXT (PDFs):\n{context_text}<|eot_id|>\n"
            f"{history_text}"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate_response(self, user_query: str, session_id: str, intent: str = "elicitation", process_state: str = "Asset", history=None):
        """Generates one chat-turn response. `intent` selects the persona
        (elicitation guidance vs. audit scoring, see app.core.prompts); `process_state`
        selects which of the 4 SRE steps (Asset/Threat/Goal/Req) is active;
        `history` is the session's prior turns, already fetched by the caller
        before the current turn was persisted (so it isn't double-counted)."""
        start_time = time.time()

        # Cap the raw user query up front. Besides bounding prompt size, this keeps the
        # embedding/retrieval step (below) from tokenizing pathologically long input.
        if len(user_query) > MAX_USER_QUERY_CHARS:
            user_query = user_query[:MAX_USER_QUERY_CHARS]

        # 1. Retrieval - session's own uploaded docs plus the shared supervisor knowledge base
        context_text, sources = ingestor.query_context(user_query, session_id)

        # 2. Shared Context - only artifacts the user has explicitly approved
        artifacts = db.get_session_artifacts(session_id, status="approved")
        artifacts_text = "No approved artifacts yet."
        if artifacts:
            artifacts_text = "\n".join([f"- [{art['type']}] {art['content']}" for art in artifacts])

        # 3. Intent Routing & State Injection
        active_prompt = AUDIT_PROMPT if intent == "audit" else ELICITATION_PROMPT

        step_guidance = STEP_GUIDANCE.get(process_state, STEP_GUIDANCE["Req"])
        state_instruction = (
            f"CURRENT PROCESS PHASE: You are currently working on defining '{process_state}s'. "
            f"Strictly focus your response and options on this specific phase.\n"
            f"ANALYSIS FOCUS: {step_guidance}\n\n"
            f"APPROVED PROJECT ARTIFACTS (Use these as context):\n{artifacts_text}\n"
        )
        if intent != "audit":
            state_instruction += f"\nSTYLE HINT: {random.choice(STYLE_HINTS)}\n"

        # 4. Build Prompt (with prior conversation turns so the model keeps context).
        # History is only meaningful for the guided Elicitation flow. Audit mode is meant to be
        # a stateless one-shot scoring tool (see AUDIT_PROMPT's CORE MISSION) - mixing in prior
        # Elicitation-formatted turns was observed to make the small model imitate that format
        # instead of producing an Audit Report, even though the correct system prompt was active.
        # History is additionally filtered to turns tagged with the SAME intent as the current
        # request (untagged/legacy turns are excluded) - a small model was found to imitate
        # whichever format dominates recent history regardless of which system prompt is active,
        # so an Elicitation request following Audit turns (or vice versa) needs its history scrubbed
        # of the other mode's turns, not just Audit requests scrubbed of Elicitation history.
        if intent == "audit":
            history_text = ""
        else:
            same_intent_history = [m for m in (history or []) if m.get("intent") == intent]
            history_text = self._format_history(same_intent_history)
        full_prompt = self._build_full_prompt(active_prompt, state_instruction, context_text, history_text, user_query)

        # Defensively trim if the assembled prompt risks exceeding the model's context window,
        # since GPT4All surfaces an overflow as plain generated text rather than a Python exception.
        # Each step re-derives full_prompt from its components rather than slicing the assembled
        # string directly, so the closing "<|start_header_id|>assistant<|end_header_id|>" turn-opener
        # is never lost - a naive prefix-slice was observed to strip it, causing the model to echo
        # raw system-prompt text back as its "answer" instead of generating a real response.
        if len(full_prompt) > PROMPT_CHAR_BUDGET:
            history_text = ""
            full_prompt = self._build_full_prompt(active_prompt, state_instruction, context_text, history_text, user_query)
        if len(full_prompt) > PROMPT_CHAR_BUDGET:
            context_text = context_text[:2000]
            full_prompt = self._build_full_prompt(active_prompt, state_instruction, context_text, history_text, user_query)
        if len(full_prompt) > PROMPT_CHAR_BUDGET:
            context_text = ""
            full_prompt = self._build_full_prompt(active_prompt, state_instruction, context_text, history_text, user_query)
        if len(full_prompt) > PROMPT_CHAR_BUDGET:
            # Last resort: the user_query itself is still too large for the budget. Shrink it
            # directly (rather than slicing the assembled prompt) so prompt structure stays intact.
            overshoot = len(full_prompt) - PROMPT_CHAR_BUDGET
            user_query = user_query[:max(0, len(user_query) - overshoot)]
            full_prompt = self._build_full_prompt(active_prompt, state_instruction, context_text, history_text, user_query)

        # 5. Generate
        # Elicitation replies are meant to stay short (ONE question + 2-3 options per ELICITATION_PROMPT),
        # so they get a tighter token cap than the longer, structured Audit Report - shorter cap means
        # less decode time, which is the dominant cost of a turn on weaker CPU-only hardware.
        max_tokens = 400 if intent == "audit" else 250
        temp = AUDIT_TEMP if intent == "audit" else ELICITATION_TEMP
        try:
            response = self.llm.generate(
                full_prompt,
                max_tokens=max_tokens,
                temp=temp,
                top_k=40,
                top_p=0.4,
                n_batch=self.n_batch,
                callback=self._make_stop_callback()
            )
        except Exception as e:
            print(f"--- Generation error: {e} ---")
            response = "⚠️ Sorry, something went wrong while generating a response. Please try again."

        response = self._clean_response(response)

        print(f"--- Generation took {round(time.time() - start_time, 2)} seconds ---")

        return {
            "answer": response,
            "sources": sources
        }

    def generate_finalize_summary(self, process_state: str, history=None) -> str:
        """Turns the elicitation conversation for the current step into a single clean,
        self-contained summary suitable to save as a final artifact - an explicit alternative
        to relying on the model to naturally arrive at a clean closing message, which was
        observed to instead produce content-free small talk and drift onto the next step."""
        same_step_history = [
            m for m in (history or [])
            if m.get("intent") == "elicitation" and m.get("process_state") == process_state
        ]
        history_text = self._format_history(same_step_history, max_turns=10)

        if not history_text:
            return f"There isn't enough conversation yet about this {process_state} to summarize. Discuss it a bit more first."

        system_prompt = FINALIZE_PROMPT.format(process_state=process_state)
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>\n"
            f"{history_text}"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Please summarize the {process_state} defined above.<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        try:
            response = self.llm.generate(
                full_prompt,
                max_tokens=200,
                temp=0.2,
                top_k=40,
                top_p=0.4,
                n_batch=self.n_batch,
                callback=self._make_stop_callback()
            )
        except Exception as e:
            print(f"--- Finalize generation error: {e} ---")
            return "⚠️ Sorry, something went wrong while summarizing. Please try again."

        return self._clean_response(response)

    def scan_full_document(self, session_id: str, filename: str):
        """Runs a cheap extraction pass over every chunk of an uploaded document,
        asking the model to propose Asset/Threat/Goal/Req artifacts implied by the
        text. Used for the auto-scan that runs in the background after a PDF upload
        (see app.api.documents). Falls back to a regex scrape if the model's JSON
        output doesn't parse cleanly - small models often produce near-JSON with a
        stray trailing comma or unescaped quote."""
        results = ingestor.collection.get(
            where={"$and": [{"session_id": session_id}, {"source": filename}]}
        )

        if not results['documents']:
            return []

        all_artifacts = []

        print(f"--- Starting Full Document Scan for {filename} ({len(results['documents'])} chunks) ---")

        for i, chunk in enumerate(results['documents']):
            print(f"Scanning chunk {i+1}/{len(results['documents'])}...")

            extraction_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a Security Architect. Analyze the following functional description.\n"
                f"Task: Identify implicit and explicit security artifacts based on the system's functionality.\n"
                f"Focus areas for requirements include Volere types: Access, Integrity, Privacy, Audit, Immunity.\n"
                f"1. Assets: What data or components have value?\n"
                f"2. Threats: What could happen to these assets?\n"
                f"3. Goals: High-level security objectives.\n"
                f"4. Req: Concrete security requirements derived from the description.\n"
                f"Return ONLY a valid JSON array using DOUBLE QUOTES.\n"
                f"Format strictly like this: [{{\"type\": \"Asset\", \"content\": \"User Study Data\"}}]\n"
                f"Allowed types: Asset, Threat, Goal, Req.\n<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"TEXT TO ANALYZE:\n{chunk}<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n["
            )

            response = self.llm.generate(extraction_prompt, max_tokens=250, temp=0.1, n_batch=self.n_batch)
            full_response = "[" + response

            artifacts = []
            try:
                start_idx = full_response.find('[')
                end_idx = full_response.rfind(']') + 1
                clean_string = full_response[start_idx:end_idx] if end_idx > 0 else full_response

                try:
                    artifacts = json.loads(clean_string)
                except Exception:
                    try:
                        artifacts = ast.literal_eval(clean_string)
                    except Exception:
                        raise ValueError("Syntax broken")

            except Exception:
                print(f"-> Syntax broken in chunk {i+1}. Engaging Regex Fallback...")
                blocks = re.findall(r'\{[^{}]*\}', full_response)
                for block in blocks:
                    t_match = re.search(r'[\'"]type[\'"]\s*:\s*[\'"](Asset|Threat|Goal|Req)[\'"]', block, re.IGNORECASE)
                    c_match = re.search(r'[\'"]content[\'"]\s*:\s*[\'"](.*?)[\'"]\s*\}', block, re.DOTALL | re.IGNORECASE)

                    if t_match and c_match:
                        artifacts.append({
                            "type": t_match.group(1).capitalize(),
                            "content": c_match.group(1).strip()
                        })

            if isinstance(artifacts, list):
                all_artifacts.extend(artifacts)
                print(f"-> Extracted {len(artifacts)} artifacts from chunk {i+1}.")

        unique_artifacts = []
        seen = set()
        for item in all_artifacts:
            content_lower = item.get("content", "").lower().strip()
            art_type = item.get("type", "")

            if content_lower and art_type in ["Asset", "Threat", "Goal", "Req"]:
                if content_lower not in seen:
                    seen.add(content_lower)
                    unique_artifacts.append(item)

        print(f"--- Scan complete. Found {len(unique_artifacts)} unique artifacts. ---")
        return unique_artifacts


# Shared singleton - loads the model once at import time. Imported by the chat/document
# routers in app.api instead of each constructing their own RagEngine.
rag_engine = RagEngine()
