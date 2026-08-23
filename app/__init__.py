"""Security Requirements Assistant - a local, RAG-powered chatbot that guides a user
through a 4-step Security Requirements Engineering process (Asset -> Threat -> Goal ->
Requirement), backed by GPT4All (Llama-3.2-3B-Instruct), ChromaDB, and SQLite.

Package layout:
- `app.config`      - all tunable constants (paths, model settings, generation limits)
- `app.core`         - the stateful engines: database access, document ingestion, the LLM/RAG engine
- `app.api`           - FastAPI routers, one module per resource (sessions, documents, artifacts, chat)
- `app.main`           - composes the FastAPI app from the pieces above
"""
