"""Stateful engines the rest of the app depends on: the SQLite repository (`database`),
the ChromaDB/PDF ingestion layer (`ingestion`), and the GPT4All-backed RAG engine
(`rag_engine`). Each module exposes a single ready-to-use singleton instance (`db`,
`ingestor`, `rag_engine`) that the API routers in `app.api` import directly, mirroring
how the original flat modules worked (one shared object per concern, no dependency
injection framework needed for an app this size).
"""
