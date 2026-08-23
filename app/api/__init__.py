"""FastAPI routers, one module per resource. Each module exposes an
`APIRouter` named `router` that `app.main` includes into the app. Handlers
here stay thin - they parse the request, delegate to the `db`/`ingestor`/
`rag_engine` singletons in `app.core`, and shape the response.
"""
