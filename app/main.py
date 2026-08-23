"""App entry point: builds the FastAPI app from the pieces in app.core and
app.api. Run with `uvicorn app.main:app` from the repository root (relative
storage paths in app.config resolve against that working directory).

Startup sequence (in order, matching the app's original behavior):
1. Initialize the SQLite schema (creates tables / applies migrations)
2. Ensure the upload folder exists
3. Ingest any new supervisor knowledge-base PDFs into ChromaDB
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import UPLOAD_DIR
from app.core.database import db
from app.core.ingestion import ingestor
from app.api import sessions, documents, artifacts, chat

app = FastAPI(title="Security Requirements Assistant")
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# --- STARTUP ---
db.init_db()
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("--- Ingesting Supervisor Docs ---")
ingestor.ingest_supervisor_docs()

# --- ROUTES ---
app.include_router(sessions.router)
app.include_router(documents.router)
app.include_router(artifacts.router)
app.include_router(chat.router)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main Chat UI (a single-page app - all interaction happens
    via fetch() calls to the routers above, see app/static/js/app.js)."""
    return templates.TemplateResponse(request=request, name="index.html")
