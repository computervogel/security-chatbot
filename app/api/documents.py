"""Document routes: upload a PDF into a session, list/remove a session's
documents, and trigger the artifact auto-scan over an uploaded document.

Uploading a PDF does two things: it's ingested into ChromaDB immediately
(so it's usable as retrieval context right away), and a background task
scans it for candidate Asset/Threat/Goal/Req artifacts, which land in the
dashboard as "pending" for the user to review (see app.api.artifacts).
"""
import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks

from app.config import UPLOAD_DIR
from app.core.database import db
from app.core.ingestion import ingestor
from app.core.rag_engine import rag_engine

router = APIRouter()


@router.get("/sessions/{session_id}/documents")
async def list_docs(session_id: str):
    """Returns files uploaded to this specific session."""
    files = ingestor.get_session_documents(session_id)
    return {"documents": files}


@router.delete("/sessions/{session_id}/documents/{filename}")
async def remove_doc(session_id: str, filename: str):
    """Removes a specific document from the vector store."""
    ingestor.delete_document(session_id, filename)
    return {"status": "removed"}


@router.post("/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Handles PDF upload and starts the artifact auto-scan in the background."""
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Ingest into ChromaDB
    ingestor.process_pdf(file_path, session_id=session_id)

    # 2. Start the artifact-extraction scan in the background so the upload response
    # doesn't block on it
    if background_tasks:
        db_filename = f"{session_id}_{file.filename}"
        background_tasks.add_task(background_scan_task, session_id, db_filename)

    return {"filename": file.filename, "status": "processing_in_background"}


@router.post("/sessions/{session_id}/scan/{filename}")
async def scan_doc(session_id: str, filename: str):
    """Manually (re-)scans a document and saves any extracted artifacts as pending."""
    extracted_items = rag_engine.scan_full_document(session_id, filename)

    saved_count = 0
    for item in extracted_items:
        if "type" in item and "content" in item and item["type"] in ["Asset", "Threat", "Goal", "Req"]:
            db.save_artifact(session_id, item["type"], item["content"])
            saved_count += 1

    return {"status": "success", "found_and_saved": saved_count}


def background_scan_task(session_id: str, filename: str):
    """Runs after an upload completes - scans the new document and saves
    whatever artifacts it finds as 'pending', ready for review in the dashboard."""
    print(f"--- Background Task Started for {filename} ---")
    extracted_items = rag_engine.scan_full_document(session_id, filename)

    saved_count = 0
    for item in extracted_items:
        if "type" in item and "content" in item and item["type"] in ["Asset", "Threat", "Goal", "Req"]:
            db.save_artifact(session_id, item["type"], item["content"], source_file=filename)
            saved_count += 1

    print(f"--- Background Task Finished! Saved {saved_count} artifacts. ---")
