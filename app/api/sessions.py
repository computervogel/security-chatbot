"""Session-level routes: create/list/delete/rename a chat session, and read
its message history. A "session" is one chat conversation - each one gets
its own row in the `sessions` table and its own uploaded documents/artifacts.
"""
from fastapi import APIRouter, Form

from app.core.database import db
from app.core.ingestion import ingestor

router = APIRouter()


@router.get("/sessions")
async def list_sessions():
    """Returns a list of all chat sessions."""
    return db.get_sessions()


@router.post("/sessions")
async def new_session():
    """Creates a new empty session."""
    session_id = db.create_session(title="New Chat")
    return {"id": session_id, "title": "New Chat"}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Deletes a session and its associated ChromaDB documents."""
    db.delete_session(session_id)          # SQL Cleanup
    ingestor.delete_session_docs(session_id)  # Vector DB Cleanup
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/rename")
async def rename_session(session_id: str, title: str = Form(...)):
    """Renames an existing session."""
    db.update_session_title(session_id, title)
    return {"status": "success", "new_title": title}


@router.get("/sessions/{session_id}/history")
async def history(session_id: str):
    """Fetches chat history for a specific session."""
    return db.get_session_history(session_id)


def auto_rename_if_first_message(session_id: str, current_history: list, user_message: str):
    """If this is the session's very first message, rename the session from
    "New Chat" to a snippet of that message - shared helper so app.api.chat
    can reuse it without duplicating the sqlite access itself (that stays
    inside ChatDatabase, see app.core.database)."""
    if current_history:
        return
    new_title = (user_message[:30] + '...') if len(user_message) > 30 else user_message
    db.update_session_title(session_id, new_title)
