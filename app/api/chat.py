"""Chat routes: the main conversational turn (/chat), the explicit "Finalize
{Step}" action (/chat/finalize), and message feedback (/feedback).
"""
from fastapi import APIRouter, Form

from app.core.database import db
from app.core.rag_engine import rag_engine
from app.api.sessions import auto_rename_if_first_message

router = APIRouter()


@router.post("/chat")
async def chat(
    session_id: str = Form(...),
    user_message: str = Form(...),
    intent: str = Form("elicitation"),
    process_state: str = Form("Asset")
):
    """Main chat turn: saves the user's message, generates a response, and
    saves the bot's reply - both tagged with the current intent/process_state
    so future turns can be filtered by mode and step (see RagEngine.generate_response)."""

    # 1. Load prior turns BEFORE the current message is persisted, so it isn't
    # double-counted as history for its own generation call. Also doubles as
    # the check for whether this is the session's first message (auto-rename).
    current_history = db.get_session_history(session_id)
    auto_rename_if_first_message(session_id, current_history, user_message)

    # 2. Save User Message
    db.add_message(session_id, "user", user_message, process_state=process_state, intent=intent)

    # 3. Generate Answer
    response_data = rag_engine.generate_response(user_message, session_id, intent=intent, process_state=process_state, history=current_history)

    # 4. Save the raw answer (no sources suffix baked in) - persisting the display-only
    # "(Sources: ...)" text was found to make the model imitate/hallucinate fake citations
    # when this message later resurfaces as conversation history. The frontend appends the
    # sources suffix itself for display, so response_data is returned unchanged here.
    db.add_message(session_id, "bot", response_data['answer'], process_state=process_state, intent=intent)

    return response_data


@router.post("/chat/finalize")
async def finalize_step(
    session_id: str = Form(...),
    process_state: str = Form(...)
):
    """Explicitly summarizes the current step's conversation into one clean, savable
    artifact - a direct fix for the bot self-completing a topic with a content-free
    closing message and drifting to the next step instead."""
    history = db.get_session_history(session_id)
    summary = rag_engine.generate_finalize_summary(process_state, history=history)
    return {"summary": summary}


@router.post("/feedback")
async def receive_feedback(
    session_id: str = Form(...),
    bot_message: str = Form(...),
    rating: str = Form(...)  # 'up' or 'down'
):
    """Stores user feedback (thumbs up/down) on a bot message."""
    db.save_feedback(session_id, bot_message, rating)
    return {"status": "success"}
