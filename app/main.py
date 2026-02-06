import os
import shutil
import sqlite3
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Import internal modules
from app.ingest import process_pdf, ingest_supervisor_docs, get_session_documents, delete_document, delete_session_docs
from app.rag import generate_response
from app.database import init_db, add_message, get_session_history, create_session, get_sessions, delete_session, DB_NAME

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# --- STARTUP ---
# Initialize DB and create folders
init_db()
os.makedirs("uploaded_docs", exist_ok=True)

# Load global supervisor knowledge base (once on startup)
print("--- Ingesting Supervisor Docs ---")
ingest_supervisor_docs()

# --- ROUTES: UI ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main Chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})

# --- ROUTES: SESSION MANAGEMENT ---
@app.get("/sessions")
async def list_sessions():
    """Returns a list of all chat sessions."""
    return get_sessions()

@app.post("/sessions")
async def new_session():
    """Creates a new empty session."""
    session_id = create_session(title="New Chat")
    return {"id": session_id, "title": "New Chat"}

@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Deletes a session and its associated ChromaDB documents."""
    delete_session(session_id)      # SQL Cleanup
    delete_session_docs(session_id) # Vector DB Cleanup
    return {"status": "deleted"}

@app.get("/sessions/{session_id}/history")
async def history(session_id: str):
    """Fetches chat history for a specific session."""
    return get_session_history(session_id)

# --- ROUTES: DOCUMENT MANAGEMENT ---
@app.get("/sessions/{session_id}/documents")
async def list_docs(session_id: str):
    """Returns files uploaded to this specific session."""
    files = get_session_documents(session_id)
    return {"documents": files}

@app.delete("/sessions/{session_id}/documents/{filename}")
async def remove_doc(session_id: str, filename: str):
    """Removes a specific document from the vector store."""
    delete_document(session_id, filename)
    return {"status": "removed"}

# --- ROUTES: CHAT & UPLOAD ---
@app.post("/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    """Handles PDF upload. Isolates file context to the session_id."""
    # Prefix filename with session_id to avoid collisions on disk
    file_path = os.path.join("uploaded_docs", f"{session_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Ingest into ChromaDB with session metadata
    process_pdf(file_path, session_id=session_id)
    return {"filename": file.filename}

@app.post("/chat")
async def chat(session_id: str = Form(...), user_message: str = Form(...)):
    """Main chat logic with Auto-Rename feature."""
    
    # 1. Auto-Rename Logic: If history is empty, rename session based on first message
    current_history = get_session_history(session_id)
    if not current_history:
        # Take first 30 chars as title
        new_title = (user_message[:30] + '...') if len(user_message) > 30 else user_message
        
        # Direct DB update (quick & dirty)
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
        conn.close()

    # 2. Save User Message
    add_message(session_id, "user", user_message)
    
    # 3. Generate Answer (searches only in session_id + supervisor docs)
    response_data = generate_response(user_message, session_id)
    
    # 4. Format and Save Bot Message
    bot_text = response_data['answer']
    if response_data['sources']:
        bot_text += f"\n\n*(Sources: {', '.join(response_data['sources'])})*"
        
    add_message(session_id, "bot", bot_text)
    
    return response_data