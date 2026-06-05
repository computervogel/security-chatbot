import os
import shutil
import sqlite3
from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Import internal modules
from app.ingest import process_pdf, ingest_supervisor_docs, get_session_documents, delete_document, delete_session_docs
from app.rag import generate_response, scan_full_document
from app.database import init_db, update_session_title, add_message, get_session_history, create_session, get_sessions, delete_session, delete_single_artifact, save_feedback, get_session_artifacts, save_artifact, DB_NAME

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
    return templates.TemplateResponse(request=request, name="index.html")

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

@app.post("/sessions/{session_id}/rename")
async def rename_session(session_id: str, title: str = Form(...)):
    """Renames an existing session."""
    update_session_title(session_id, title)
    return {"status": "success", "new_title": title}

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

# --- ROUTES: ARTIFACTS ---
@app.get("/sessions/{session_id}/artifacts")
async def list_artifacts(session_id: str):
    artifacts = get_session_artifacts(session_id)
    return {"artifacts": artifacts}

@app.delete("/artifacts/{artifact_id}")
async def remove_artifact(artifact_id: int):
    """Removes a specific artifact."""
    delete_single_artifact(artifact_id)
    return {"status": "removed"}

@app.post("/artifacts")
async def create_artifact(
    session_id: str = Form(...), 
    artifact_type: str = Form(...), 
    content: str = Form(...)
):
    save_artifact(session_id, artifact_type, content)
    return {"status": "saved"}

# --- ROUTES: CHAT & UPLOAD ---
@app.post("/upload")
async def upload_file(
    session_id: str = Form(...), 
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Handles PDF upload and starts auto-scan in the background."""
    file_path = os.path.join("uploaded_docs", f"{session_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Ingest into ChromaDB
    process_pdf(file_path, session_id=session_id)
    
    # 2. Starting scan in background
    if background_tasks:
        db_filename = f"{session_id}_{file.filename}"
        background_tasks.add_task(background_scan_task, session_id, db_filename)
            
    # 3. Answer for
    return {"filename": file.filename, "status": "processing_in_background"}

@app.post("/chat")
async def chat(
    session_id: str = Form(...), 
    user_message: str = Form(...),
    intent: str = Form("qa"),
    process_state: str = Form("Asset")
):
    """Main chat logic with Auto-Rename feature and Intent Routing."""
    
    # 1. Auto-Rename Logic
    current_history = get_session_history(session_id)
    if not current_history:
        new_title = (user_message[:30] + '...') if len(user_message) > 30 else user_message
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
        conn.close()

    # 2. Save User Message
    add_message(session_id, "user", user_message)
    
    # 3. Generate Answer
    response_data = generate_response(user_message, session_id, intent=intent, process_state=process_state)
    
    # 4. Format and Save Bot Message
    bot_text = response_data['answer']
    if response_data['sources']:
        bot_text += f"\n\n*(Sources: {', '.join(response_data['sources'])})*"
        
    add_message(session_id, "bot", bot_text)
    
    return response_data

@app.post("/feedback")
async def receive_feedback(
    session_id: str = Form(...), 
    bot_message: str = Form(...), 
    rating: str = Form(...) # 'up' oder 'down'
):
    """Stores user feedback silently."""
    save_feedback(session_id, bot_message, rating)
    return {"status": "success"}

@app.post("/sessions/{session_id}/scan/{filename}")
async def scan_doc(session_id: str, filename: str):
    """Scans a document and automatically saves extracted artifacts."""
    extracted_items = scan_full_document(session_id, filename)
    
    saved_count = 0
    for item in extracted_items:
        # Varify that the AI handels the format correctly
        if "type" in item and "content" in item and item["type"] in ["Asset", "Threat", "Goal", "Req"]:
            save_artifact(session_id, item["type"], item["content"])
            saved_count += 1
            
    return {"status": "success", "found_and_saved": saved_count}

def background_scan_task(session_id: str, filename: str):
    print(f"--- Background Task Started for {filename} ---")
    extracted_items = scan_full_document(session_id, filename)
    
    saved_count = 0
    for item in extracted_items:
        if "type" in item and "content" in item and item["type"] in ["Asset", "Threat", "Goal", "Req"]:
            save_artifact(session_id, item["type"], item["content"], source_file=filename)
            saved_count += 1
            
    print(f"--- Background Task Finished! Saved {saved_count} artifacts. ---")
