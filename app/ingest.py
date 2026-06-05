from fileinput import filename
import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# --- CONFIGURATION ---
DB_DIR = os.path.join(os.getcwd(), "chroma_db")
client = chromadb.PersistentClient(path=DB_DIR)

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="security_docs",
    embedding_function=embed_fn
)

def process_pdf(file_path: str, session_id: str, source_tag="user_upload"):
    """Reads PDF, chunks it, and stores it with Session ID in Metadata."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
        
        chunk_size = 800
        overlap = 100
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])

        if chunks:
            filename = os.path.basename(file_path)
            # UNIQUE ID: session_id + filename + chunk_index
            ids = [f"{session_id}_{filename}_{i}" for i in range(len(chunks))]
            
            # METADATA
            metadatas = [{
                "source": filename, 
                "type": source_tag,
                "session_id": session_id 
            } for _ in chunks]
            
            collection.upsert(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            print(f"-> Ingested {len(chunks)} chunks from {filename} for Session {session_id}")
            return len(chunks)
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return 0

def get_session_documents(session_id: str):
    """Returns list of unique files for a specific session."""
    result = collection.get(where={"session_id": session_id}, include=["metadatas"])
    
    if not result['metadatas']:
        return []
    
    files = set()
    for meta in result['metadatas']:
        files.add(meta['source'])
    return list(files)

def delete_document(session_id: str, filename: str):
    """Deletes a specific file from a session."""
    collection.delete(where={"$and": [{"session_id": session_id}, {"source": filename}]})

    from app.database import delete_artifacts_by_source
    delete_artifacts_by_source(session_id, filename)

    file_path = os.path.join("uploaded_docs", filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"-> File deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def delete_session_docs(session_id: str):
    """Deletes ALL files belonging to a session."""
    collection.delete(where={"session_id": session_id})
    
    directory = "uploaded_docs"
    if os.path.exists(directory):
        prefix = f"{session_id}_"
        for f in os.listdir(directory):
            if f.startswith(prefix):
                file_path = os.path.join(directory, f)
                try:
                    os.remove(file_path)
                    print(f"-> Session file deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

def ingest_supervisor_docs():
    """Ingests global docs (session_id='global')."""
    kb_path = os.path.join(os.getcwd(), "knowledge_base")
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
        return

    print("--- Checking Supervisor Knowledge Base ---")
    files = [f for f in os.listdir(kb_path) if f.lower().endswith('.pdf')]
    
    count = 0
    for f in files:
        # Check if exists (using global prefix)
        test_id = f"global_{f}_0"
        existing = collection.get(ids=[test_id])
        
        if len(existing['ids']) == 0:
            print(f"Processing Supervisor Doc: {f}")
            process_pdf(os.path.join(kb_path, f), session_id="global", source_tag="supervisor")
            count += 1
    
    if count > 0: print(f"--- Added {count} documents to Knowledge Base ---")