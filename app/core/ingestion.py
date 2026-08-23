"""Document ingestion and retrieval, backed by a persistent ChromaDB collection.

Uploaded PDFs (and the supervisor's shared knowledge-base PDFs) are chunked
and embedded into one collection ("security_docs"), tagged in ChromaDB
metadata with `session_id` so a query can be scoped to "this session's docs
plus the globally shared supervisor docs" (see `query_context`). That query
is what `RagEngine.generate_response` uses to pull retrieval context - this
module owns all direct ChromaDB access so `RagEngine` never has to reach into
`.collection` itself.
"""
import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from app.config import CHROMA_DIR, KNOWLEDGE_BASE_DIR, UPLOAD_DIR, EMBEDDING_MODEL_NAME, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP, RETRIEVAL_N_RESULTS
from app.core.database import db


class DocumentIngestor:
    """Owns the ChromaDB client/collection and all PDF chunking/embedding logic."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name="security_docs",
            embedding_function=self.embed_fn
        )

    def process_pdf(self, file_path: str, session_id: str, source_tag="user_upload"):
        """Reads a PDF, splits it into overlapping character chunks, and upserts
        each chunk into ChromaDB tagged with the owning session_id. Returns the
        number of chunks written (0 if the PDF was empty/unreadable)."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extract = page.extract_text()
                if extract:
                    text += extract + "\n"

            chunks = []
            step = PDF_CHUNK_SIZE - PDF_CHUNK_OVERLAP
            for i in range(0, len(text), step):
                chunks.append(text[i:i + PDF_CHUNK_SIZE])

            if chunks:
                filename = os.path.basename(file_path)
                # UNIQUE ID: session_id + filename + chunk_index
                ids = [f"{session_id}_{filename}_{i}" for i in range(len(chunks))]

                metadatas = [{
                    "source": filename,
                    "type": source_tag,
                    "session_id": session_id
                } for _ in chunks]

                self.collection.upsert(
                    documents=chunks,
                    ids=ids,
                    metadatas=metadatas
                )
                print(f"-> Ingested {len(chunks)} chunks from {filename} for Session {session_id}")
                return len(chunks)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return 0

    def get_session_documents(self, session_id: str):
        """Returns the list of unique filenames uploaded to a specific session."""
        result = self.collection.get(where={"session_id": session_id}, include=["metadatas"])

        if not result['metadatas']:
            return []

        files = set()
        for meta in result['metadatas']:
            files.add(meta['source'])
        return list(files)

    def delete_document(self, session_id: str, filename: str):
        """Removes a single file's chunks from ChromaDB, any artifacts that were
        auto-extracted from it, and the uploaded file itself."""
        self.collection.delete(where={"$and": [{"session_id": session_id}, {"source": filename}]})

        db.delete_artifacts_by_source(session_id, filename)

        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"-> File deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    def delete_session_docs(self, session_id: str):
        """Removes ALL of a session's chunks from ChromaDB and their uploaded files."""
        self.collection.delete(where={"session_id": session_id})

        if os.path.exists(UPLOAD_DIR):
            prefix = f"{session_id}_"
            for f in os.listdir(UPLOAD_DIR):
                if f.startswith(prefix):
                    file_path = os.path.join(UPLOAD_DIR, f)
                    try:
                        os.remove(file_path)
                        print(f"-> Session file deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")

    def ingest_supervisor_docs(self):
        """Ingests every PDF in the knowledge_base/ folder once, under the shared
        session_id 'global' - these become the always-available background
        knowledge every session can retrieve from (see query_context's `$or`)."""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            os.makedirs(KNOWLEDGE_BASE_DIR)
            return

        print("--- Checking Supervisor Knowledge Base ---")
        files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.lower().endswith('.pdf')]

        count = 0
        for f in files:
            # Check if this file was already ingested (its first chunk's id would already exist)
            test_id = f"global_{f}_0"
            existing = self.collection.get(ids=[test_id])

            if len(existing['ids']) == 0:
                print(f"Processing Supervisor Doc: {f}")
                self.process_pdf(os.path.join(KNOWLEDGE_BASE_DIR, f), session_id="global", source_tag="supervisor")
                count += 1

        if count > 0:
            print(f"--- Added {count} documents to Knowledge Base ---")

    def query_context(self, query_text: str, session_id: str, n_results: int = RETRIEVAL_N_RESULTS):
        """Runs the retrieval query the RAG engine needs: the top-N chunks most
        relevant to `query_text`, scoped to this session's own uploads plus the
        globally shared supervisor knowledge base. Returns (context_text, sources)."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={
                "$or": [
                    {"session_id": session_id},
                    {"type": "supervisor"}
                ]
            }
        )

        context_text = ""
        sources = []
        if results['documents'] and results['documents'][0]:
            context_text = "\n\n".join(results['documents'][0])
            sources = list(set(m['source'] for m in results['metadatas'][0]))

        return context_text, sources


# Shared singleton - imported by app.core.rag_engine and the document/chat routers in app.api.
ingestor = DocumentIngestor()
