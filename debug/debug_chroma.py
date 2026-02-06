import os
import chromadb
from chromadb.utils import embedding_functions

# Pfad zur DB
DB_DIR = os.path.join(os.getcwd(), "chroma_db")
client = chromadb.PersistentClient(path=DB_DIR)

print(f"--- UNTERSUCHE DATENBANK IN: {DB_DIR} ---")

try:
    collection = client.get_collection("security_docs")
    count = collection.count()
    print(f"Gesamtanzahl Chunks in DB: {count}")
    
    if count == 0:
        print("WARNUNG: Die Datenbank ist leer!")
    else:
        # Hole alle Metadaten
        data = collection.get(include=["metadatas"])
        metadatas = data["metadatas"]
        
        # Zähle Dokumente pro Session
        stats = {}
        for meta in metadatas:
            sess_id = meta.get("session_id", "MISSING_ID")
            source = meta.get("source", "UNKNOWN_FILE")
            
            if sess_id not in stats:
                stats[sess_id] = set()
            stats[sess_id].add(source)
            
        print("\n--- GEFUNDENE SESSIONS & DATEIEN ---")
        for sess, files in stats.items():
            print(f"Session ID: '{sess}'")
            for f in files:
                print(f"  - {f}")
                
except Exception as e:
    print(f"Fehler beim Zugriff auf ChromaDB: {e}")
    print("Existiert der Ordner 'chroma_db' überhaupt?")