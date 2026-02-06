import os
# Wir importieren direkt deine Funktion, um zu sehen, ob sie crasht
from app.ingest import process_pdf

# 1. Hier deinen ECHTEN Dateinamen aus 'uploaded_docs' eintragen:
REAL_FILENAME = "1a4548c6-3be0-4be2-b190-2e74b3e37ef7_best_practices_secureflow_inc.pdf" 
FILE_PATH = os.path.join("uploaded_docs", REAL_FILENAME)

# Eine Fake-Session-ID für den Test
TEST_SESSION_ID = "test_session_123"

print(f"--- STARTE MANUELLEN TEST FÜR: {FILE_PATH} ---")

if not os.path.exists(FILE_PATH):
    print("❌ FEHLER: Datei nicht gefunden! Hast du den Namen oben angepasst?")
    exit()

try:
    # Hier wird es wahrscheinlich knallen, wenn ingest.py falsch ist
    num_chunks = process_pdf(FILE_PATH, session_id=TEST_SESSION_ID)
    
    if num_chunks > 0:
        print(f"✅ ERFOLG! {num_chunks} Chunks wurden in die DB geschrieben.")
    else:
        print("⚠️ WARNUNG: Prozess lief durch, aber 0 Chunks erstellt (PDF leer/nicht lesbar?).")

except TypeError as te:
    print("\n❌ KRITISCHER FEHLER (TypeError):")
    print(te)
    print("\n👉 URSACHE: Deine 'app/ingest.py' ist veraltet!")
    print("Sie akzeptiert 'session_id' nicht als Argument. Bitte update den Code (siehe unten).")

except Exception as e:
    print(f"\n❌ ALLGEMEINER FEHLER: {e}")