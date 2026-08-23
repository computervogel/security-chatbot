"""SQLite persistence layer for chat sessions, messages, and security artifacts.

This is the only place in the app that talks to sqlite3 directly. Every route
handler in `app.api` goes through the shared `db` singleton defined at the
bottom of this file instead of opening its own connection.

Three tables matter here:
- `sessions`  - one row per chat conversation (id, title, created_at)
- `messages`  - every chat turn, tagged with which of the 4 process steps
                (Asset/Threat/Goal/Req) and which mode (elicitation/audit) it
                belongs to, so later turns can be filtered by both
- `artifacts` - the security artifacts a user has saved from the chat, each
                starting as "pending" until explicitly "approved" in the
                dashboard (see the accept/refine/discard workflow in the UI)
- `feedback`  - raw thumbs-up/down ratings on bot messages
"""
import sqlite3
import uuid
from app.config import DB_NAME


class ChatDatabase:
    """Thin repository wrapper around the app's SQLite database.

    Opens and closes a fresh connection per call rather than holding one open
    connection across requests - simple and safe under FastAPI's threaded
    request handling, and sqlite/file-based I/O is cheap enough at this scale
    that connection pooling isn't worth the added complexity.
    """

    def __init__(self, db_name: str = DB_NAME):
        self.db_name = db_name

    def _connect(self):
        return sqlite3.connect(self.db_name)

    def init_db(self):
        """Creates all tables if they don't exist yet, and applies additive
        column migrations for tables that existed before those columns were
        introduced. Migrations are wrapped in try/except so re-running this
        against an already-migrated database is a harmless no-op - this is
        called explicitly at app startup (see app.main), not from __init__,
        so the startup sequence stays visible in one place."""
        conn = self._connect()
        c = conn.cursor()
        # Table for Chat Sessions
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Table for Messages
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')
        # Table for Feedback
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                bot_message TEXT,
                rating TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Table for accepted artifacts
        c.execute('''
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                type TEXT,
                content TEXT,
                source_file TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Migrations for columns added after initial release
        try:
            c.execute("ALTER TABLE artifacts ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'")
        except sqlite3.OperationalError:
            pass  # column already exists

        try:
            c.execute("ALTER TABLE messages ADD COLUMN process_state TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

        try:
            c.execute("ALTER TABLE messages ADD COLUMN intent TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

        conn.commit()
        conn.close()

    # --- Sessions ---

    def create_session(self, title="New Chat"):
        session_id = str(uuid.uuid4())
        conn = self._connect()
        c = conn.cursor()
        c.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (session_id, title))
        conn.commit()
        conn.close()
        return session_id

    def get_sessions(self):
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_session_title(self, session_id):
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
        row = c.fetchone()
        conn.close()
        return row["title"] if row else "Security Requirements"

    def delete_session(self, session_id):
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))  # Cleanup messages
        conn.commit()
        conn.close()

    def update_session_title(self, session_id: str, new_title: str):
        conn = self._connect()
        c = conn.cursor()
        c.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
        conn.close()

    # --- Messages ---

    def add_message(self, session_id, role, content, process_state=None, intent=None):
        conn = self._connect()
        c = conn.cursor()
        c.execute("INSERT INTO messages (session_id, role, content, process_state, intent) VALUES (?, ?, ?, ?, ?)",
                  (session_id, role, content, process_state, intent))
        conn.commit()
        conn.close()

    def get_session_history(self, session_id):
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT role, content, process_state, intent FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
        rows = c.fetchall()
        conn.close()
        return [{"role": r["role"], "content": r["content"], "process_state": r["process_state"], "intent": r["intent"]} for r in rows]

    # --- Feedback ---

    def save_feedback(self, session_id, bot_message, rating):
        conn = self._connect()
        c = conn.cursor()
        c.execute("INSERT INTO feedback (session_id, bot_message, rating) VALUES (?, ?, ?)",
                  (session_id, bot_message, rating))
        conn.commit()
        conn.close()

    # --- Artifacts ---

    def save_artifact(self, session_id, artifact_type, content, source_file=None):
        conn = self._connect()
        c = conn.cursor()
        c.execute("INSERT INTO artifacts (session_id, type, content, source_file) VALUES (?, ?, ?, ?)",
                  (session_id, artifact_type, content, source_file))
        conn.commit()
        conn.close()

    def get_session_artifacts(self, session_id, status=None):
        """Returns all artifacts for a session, optionally filtered to a single
        status ('pending' or 'approved'). Only 'approved' artifacts should ever
        be fed back into the RAG engine as project context - see RagEngine."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        if status:
            c.execute("SELECT id, type, content, status FROM artifacts WHERE session_id = ? AND status = ? ORDER BY timestamp ASC", (session_id, status))
        else:
            c.execute("SELECT id, type, content, status FROM artifacts WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
        rows = c.fetchall()
        conn.close()
        return [{"id": r["id"], "type": r["type"], "content": r["content"], "status": r["status"]} for r in rows]

    def approve_artifact(self, artifact_id: int):
        conn = self._connect()
        c = conn.cursor()
        c.execute("UPDATE artifacts SET status = 'approved' WHERE id = ?", (artifact_id,))
        conn.commit()
        conn.close()

    def update_artifact_content(self, artifact_id: int, content: str):
        """Overwrites an artifact's content (used by the dashboard's Refine action).
        Status is left unchanged - refining an already-approved artifact doesn't
        require re-approval."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("UPDATE artifacts SET content = ? WHERE id = ?", (content, artifact_id))
        conn.commit()
        conn.close()

    def delete_single_artifact(self, artifact_id: int):
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        conn.commit()
        conn.close()

    def delete_artifacts_by_source(self, session_id, source_file):
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM artifacts WHERE session_id = ? AND source_file = ?", (session_id, source_file))
        conn.commit()
        conn.close()


# Shared singleton - imported by app.core.ingestion, app.core.rag_engine, and every
# router in app.api instead of each opening its own ChatDatabase.
db = ChatDatabase()
