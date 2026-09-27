"""
messages.py - a minimal message board for the SME team (part of the admin shell).

The prescriptive loop needs a human hand-off: the boss sees "RESTOCK 22411" on the
prediction page and tells an employee to act on it. This is the thinnest version of
that hand-off - the boss (admin) posts a message to one employee or to everyone; each
user reads the messages addressed to them plus the broadcasts. No threads, no editing,
no read-receipts: deliberately small (the admin shell is the documented timeboxed
module).

Storage: same SQLite file as the users (data/users.db).
"""
import sqlite3

from flask import jsonify, request

from ..config import USERS_DB
from .auth import require_auth

MAX_BODY = 500


def _db():
    con = sqlite3.connect(USERS_DB)
    con.row_factory = sqlite3.Row
    return con


def init_messages():
    with _db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            from_user  TEXT NOT NULL,
            to_user    TEXT,                -- NULL = broadcast to everyone
            body       TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')))""")


def register_message_routes(app):
    @app.get("/api/messages")
    @require_auth()
    def list_messages():
        """Broadcasts + messages addressed to me + (for the admin) messages I sent."""
        me = request.user["u"]
        with _db() as con:
            rows = con.execute(
                """SELECT id, from_user, to_user, body, created_at FROM messages
                   WHERE to_user IS NULL OR to_user = ? OR from_user = ?
                   ORDER BY id DESC LIMIT 100""", (me, me)).fetchall()
        return jsonify([dict(r) for r in rows])

    @app.post("/api/messages")
    @require_auth(role="admin")
    def post_message():
        b = request.get_json(silent=True) or {}
        to = b.get("to") or None                 # None / "" -> broadcast
        body = str(b.get("body", "")).strip()
        if not body or len(body) > MAX_BODY:
            return jsonify({"error": f"message must be 1..{MAX_BODY} characters"}), 400
        with _db() as con:
            if to is not None:
                if con.execute("SELECT 1 FROM users WHERE username = ?", (to,)).fetchone() is None:
                    return jsonify({"error": f"no such user: {to}"}), 404
            con.execute("INSERT INTO messages (from_user, to_user, body) VALUES (?,?,?)",
                        (request.user["u"], to, body))
        return jsonify({"ok": True})
