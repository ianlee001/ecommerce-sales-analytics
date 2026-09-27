"""
auth.py - authentication + the thinnest admin-shell endpoints (dashboard modules 1-2).

Module 1 (auth) and module 2 (generic admin shell) manage USERS and the SYSTEM - never
the retail data, which stays read-only everywhere. This is the documented
lowest-priority, timeboxed part of the dashboard (project-brief s8), so it is kept
deliberately small and conventional:

  * users live in SQLite (data/users.db, gitignored); passwords are stored as salted
    hashes (werkzeug generate_password_hash) - never plaintext.
  * login returns a SIGNED, EXPIRING token (itsdangerous, 12 h) carrying username+role;
    protected endpoints expect  Authorization: Bearer <token>.
  * roles: 'admin' (can manage users) and 'user'. The first run seeds admin/admin123
    and prints a warning - a documented demo convenience, change it before showing.
"""
import os
import re
import sqlite3
from functools import wraps

from flask import jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

from ..config import USERS_DB

SECRET_KEY = os.environ.get("FYP_SECRET_KEY", "dev-secret-change-me")
# Registering as the OWNER (admin) requires this invite code - otherwise anyone could
# sign up as the boss. Change via env for real demos; documented demo default.
ADMIN_CODE = os.environ.get("FYP_ADMIN_CODE", "FYP-OWNER")
TOKEN_MAX_AGE_S = 12 * 3600
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
MIN_PASSWORD_LEN = 6
ROLES = ("admin", "staff")

_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="fyp-auth")


def _db():
    con = sqlite3.connect(USERS_DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    USERS_DB.parent.mkdir(parents=True, exist_ok=True)
    with _db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'staff',
            created_at    TEXT NOT NULL DEFAULT (datetime('now')))""")
        con.execute("UPDATE users SET role = 'staff' WHERE role = 'user'")  # rename migration
        if con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            con.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                        ("admin", generate_password_hash("admin123"), "admin"))
            print("[auth] seeded default admin/admin123 - change this password before demos")


def _issue(row) -> str:
    return _serializer.dumps({"u": row["username"], "r": row["role"]})


def _verify(token: str):
    try:
        return _serializer.loads(token, max_age=TOKEN_MAX_AGE_S)
    except (BadSignature, SignatureExpired):
        return None


def require_auth(role: str | None = None):
    """Protect an endpoint; optionally require a specific role ('admin')."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            hdr = request.headers.get("Authorization", "")
            claims = _verify(hdr[7:]) if hdr.startswith("Bearer ") else None
            if claims is None:
                return jsonify({"error": "authentication required"}), 401
            if role and claims["r"] != role:
                return jsonify({"error": f"{role} role required"}), 403
            request.user = claims          # {'u': username, 'r': role}
            return fn(*args, **kwargs)
        return wrapper
    return deco


def register_auth_routes(app):
    @app.post("/api/auth/register")
    def register():
        b = request.get_json(silent=True) or {}
        username, password = str(b.get("username", "")), str(b.get("password", ""))
        role = str(b.get("role", "staff"))
        if not USERNAME_RE.match(username):
            return jsonify({"error": "username must be 3-30 letters/digits/underscores"}), 400
        if len(password) < MIN_PASSWORD_LEN:
            return jsonify({"error": f"password must be at least {MIN_PASSWORD_LEN} characters"}), 400
        if role not in ROLES:
            return jsonify({"error": "role must be 'admin' or 'staff'"}), 400
        if role == "admin" and str(b.get("admin_code", "")) != ADMIN_CODE:
            return jsonify({"error": "owner registration needs the correct owner code"}), 403
        try:
            with _db() as con:
                con.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                            (username, generate_password_hash(password), role))
        except sqlite3.IntegrityError:
            return jsonify({"error": "username already taken"}), 409
        return jsonify({"ok": True})

    @app.post("/api/auth/login")
    def login():
        b = request.get_json(silent=True) or {}
        with _db() as con:
            row = con.execute("SELECT * FROM users WHERE username = ?",
                              (str(b.get("username", "")),)).fetchone()
        if row is None or not check_password_hash(row["password_hash"],
                                                  str(b.get("password", ""))):
            return jsonify({"error": "wrong username or password"}), 401
        return jsonify({"token": _issue(row), "username": row["username"], "role": row["role"]})

    @app.get("/api/me")
    @require_auth()
    def me():
        return jsonify({"username": request.user["u"], "role": request.user["r"]})

    @app.post("/api/auth/change-password")
    @require_auth()
    def change_password():
        b = request.get_json(silent=True) or {}
        old, new = str(b.get("old_password", "")), str(b.get("new_password", ""))
        if len(new) < MIN_PASSWORD_LEN:
            return jsonify({"error": f"new password must be at least {MIN_PASSWORD_LEN} characters"}), 400
        with _db() as con:
            row = con.execute("SELECT * FROM users WHERE username = ?",
                              (request.user["u"],)).fetchone()
            if not check_password_hash(row["password_hash"], old):
                return jsonify({"error": "old password is wrong"}), 401
            con.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                        (generate_password_hash(new), row["id"]))
        return jsonify({"ok": True})

    # ---- thinnest staff management ----
    # The staff LIST is visible to every signed-in user (it is the team page and the
    # message-board recipient list); every MUTATION below stays admin-only.
    @app.get("/api/users")
    @require_auth()
    def list_users():
        with _db() as con:
            rows = con.execute(
                "SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()
        return jsonify([dict(r) for r in rows])

    @app.put("/api/users/<int:uid>")
    @require_auth(role="admin")
    def update_user(uid: int):
        b = request.get_json(silent=True) or {}
        role = str(b.get("role", ""))
        if role not in ROLES:
            return jsonify({"error": "role must be 'admin' or 'staff'"}), 400
        with _db() as con:
            row = con.execute("SELECT username FROM users WHERE id = ?", (uid,)).fetchone()
            if row is None:
                return jsonify({"error": "no such user"}), 404
            if row["username"] == request.user["u"]:
                return jsonify({"error": "you cannot change your own role"}), 400
            con.execute("UPDATE users SET role = ? WHERE id = ?", (role, uid))
        return jsonify({"ok": True})

    @app.post("/api/users/<int:uid>/password")
    @require_auth(role="admin")
    def reset_password(uid: int):
        b = request.get_json(silent=True) or {}
        new = str(b.get("new_password", ""))
        if len(new) < MIN_PASSWORD_LEN:
            return jsonify({"error": f"new password must be at least {MIN_PASSWORD_LEN} characters"}), 400
        with _db() as con:
            row = con.execute("SELECT username FROM users WHERE id = ?", (uid,)).fetchone()
            if row is None:
                return jsonify({"error": "no such user"}), 404
            if row["username"] == request.user["u"]:
                return jsonify({"error": "use change-password for your own account"}), 400
            con.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                        (generate_password_hash(new), uid))
        return jsonify({"ok": True})

    @app.delete("/api/users/<int:uid>")
    @require_auth(role="admin")
    def delete_user(uid: int):
        with _db() as con:
            row = con.execute("SELECT username FROM users WHERE id = ?", (uid,)).fetchone()
            if row is None:
                return jsonify({"error": "no such user"}), 404
            if row["username"] == request.user["u"]:
                return jsonify({"error": "you cannot delete yourself"}), 400
            con.execute("DELETE FROM users WHERE id = ?", (uid,))
        return jsonify({"ok": True})
