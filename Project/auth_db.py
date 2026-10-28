"""
SQLite storage for user features (stdlib only, no extra dependencies):
  - accounts (PBKDF2-hashed passwords)
  - login sessions (random tokens, only a SHA-256 hash of the token is stored)
  - per-user lists: watchlist / favorites / history
  - per-user ratings (1-5)
"""
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("USERS_DB_PATH", os.path.join(BASE_DIR, "users.db"))

SESSION_TTL = 30 * 24 * 3600  # 30 days
PBKDF2_ITERS = 200_000
LIST_NAMES = {"watchlist", "favorites", "history"}
HISTORY_LIMIT = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt          BLOB NOT NULL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS movie_lists (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    list_name  TEXT NOT NULL CHECK (list_name IN ('watchlist','favorites','history')),
    tmdb_id    INTEGER NOT NULL,
    title      TEXT NOT NULL,
    poster_url TEXT,
    added_at   REAL NOT NULL,
    PRIMARY KEY (user_id, list_name, tmdb_id)
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tmdb_id    INTEGER NOT NULL,
    rating     INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    title      TEXT NOT NULL,
    poster_url TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, tmdb_id)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as c:
        c.executescript(SCHEMA)


# ---------- passwords ----------
def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERS
    ).hex()


def create_user(username: str, password: str) -> Optional[int]:
    """Returns the new user id, or None if the username is already taken."""
    salt = os.urandom(16)
    try:
        with get_conn() as c:
            cur = c.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, _hash_password(password, salt), salt, time.time()),
            )
            return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        return None


def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    with get_conn() as c:
        row = c.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        # burn comparable time so unknown users aren't distinguishable by timing
        _hash_password(password, b"\x00" * 16)
        return None
    expected = _hash_password(password, bytes(row["salt"]))
    if not hmac.compare_digest(expected, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


# ---------- sessions ----------
def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with get_conn() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        c.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (_token_hash(token), user_id, now + SESSION_TTL),
        )
    return token


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    with get_conn() as c:
        row = c.execute(
            "SELECT u.id, u.username FROM sessions s "
            "JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (_token_hash(token), time.time()),
        ).fetchone()
    return {"id": row["id"], "username": row["username"]} if row else None


def delete_session(token: str) -> None:
    with get_conn() as c:
        c.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


# ---------- lists (watchlist / favorites / history) ----------
def list_add(
    user_id: int,
    list_name: str,
    tmdb_id: int,
    title: str,
    poster_url: Optional[str],
) -> None:
    assert list_name in LIST_NAMES
    with get_conn() as c:
        c.execute(
            "INSERT INTO movie_lists (user_id, list_name, tmdb_id, title, poster_url, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, list_name, tmdb_id) DO UPDATE SET "
            "title = excluded.title, poster_url = excluded.poster_url, "
            "added_at = excluded.added_at",
            (user_id, list_name, tmdb_id, title, poster_url, time.time()),
        )
        if list_name == "history":
            c.execute(
                "DELETE FROM movie_lists WHERE user_id = ? AND list_name = 'history' "
                "AND tmdb_id NOT IN ("
                "  SELECT tmdb_id FROM movie_lists "
                "  WHERE user_id = ? AND list_name = 'history' "
                "  ORDER BY added_at DESC LIMIT ?)",
                (user_id, user_id, HISTORY_LIMIT),
            )


def list_remove(user_id: int, list_name: str, tmdb_id: int) -> None:
    with get_conn() as c:
        c.execute(
            "DELETE FROM movie_lists WHERE user_id = ? AND list_name = ? AND tmdb_id = ?",
            (user_id, list_name, tmdb_id),
        )


def list_clear(user_id: int, list_name: str) -> None:
    with get_conn() as c:
        c.execute(
            "DELETE FROM movie_lists WHERE user_id = ? AND list_name = ?",
            (user_id, list_name),
        )


def list_get(user_id: int, list_name: str, limit: int = 100) -> List[Dict[str, Any]]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT tmdb_id, title, poster_url, added_at FROM movie_lists "
            "WHERE user_id = ? AND list_name = ? ORDER BY added_at DESC LIMIT ?",
            (user_id, list_name, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- ratings ----------
def set_rating(
    user_id: int, tmdb_id: int, rating: int, title: str, poster_url: Optional[str]
) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO ratings (user_id, tmdb_id, rating, title, poster_url, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, tmdb_id) DO UPDATE SET "
            "rating = excluded.rating, title = excluded.title, "
            "poster_url = excluded.poster_url, updated_at = excluded.updated_at",
            (user_id, tmdb_id, rating, title, poster_url, time.time()),
        )


def delete_rating(user_id: int, tmdb_id: int) -> None:
    with get_conn() as c:
        c.execute(
            "DELETE FROM ratings WHERE user_id = ? AND tmdb_id = ?", (user_id, tmdb_id)
        )


def get_ratings(user_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT tmdb_id, rating, title, poster_url, updated_at FROM ratings "
            "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_status(user_id: int, tmdb_id: int) -> Dict[str, Any]:
    with get_conn() as c:
        lists = c.execute(
            "SELECT list_name FROM movie_lists "
            "WHERE user_id = ? AND tmdb_id = ? AND list_name IN ('watchlist','favorites')",
            (user_id, tmdb_id),
        ).fetchall()
        r = c.execute(
            "SELECT rating FROM ratings WHERE user_id = ? AND tmdb_id = ?",
            (user_id, tmdb_id),
        ).fetchone()
    names = {row["list_name"] for row in lists}
    return {
        "in_watchlist": "watchlist" in names,
        "is_favorite": "favorites" in names,
        "rating": r["rating"] if r else None,
    }
