"""
db.py
-----
Handles all MongoDB connectivity and persistence for BiblioBot.

Responsibilities:
- Maintain a single cached MongoDB client/database handle
- Log every chat turn (user question + assistant response)
- Log every database mutation (add/remove book) as a separate audit trail
- Provide read-back of chat history for a given session

All functions fail "soft": if MONGO_URI is missing or the connection
drops, the app keeps working (CSV + vector store still function), it
just stops persisting history and prints a warning to the console.
"""

import os
import uuid
from datetime import datetime, timezone

import certifi
from pymongo import MongoClient
from pymongo.server_api import ServerApi

_client = None
_db = None
_connection_error = None


def get_db():
    """Return a cached MongoDB database handle, or None if unavailable."""
    global _client, _db, _connection_error

    if _db is not None:
        return _db
    if _connection_error is not None:
        # Already tried and failed this session; don't hammer the server.
        return None

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        _connection_error = "MONGO_URI not set in environment"
        return None

    try:
        _client = MongoClient(
            mongo_uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=5000,
            tlsCAFile=certifi.where(),  # <-- fixes CERTIFICATE_VERIFY_FAILED /
                                        #     TLS handshake errors caused by a
                                        #     missing/stale local CA bundle
        )
        _client.admin.command("ping")
        _db = _client["bibliobot"]
        return _db
    except Exception as e:
        _connection_error = str(e)
        print(f"[MongoDB] Connection failed: {e}")
        return None


def is_connected():
    """Cheap status check for the UI (does not force a reconnect attempt)."""
    return _db is not None


def connection_error():
    return _connection_error


def new_session_id():
    return str(uuid.uuid4())


def log_message(session_id, role, content, extra=None):
    """Persist a single chat message (user or assistant) to MongoDB."""
    db = get_db()
    if db is None:
        return

    doc = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc),
    }
    if extra:
        doc.update(extra)

    try:
        db["chat_logs"].insert_one(doc)
    except Exception as e:
        print(f"[MongoDB] Failed to log message: {e}")


def log_db_mutation(session_id, action, target_db, book_title, result_message):
    """Persist add/remove book actions as a separate audit trail."""
    db = get_db()
    if db is None:
        return

    doc = {
        "session_id": session_id,
        "action": action,
        "target_db": target_db,
        "book_title": book_title,
        "result": result_message,
        "timestamp": datetime.now(timezone.utc),
    }

    try:
        db["mutation_logs"].insert_one(doc)
    except Exception as e:
        print(f"[MongoDB] Failed to log mutation: {e}")


def get_chat_history(session_id, limit=100):
    """Fetch chat history for a given session, oldest first."""
    db = get_db()
    if db is None:
        return []

    try:
        cursor = (
            db["chat_logs"]
            .find({"session_id": session_id})
            .sort("timestamp", 1)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        print(f"[MongoDB] Failed to fetch history: {e}")
        return []


def list_recent_sessions(limit=20):
    """Return the most recent distinct session ids, most recent first."""
    db = get_db()
    if db is None:
        return []

    try:
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {"_id": "$session_id", "last_ts": {"$first": "$timestamp"}}},
            {"$sort": {"last_ts": -1}},
            {"$limit": limit},
        ]
        return list(db["chat_logs"].aggregate(pipeline))
    except Exception as e:
        print(f"[MongoDB] Failed to list sessions: {e}")
        return []


def delete_session(session_id):
    """Deletes all logs associated with a session ID from MongoDB."""
    db = get_db()
    if db is None:
        return False

    try:
        db["chat_logs"].delete_many({"session_id": session_id})
        db["mutation_logs"].delete_many({"session_id": session_id})
        return True
    except Exception as e:
        print(f"[MongoDB] Failed to delete session: {e}")
        return False