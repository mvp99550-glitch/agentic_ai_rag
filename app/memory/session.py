"""
Redis short-term session memory.

Stores the last N messages per tenant:user:session.
Keys expire after 1 hour of inactivity.
"""

import os
import json
import redis
from typing import Optional

SESSION_TTL     = 3600   # 1 hour
MAX_HISTORY     = 20     # keep last 20 messages per session

_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
            protocol=2,
        )
    return _client


def _key(redis_prefix: str) -> str:
    return f"{redis_prefix}:history"


def append_message(redis_prefix: str, role: str, content: str) -> None:
    """Append a message to the session history and reset TTL."""
    r   = _get_client()
    key = _key(redis_prefix)
    r.rpush(key, json.dumps({"role": role, "content": content}))
    r.ltrim(key, -MAX_HISTORY, -1)   # keep only last N messages
    r.expire(key, SESSION_TTL)


def get_history(redis_prefix: str) -> list[dict]:
    """Return the full message history for this session."""
    r   = _get_client()
    raw = r.lrange(_key(redis_prefix), 0, -1)
    return [json.loads(m) for m in raw]


def clear_session(redis_prefix: str) -> None:
    _get_client().delete(_key(redis_prefix))


def ping() -> bool:
    try:
        return _get_client().ping()
    except Exception:
        return False
