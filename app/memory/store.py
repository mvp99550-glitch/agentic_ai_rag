"""
PostgreSQL long-term store — query logs scoped by tenant + user.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional


def _conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres123"),
        dbname=os.getenv("POSTGRES_DB", "postgres"),
    )


def setup_tables() -> None:
    """Create tables if they don't exist. Run once at app startup."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id          SERIAL PRIMARY KEY,
                    tenant_id   TEXT NOT NULL,
                    user_id     TEXT NOT NULL,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    message     TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id           SERIAL PRIMARY KEY,
                    tenant_id    TEXT NOT NULL,
                    user_id      TEXT NOT NULL,
                    session_id   TEXT NOT NULL,
                    query        TEXT NOT NULL,
                    retrieved    TEXT,
                    answer       TEXT,
                    created_at   TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ql_tenant_user ON query_logs(tenant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ch_session ON conversation_history(session_id);")


def log_query(
    tenant_id:  str,
    user_id:    str,
    session_id: str,
    query:      str,
    retrieved:  str,
    answer:     Optional[str] = None,
) -> None:
    """Log a query and its retrieved chunks. Silently swallows errors."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO query_logs
                       (tenant_id, user_id, session_id, query, retrieved, answer)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (tenant_id, user_id, session_id, query, retrieved, answer),
                )
    except Exception:
        pass


def get_query_history(tenant_id: str, user_id: str, limit: int = 20) -> list[dict]:
    """Return recent queries for a user."""
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT query, answer, created_at FROM query_logs
                       WHERE tenant_id=%s AND user_id=%s
                       ORDER BY created_at DESC LIMIT %s""",
                    (tenant_id, user_id, limit),
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


def ping() -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False
