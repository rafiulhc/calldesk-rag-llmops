import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]   # no fallback: fail loudly if it's missing

SCHEMA = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS chunks (
        id        bigserial PRIMARY KEY,
        source    text NOT NULL,
        chunk_id  int  NOT NULL,
        content   text NOT NULL,
        embedding vector(1536) NOT NULL,
        UNIQUE (source, chunk_id)
    )""",
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)",
]


def get_connection():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_db():
    with get_connection() as conn:
        for statement in SCHEMA:
            conn.execute(statement)


_pool = None


def get_pool():
    """One shared pool of open connections, created on first use."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=True,
                               kwargs={"autocommit": True}, configure=register_vector)
    return _pool


def close_pool():
    """Close every pooled connection cleanly (called when the API shuts down)."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None