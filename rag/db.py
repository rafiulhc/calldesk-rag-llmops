import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

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