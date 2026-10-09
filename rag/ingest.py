import json                       # NEW
import os                         # NEW
from pathlib import Path

from dotenv import load_dotenv    # NEW
from openai import OpenAI         # NEW
import numpy as np
from db import get_connection, init_db

load_dotenv()                                                                      # NEW
client = OpenAI()                                                                  # NEW
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")    # NEW

DATA_DIR = Path(__file__).parent / "data"
INDEX_FILE = Path(__file__).parent / "index.json"                                  # NEW
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # characters shared between neighbouring chunks


def load_documents():
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        docs.append({"source": path.name, "text": path.read_text()})
    return docs


def split_fixed(text, size, overlap):
    """Fallback: cut text into fixed-size windows that overlap."""
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split on blank lines so each paragraph becomes one chunk."""
    chunks = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            chunks.append(paragraph)
        else:
            chunks.extend(split_fixed(paragraph, size, overlap))
    return chunks


def embed_texts(texts):                                                            # NEW
    """Turn a list of strings into a list of vectors, in one API call."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    print(f"[embedding tokens] {response.usage.total_tokens}")
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    docs = load_documents()
    chunks = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"])):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": piece})

    print(f"{len(docs)} documents -> {len(chunks)} chunks")

    vectors = embed_texts([c["text"] for c in chunks])                             # NEW
    for chunk, vector in zip(chunks, vectors):                                     # NEW
        chunk["embedding"] = vector                                                # NEW

    init_db()
    with get_connection() as conn, conn.transaction():
        conn.execute("TRUNCATE chunks")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (source, chunk_id, content, embedding) VALUES (%s, %s, %s, %s)",
                [(c["source"], c["chunk_id"], c["text"], np.array(v)) for c, v in zip(chunks, vectors)],
            )
    print(f"Stored {len(chunks)} chunks in Postgres")