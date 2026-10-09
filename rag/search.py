import sys

import numpy as np

from db import get_pool
from ingest import embed_texts


def search(question, top_k=3):
    """Return (results, embedding_tokens). Each result is (score, chunk_dict)."""
    vectors, tokens = embed_texts([question])
    query_vector = np.array(vectors[0])
    with get_pool().connection() as conn:
        rows = conn.execute(
            """SELECT source, chunk_id, content, 1 - (embedding <=> %s) AS score
               FROM chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (query_vector, query_vector, top_k),
        ).fetchall()
    results = [(score, {"source": s, "chunk_id": i, "text": t}) for s, i, t, score in rows]
    return results, tokens


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much do extra minutes cost?"
    print(f"Question: {question}\n")
    results, _ = search(question, top_k=8)
    for score, c in results:
        print(f"{score:.3f}  {c['source']} #{c['chunk_id']}  {c['text'][:90]!r}")