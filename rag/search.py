import sys

import numpy as np

from db import get_connection
from ingest import embed_texts


def search(question, top_k=3):
    query_vector = np.array(embed_texts([question])[0])
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT source, chunk_id, content, 1 - (embedding <=> %s) AS score
               FROM chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (query_vector, query_vector, top_k),
        ).fetchall()
    return [(score, {"source": s, "chunk_id": i, "text": t}) for s, i, t, score in rows]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much do extra minutes cost?"
    print(f"Question: {question}\n")
    for score, c in search(question):
        print(f"{score:.3f}  {c['source']} #{c['chunk_id']}  {c['text'][:90]!r}")