import json
import math
import sys

from ingest import INDEX_FILE, embed_texts


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def search(question, top_k=3):
    chunks = json.loads(INDEX_FILE.read_text())
    query_vector = embed_texts([question])[0]

    scored = [(cosine_similarity(query_vector, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much do extra minutes cost?"
    print(f"Question: {question}\n")
    for score, c in search(question):
        print(f"{score:.3f}  {c['source']} #{c['chunk_id']}  {c['text'][:90]!r}")