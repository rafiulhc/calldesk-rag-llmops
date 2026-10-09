import os
import sys

from ingest import client
from search import search

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = 4
MIN_SCORE = 0.3   # chunks scoring below this are treated as irrelevant

INSTRUCTIONS = """You are Calldesk's support assistant.
Answer ONLY using the context provided. If the answer is not in the context,
reply exactly: "I don't know based on the Calldesk docs." Never guess.
After each fact, cite its source in brackets, like [pricing.txt #6]."""


def build_context(results):
    return "\n\n".join(f"[{c['source']} #{c['chunk_id']}]\n{c['text']}" for _, c in results)


def ask(question):
    results = [r for r in search(question, top_k=TOP_K) if r[0] >= MIN_SCORE]
    if not results:
        return "I don't know based on the Calldesk docs.", []

    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=f"Context:\n{build_context(results)}\n\nQuestion: {question}",
    )
    return response.output_text, results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much do extra minutes cost?"
    answer, results = ask(question)
    print(f"Q: {question}\n\nA: {answer}\n\nRetrieved:")
    for score, c in results:
        print(f"  {score:.3f}  {c['source']} #{c['chunk_id']}")