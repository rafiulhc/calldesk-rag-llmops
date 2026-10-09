import os
import sys
import time

from ingest import EMBEDDING_MODEL, client
from request_log import cost_usd, log_request
from search import search

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = 4
MIN_SCORE = 0.3   # chunks scoring below this are treated as irrelevant

INSTRUCTIONS = """You are Calldesk's support assistant.
Answer using ONLY the context provided.
- If the context answers the question, answer it directly.
- If the context contains a relevant policy or rule, apply it to the user's situation and explain.
- If the context only partly covers the question, say what the docs do say, say what they don't,
  and suggest contacting support@calldesk.app.
- If the context has nothing relevant, reply exactly: "I don't know based on the Calldesk docs."
Never invent facts that are not in the context.
After each fact, cite its source in brackets, like [pricing.txt #6]."""


def build_context(results):
    return "\n\n".join(f"[{c['source']} #{c['chunk_id']}]\n{c['text']}" for _, c in results)


REFUSAL_ANSWER = "I don't know based on the Calldesk docs."


def ask(question):
    start = time.perf_counter()
    record = {"question": question, "model": MODEL, "top_k": TOP_K, "min_score": MIN_SCORE}
    try:
        hits, embed_tokens = search(question, top_k=TOP_K)
        record["retrieval_ms"] = round((time.perf_counter() - start) * 1000)
        results = [r for r in hits if r[0] >= MIN_SCORE]
        record["retrieved"] = [{"source": c["source"], "chunk_id": c["chunk_id"], "score": round(s, 3)}
                               for s, c in results]
        input_tokens = output_tokens = 0

        if not results:
            answer = REFUSAL_ANSWER
        else:
            llm_start = time.perf_counter()
            response = client.responses.create(
                model=MODEL,
                instructions=INSTRUCTIONS,
                input=f"Context:\n{build_context(results)}\n\nQuestion: {question}",
            )
            record["llm_ms"] = round((time.perf_counter() - llm_start) * 1000)
            record["model_version"] = response.model
            answer = response.output_text
            input_tokens, output_tokens = response.usage.input_tokens, response.usage.output_tokens

        llm_cost = cost_usd(MODEL, input_tokens, output_tokens)
        embed_cost = cost_usd(EMBEDDING_MODEL, embed_tokens)
        record.update(answer=answer, embed_tokens=embed_tokens,
                      input_tokens=input_tokens, output_tokens=output_tokens,
                      cost_usd=None if None in (llm_cost, embed_cost) else round(llm_cost + embed_cost, 8))
        return answer, results
    except Exception as e:
        record["error"] = repr(e)
        raise
    finally:
        record["total_ms"] = round((time.perf_counter() - start) * 1000)
        log_request(record)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much do extra minutes cost?"
    answer, results = ask(question)
    print(f"Q: {question}\n\nA: {answer}\n\nRetrieved:")
    for score, c in results:
        print(f"  {score:.3f}  {c['source']} #{c['chunk_id']}")