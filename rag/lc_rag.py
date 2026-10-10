import os
import sys
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

from ask import INSTRUCTIONS, MIN_SCORE, MODEL, REFUSAL_ANSWER, TOP_K
from ingest import EMBEDDING_MODEL
from request_log import cost_usd, log_request

PG_URL = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg://", 1)

# The collection was filled in A3 (run rag/lc_vectorstores.py first)
store = PGVector(embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL), collection_name="calldesk_lc",
                 connection=PG_URL, use_jsonb=True)
llm = ChatOpenAI(model=MODEL, timeout=20, max_retries=3)
prompt = ChatPromptTemplate.from_messages([
    ("system", INSTRUCTIONS),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def retrieve(question):
    """Search with scores (as_retriever() would drop them), convert distance -> similarity, apply MIN_SCORE."""
    hits = store.similarity_search_with_score(question, k=TOP_K)
    return [(1 - distance, doc) for doc, distance in hits if 1 - distance >= MIN_SCORE]


def format_context(results):
    return "\n\n".join(f"[{d.metadata['source']} #{d.metadata['chunk_id']}]\n{d.page_content}" for _, d in results)


rag_chain = (
    RunnableParallel(results=RunnableLambda(retrieve), question=RunnablePassthrough())
    | RunnablePassthrough.assign(context=lambda x: format_context(x["results"]))
    | RunnableBranch(
        (lambda x: not x["results"], RunnablePassthrough.assign(message=lambda x: None)),   # nothing relevant: skip GPT
        RunnablePassthrough.assign(message=prompt | llm),                                    # keep the AIMessage (usage!)
    )
)


def lc_ask(question):
    """Same inputs and outputs as ask.ask(), so eval.py can test either one."""
    start = time.perf_counter()
    out = rag_chain.invoke(question)
    message = out["message"]
    answer = message.content if message else REFUSAL_ANSWER
    usage = message.usage_metadata if message else {"input_tokens": 0, "output_tokens": 0}
    results = [(score, {"source": d.metadata["source"], "chunk_id": d.metadata["chunk_id"], "text": d.page_content})
               for score, d in out["results"]]
    log_request({
        "impl": "langchain", "question": question, "model": MODEL, "answer": answer,
        "retrieved": [{"source": c["source"], "chunk_id": c["chunk_id"], "score": round(s, 3)} for s, c in results],
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "cost_usd": cost_usd(MODEL, usage["input_tokens"], usage["output_tokens"]),
        "total_ms": round((time.perf_counter() - start) * 1000),
    })
    return answer, results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much do extra minutes cost?"
    answer, results = lc_ask(question)
    print(f"Q: {question}\n\nA: {answer}\n\nRetrieved:")
    for score, c in results:
        print(f"  {score:.3f}  {c['source']} #{c['chunk_id']}")
