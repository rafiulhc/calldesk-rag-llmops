"""G1: LangGraph basics - state, nodes, edges, conditional edges (including a loop)."""
import operator
import sys
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ask import INSTRUCTIONS, MIN_SCORE, MODEL, REFUSAL_ANSWER, TOP_K, build_context
from ingest import client
from search import search

GOOD_SCORE = 0.5     # best chunk must reach this to answer straight away
MAX_REWRITES = 2     # loop limit: never rewrite more than twice


# ============ Part 1: a tiny graph with no AI, to learn the mechanics ============
class CounterState(TypedDict):
    count: int
    log: Annotated[list[str], operator.add]   # a "reducer": node updates are APPENDED, not replaced


def add_one(state):
    return {"count": state["count"] + 1, "log": [f"add_one: {state['count']} -> {state['count'] + 1}"]}


def finish(state):
    return {"log": [f"finish: stopped at {state['count']}"]}


def keep_counting(state):          # a conditional edge: look at the state, choose the next node
    return "add_one" if state["count"] < 3 else "finish"


counter = StateGraph(CounterState)
counter.add_node("add_one", add_one)
counter.add_node("finish", finish)
counter.add_edge(START, "add_one")
counter.add_conditional_edges("add_one", keep_counting, ["add_one", "finish"])   # loops back to itself
counter.add_edge("finish", END)
counter_app = counter.compile()


# ============ Part 2: a self-correcting RAG ============
class RagState(TypedDict):
    question: str                              # the user's original question (never changes)
    query: str                                 # what we search for (may be rewritten)
    chunks: list                               # [(score, chunk), ...] above MIN_SCORE
    attempts: int                              # how many rewrites so far
    answer: str
    path: Annotated[list[str], operator.add]   # which nodes ran, in order


def retrieve(state):
    results, _ = search(state["query"], top_k=TOP_K)
    chunks = [(s, c) for s, c in results if s >= MIN_SCORE]
    best = chunks[0][0] if chunks else 0
    return {"chunks": chunks, "path": [f"retrieve(best={best:.2f})"]}


def rewrite_query(state):
    response = client.responses.create(
        model=MODEL,
        instructions="Rewrite the customer's question as a clear, specific search query about Calldesk "
                     "(an AI phone receptionist: plans, prices, minutes, features, setup, policies). "
                     "Return only the query.",
        input=f"Original question: {state['question']}\nPrevious search query: {state['query']}",
    )
    new_query = response.output_text.strip()
    return {"query": new_query, "attempts": state["attempts"] + 1, "path": [f"rewrite -> {new_query!r}"]}


def generate(state):
    response = client.responses.create(
        model=MODEL, instructions=INSTRUCTIONS,
        input=f"Context:\n{build_context(state['chunks'])}\n\nQuestion: {state['question']}",
    )
    return {"answer": response.output_text, "path": ["generate"]}


def refuse(state):
    return {"answer": REFUSAL_ANSWER, "path": ["refuse"]}


def after_retrieve(state):
    """The decision: answer, retry with a better query, or give up."""
    if not state["chunks"]:
        return "refuse"                        # nothing related at all: off-topic, don't waste rewrites
    if state["chunks"][0][0] >= GOOD_SCORE:
        return "generate"                      # strong match
    if state["attempts"] < MAX_REWRITES:
        return "rewrite_query"                 # weak match: try a better query
    return "generate"                          # out of retries: answer with the best we have


rag = StateGraph(RagState)
for name, fn in [("retrieve", retrieve), ("rewrite_query", rewrite_query), ("generate", generate), ("refuse", refuse)]:
    rag.add_node(name, fn)
rag.add_edge(START, "retrieve")
rag.add_conditional_edges("retrieve", after_retrieve, ["generate", "rewrite_query", "refuse"])
rag.add_edge("rewrite_query", "retrieve")      # the loop back
rag.add_edge("generate", END)
rag.add_edge("refuse", END)
rag_app = rag.compile()


def graph_ask(question):
    """Same inputs/outputs as ask.ask(), so eval.py can test the graph."""
    state = rag_app.invoke({"question": question, "query": question, "attempts": 0, "path": []})
    return state["answer"], state["chunks"] if state["answer"] != REFUSAL_ANSWER else []


if __name__ == "__main__":
    print("=== Part 1: counter graph ===")
    for step in counter_app.stream({"count": 0, "log": []}, stream_mode="updates"):
        print("  ", step)
    print("   final state:", counter_app.invoke({"count": 0, "log": []}))

    print("\n=== Part 2: the self-correcting RAG graph (Mermaid diagram) ===")
    print(rag_app.get_graph().draw_mermaid())

    print("=== Part 3: which path does each question take? ===")
    for q in sys.argv[1:] and [" ".join(sys.argv[1:])] or [
            "How much is the Growth plan?", "minutes thing?", "what about the trial", "What is the capital of France?"]:
        state = rag_app.invoke({"question": q, "query": q, "attempts": 0, "path": []})
        print(f"\n  Q: {q}\n  path: {' -> '.join(state['path'])}\n  A: {state['answer'][:160]}")
