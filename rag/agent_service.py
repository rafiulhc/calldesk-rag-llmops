"""The production agent behind POST /agent.

A LangGraph tool loop (model <-> tools) with read-only tools, code-based billing,
conversation memory in Postgres checkpoints (one thread per conversation), history trimming,
today's date in the prompt, a step limit, and the same JSON request logging as /ask.
"""
import json
import time
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

import billing
from ask import MIN_SCORE, MODEL
from db import DATABASE_URL
from request_log import cost_usd, log_request
from search import search

MAX_HISTORY_MESSAGES = 12      # trim long conversations before each model call
RECURSION_LIMIT = 12           # max graph steps per message: about 5 rounds of tool calls
GAVE_UP_ANSWER = "Sorry, I couldn't finish that request. Please try rephrasing it."


# ---------- Read-only tools ----------
@tool
def search_docs(query: str) -> str:
    """Search Calldesk's documentation (features, support levels, policies, setup, refunds, countries).
    Write a full, specific question. Returns passages labelled [source #chunk] to cite."""
    results, _ = search(query, top_k=4)
    hits = [(s, c) for s, c in results if s >= MIN_SCORE]
    if not hits:
        return "No relevant documentation found. Try a more specific query."
    return "\n\n".join(f"[{c['source']} #{c['chunk_id']}] {c['text']}" for _, c in hits)


@tool
def get_plan(name: Literal["starter", "growth", "scale"]) -> str:
    """Get one Calldesk plan's monthly price, included minutes and phone numbers. For features use search_docs."""
    return json.dumps({"plan": name, **billing.PLANS[name]})


@tool
def calculate_bill(plan: Literal["starter", "growth", "scale"], extra_minutes: int = 0,
                   billing_period: Literal["monthly", "annual"] = "monthly") -> str:
    """Calculate a Calldesk bill exactly: plan price + extra-minute top-ups (300 minutes per 100 USD,
    partial top-ups round up), optionally annual billing (20% off). Use for ANY price, total or yearly cost."""
    try:
        return json.dumps(billing.calculate_bill(plan, extra_minutes, billing_period))
    except ValueError as e:
        return f"ERROR: {e}"


TOOLS = [search_docs, get_plan, calculate_bill]


def instructions():
    return f"""You are Calldesk's assistant. Calldesk is an AI phone receptionist for home-service businesses.
- Look up every fact with a tool before stating it. Never answer Calldesk questions from memory.
- Answer every part of the customer's question.
- Use calculate_bill for any price, bill or yearly cost. Never do pricing arithmetic yourself.
- Cite search_docs results like [pricing.txt #5].
- If the tools don't contain the answer, say you don't know based on the Calldesk docs.
- For greetings or small talk, reply briefly without tools.
- Today is {datetime.now().strftime('%A %Y-%m-%d')}."""


# ---------- The graph ----------
def build_graph(checkpointer):
    llm = ChatOpenAI(model=MODEL, timeout=20, max_retries=3).bind_tools(TOOLS)

    def model(state):
        history = trim_messages(state["messages"], max_tokens=MAX_HISTORY_MESSAGES, token_counter=len,
                                strategy="last", start_on="human")
        return {"messages": [llm.invoke([SystemMessage(instructions()), *history])]}

    def after_model(state):
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=checkpointer)


_pool = None
_app = None


def init_agent():
    """Called once at API startup: the checkpointer needs its own pool (dict rows, no prepared statements)."""
    global _pool, _app
    _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=True,
                           kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    saver = PostgresSaver(_pool)
    saver.setup()                       # creates the checkpoint tables if they don't exist yet
    _app = build_graph(saver)


def close_agent():
    global _pool, _app
    if _pool is not None:
        _pool.close()
    _pool, _app = None, None


def run_agent(message, thread_id):
    """Run one customer message in a conversation thread. Returns (answer, tool_calls)."""
    start = time.perf_counter()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT,
              "run_name": "calldesk-agent", "metadata": {"model": MODEL}}
    record = {"impl": "agent", "thread_id": thread_id, "question": message, "model": MODEL}
    try:
        before = len(_app.get_state(config).values.get("messages", []))
        try:
            state = _app.invoke({"messages": [HumanMessage(message)]}, config)
            new = state["messages"][before:]
            answer = state["messages"][-1].content
        except GraphRecursionError:
            new, answer = [], GAVE_UP_ANSWER
            record["error"] = "step limit reached"
        ai = [m for m in new if isinstance(m, AIMessage)]
        tool_calls = [{"name": tc["name"], "args": tc["args"]} for m in ai for tc in m.tool_calls]
        tokens_in = sum(m.usage_metadata["input_tokens"] for m in ai if m.usage_metadata)
        tokens_out = sum(m.usage_metadata["output_tokens"] for m in ai if m.usage_metadata)
        record.update(answer=answer, tool_calls=tool_calls, model_calls=len(ai), input_tokens=tokens_in,
                      output_tokens=tokens_out, cost_usd=cost_usd(MODEL, tokens_in, tokens_out))
        return answer, tool_calls
    except Exception as e:
        record["error"] = repr(e)
        raise
    finally:
        record["total_ms"] = round((time.perf_counter() - start) * 1000)
        log_request(record)
