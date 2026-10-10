"""G4: human-in-the-loop - pause before risky actions, resume with approve / edit / reject.

All "write" tools here are SIMULATED: they only record what they would do. Nothing is sent or charged.
"""
import uuid
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

import agent_scratch
from db import DATABASE_URL
from lc_agent import TOOLS as READ_TOOLS
from lc_agent import llm

WORLD = []   # what the simulated write tools "did" in the real world
AUDIT = []   # every human decision: who decided what, and when


# ---------- Simulated write tools ----------
@tool
def book_job(customer_name: str, phone: str, service: str, when: str) -> str:
    """Book a job on the business's job board for a customer."""
    WORLD.append(f"BOOKED {service} for {customer_name} ({phone}) at {when}")
    return f"Booked {service} for {customer_name} at {when}."


@tool
def send_sms(phone: str, message: str) -> str:
    """Send an SMS text message to a customer."""
    WORLD.append(f"SMS to {phone}: {message}")
    return f"SMS sent to {phone}."


@tool
def issue_refund(amount_usd: float, reason: str) -> str:
    """Issue a refund to the customer's card."""
    WORLD.append(f"REFUND ${amount_usd} ({reason})")
    return f"Refunded ${amount_usd}."


WRITE_TOOLS = {t.name: t for t in [book_job, send_sms, issue_refund]}
ALL_TOOLS = {t.name: t for t in READ_TOOLS} | WRITE_TOOLS

INSTRUCTIONS = agent_scratch.AGENT_INSTRUCTIONS + f"""
- You can also book jobs, send SMS confirmations and issue refunds with tools. A staff member approves these.
- If staff reject an action, tell the customer clearly why and what happens next.
- If staff CHANGED an action, tell the customer the changed details, not the original request.
- Today is {datetime.now().strftime('%A %Y-%m-%d')}. Write dates as e.g. 'Tuesday 2026-10-13 10:00'."""

model_with_tools = llm.bind_tools(list(ALL_TOOLS.values()))


# ---------- Nodes ----------
def model(state):
    return {"messages": [model_with_tools.invoke([SystemMessage(INSTRUCTIONS)] + state["messages"])]}


def tools_with_approval(state):
    """Read tools run automatically. Write tools wait for ONE human decision covering all of them.
    Nothing with a side effect happens before interrupt(): this node re-runs from the top on resume."""
    calls = state["messages"][-1].tool_calls
    pending = [tc for tc in calls if tc["name"] in WRITE_TOOLS]
    decisions = interrupt({"actions": [{"id": tc["id"], "tool": tc["name"], "args": tc["args"]} for tc in pending]}) \
        if pending else {}

    results = []
    for tc in calls:
        if tc["name"] not in WRITE_TOOLS:
            output = ALL_TOOLS[tc["name"]].invoke(tc["args"])
        else:
            d = decisions[tc["id"]]
            AUDIT.append({"time": datetime.now(timezone.utc).strftime("%H:%M:%S"), "tool": tc["name"],
                          "args": tc["args"], "decision": d["type"], "by": d.get("by", "staff"),
                          **({"edited_args": d["args"]} if d["type"] == "edit" else {}),
                          **({"reason": d["reason"]} if d["type"] == "reject" else {})})
            if d["type"] == "approve":
                output = WRITE_TOOLS[tc["name"]].invoke(tc["args"])
            elif d["type"] == "edit":
                done = WRITE_TOOLS[tc["name"]].invoke({**tc["args"], **d["args"]})
                output = f"{done} NOTE: staff CHANGED these details before running it: {d['args']}."
            else:
                output = f"REJECTED by staff: {d['reason']}. The action was NOT performed."
        results.append(ToolMessage(output, tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": results}


def after_model(state):
    return "tools" if state["messages"][-1].tool_calls else END


def build_app(checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("model", model)
    graph.add_node("tools", tools_with_approval)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=checkpointer)


def run(app, thread_id, request, decide):
    """Drive the graph: whenever it pauses, show the pending actions, get decisions, resume."""
    cfg = {"configurable": {"thread_id": thread_id}}
    print(f"\n  Customer: {request}")
    result = app.invoke({"messages": [HumanMessage(request)]}, cfg)
    while "__interrupt__" in result:
        actions = result["__interrupt__"][0].value["actions"]
        decisions = {}
        for a in actions:
            print(f"    ⏸  PAUSED for approval: {a['tool']}({a['args']})")
            decisions[a["id"]] = decide(a)
            extra = d.get("args") or d.get("reason") if (d := decisions[a["id"]])["type"] != "approve" else ""
            print(f"    👤 staff decision: {d['type'].upper()} {extra or ''}")
        result = app.invoke(Command(resume=decisions), cfg)
    print(f"  Bot: {result['messages'][-1].content[:300]}")
    return result


def approve(action):
    return {"type": "approve"}


if __name__ == "__main__":
    memory = InMemorySaver()
    app = build_app(memory)
    booking = ("Please book a plumber for Mr Khan (phone +1-555-0100) tomorrow at 10am for a leaking kitchen sink, "
               "and text him a confirmation.")

    print("=== A: approve both actions ===")
    run(app, "thread-A", booking, approve)

    print("\n=== B: edit the time to 11am ===")
    run(app, "thread-B", booking,
        lambda a: {"type": "edit", "args": {"when": "tomorrow at 11am"}} if a["tool"] == "book_job"
        else {"type": "edit", "args": {"message": "Hi Mr Khan, your plumber is booked for tomorrow at 11am. - Calldesk"}})

    print("\n=== C: reject a refund ===")
    reject = lambda a: {"type": "reject", "reason": "refunds over $100 need a manager's approval; a manager will "
                                                    "review this within 1 business day"}
    run(app, "thread-C", "I was charged twice this month. Please refund the duplicate charge of $250.", reject)
    run(app, "thread-C", "The reason is the duplicate charge I mentioned. Please go ahead with the $250 refund.", reject)

    print("\n=== D: read-only question - no pause ===")
    run(app, "thread-D", "How much is the Growth plan?", approve)

    print("\n=== E: pause, server 'restarts', approve later ===")
    t = f"pg-approval-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": t}}
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        result = build_app(saver).invoke({"messages": [HumanMessage(
            "Book an HVAC check for Ms Lopez (phone +1-555-0199) on Friday at 2pm.")]}, cfg)
        print(f"    ⏸  paused, waiting for: {[a['tool'] for a in result['__interrupt__'][0].value['actions']]}")
    print("    --- server restarts (the paused run lives only in Postgres now) ---")
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        app_after = build_app(saver)
        snapshot = app_after.get_state(cfg)
        pending = [a for task in snapshot.tasks for i in task.interrupts for a in i.value["actions"]]
        print(f"    found pending approval in Postgres: {[(a['tool'], a['args']['when']) for a in pending]}")
        result = app_after.invoke(Command(resume={a["id"]: {"type": "approve", "by": "manager"} for a in pending}), cfg)
        print(f"  Bot: {result['messages'][-1].content[:200]}")

    print("\n=== F: the trap - a side effect BEFORE interrupt() runs twice ===")

    class Demo(TypedDict):
        sent: int

    def make_demo(side_effect_first):
        sends = []

        def node(state):
            if side_effect_first:
                sends.append("SMS")                          # BUG: runs on pause AND again on resume
            decision = interrupt("approve SMS?")
            if not side_effect_first and decision == "yes":
                sends.append("SMS")                          # FIX: only after the decision
            return {"sent": len(sends)}

        g = StateGraph(Demo)
        g.add_node("node", node)
        g.add_edge(START, "node")
        g.add_edge("node", END)
        return g.compile(checkpointer=InMemorySaver()), sends

    for label, first in [("side effect BEFORE interrupt", True), ("side effect AFTER interrupt ", False)]:
        demo, sends = make_demo(first)
        cfg = {"configurable": {"thread_id": "demo"}}
        demo.invoke({"sent": 0}, cfg)
        demo.invoke(Command(resume="yes"), cfg)
        print(f"  {label}: SMS sent {len(sends)} time(s)")

    print("\n=== What the simulated tools actually did ===")
    for w in WORLD:
        print(f"  {w}")
    print("\n=== Audit log ===")
    for entry in AUDIT:
        print(f"  {entry}")
