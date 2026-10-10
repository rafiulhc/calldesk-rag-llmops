"""G5: multiple agents - a supervisor routing to pricing / support / booking specialists,
compared head-to-head with a single agent that has all the tools.

Write tools are the SIMULATED ones from G4: nothing is really booked or sent.
"""
import operator
import time
from datetime import datetime
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from ask import MODEL
from graph_approval import WORLD, book_job, send_sms
from lc_agent import calculate, get_plan, llm, search_docs
from request_log import cost_usd

MAX_HANDOFFS = 4
TODAY = f"Today is {datetime.now().strftime('%A %Y-%m-%d')}. Write dates as e.g. 'Monday 2026-10-12 15:00'."
COMMON = "You are part of Calldesk's assistant team. Only state facts your tools return. Be brief."

SPECIALISTS = {
    "pricing": create_agent(llm, [get_plan, calculate], system_prompt=COMMON +
                            " You handle plan prices, minutes and bills. Use calculate for all arithmetic."),
    "support": create_agent(llm, [search_docs], system_prompt=COMMON +
                            " You handle features, support levels, policies and setup. Cite sources."),
    "booking": create_agent(llm, [book_job, send_sms], system_prompt=COMMON + " " + TODAY +
                            " You book jobs and demos and send SMS confirmations. Never discuss prices."),
}


class TeamState(MessagesState):
    route: Annotated[list, operator.add]
    handoffs: int
    usage: Annotated[list, operator.add]       # (input_tokens, output_tokens) of every model call


class Route(BaseModel):
    next: Literal["pricing", "support", "booking", "FINISH"]
    reason: str = Field(description="One short sentence")


SUPERVISOR = """You coordinate Calldesk's specialists:
- pricing: plan prices, minutes, bills and costs
- support: features, support levels, policies, setup, countries
- booking: booking jobs or demos, sending SMS confirmations
Look at the customer's request and at what specialists have ALREADY answered (messages named after them).
Pick the next specialist for any part of the request that is still unhandled. Never pick the same specialist
twice for the same part. If every part is handled, or it is just small talk, pick FINISH."""

router = llm.with_structured_output(Route, include_raw=True)


def usage_of(messages):
    return [(m.usage_metadata["input_tokens"], m.usage_metadata["output_tokens"])
            for m in messages if isinstance(m, AIMessage) and m.usage_metadata]


# ---------- Nodes ----------
def supervisor(state):
    if state["handoffs"] >= MAX_HANDOFFS:
        return {"route": ["FINISH (handoff limit)"]}
    out = router.invoke([SystemMessage(SUPERVISOR)] + state["messages"])
    choice = out["parsed"].next
    return {"route": [choice], "usage": usage_of([out["raw"]])}


def make_specialist_node(name):
    agent = SPECIALISTS[name]

    def node(state):
        result = agent.invoke({"messages": state["messages"]})       # the specialist sees the whole conversation
        reply = result["messages"][-1].content
        return {"messages": [AIMessage(reply, name=name)], "handoffs": state["handoffs"] + 1,
                "usage": usage_of(result["messages"][len(state["messages"]):])}
    return node


def finalize(state):
    reports = [m for m in state["messages"] if isinstance(m, AIMessage) and m.name in SPECIALISTS]
    if len(reports) == 1:                                            # one specialist: its answer IS the answer
        return {"messages": [AIMessage(reports[0].content, name="final")]}
    reply = llm.invoke([SystemMessage("Combine the specialists' answers into ONE short, friendly reply to the "
                                      "customer. Keep every fact, date, price and source exactly as given. "
                                      "If there are no specialist answers, just reply to the customer.")]
                       + state["messages"])
    return {"messages": [AIMessage(reply.content, name="final")], "usage": usage_of([reply])}


def after_supervisor(state):
    last = state["route"][-1]
    return "finalize" if last.startswith("FINISH") else last


graph = StateGraph(TeamState)
graph.add_node("supervisor", supervisor)
for name in SPECIALISTS:
    graph.add_node(name, make_specialist_node(name))
    graph.add_edge(name, "supervisor")                              # every specialist reports back
graph.add_node("finalize", finalize)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", after_supervisor, [*SPECIALISTS, "finalize"])
graph.add_edge("finalize", END)
team = graph.compile()

single_agent = create_agent(llm, [get_plan, calculate, search_docs, book_job, send_sms], system_prompt=(
    "You are Calldesk's assistant. Only state facts your tools return. Be brief. Use calculate for all arithmetic. "
    "Cite sources from search results. You can book jobs and demos and send SMS confirmations. " + TODAY))


# ---------- Test requests with automatic checks ----------
def has(*options):
    return lambda answer, booked: any(o.lower() in answer.lower() for o in options)


REQUESTS = [
    ("How much is the Growth plan?", [has("899")]),
    ("Does Calldesk make outbound calls?", [has("not", "no outbound", "only inbound", "inbound only")]),
    ("Book a plumber for Mr Khan (phone +1-555-0100) tomorrow at 10am for a leaking sink.",
     [lambda a, booked: any("Khan" in b and "2026-10-11" in b for b in booked)]),
    ("What does the Scale plan cost, does it include 24/7 support, and please book a demo for me "
     "(Sam Lee, +1-555-0123) on Monday at 3pm.",
     [has("1,299", "1299"), has("24/7"), lambda a, booked: any("Lee" in b and "2026-10-12" in b for b in booked)]),
    ("Hi there!", [lambda a, booked: not booked]),
]


def run_team(request):
    before = len(WORLD)
    start = time.perf_counter()
    s = team.invoke({"messages": [HumanMessage(request)], "route": [], "handoffs": 0, "usage": []},
                    config={"recursion_limit": 30})
    return s["messages"][-1].content, WORLD[before:], s["route"], s["usage"], time.perf_counter() - start


def run_single(request):
    before = len(WORLD)
    start = time.perf_counter()
    s = single_agent.invoke({"messages": [HumanMessage(request)]})
    route = [tc["name"] for m in s["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls]
    return s["messages"][-1].content, WORLD[before:], route, usage_of(s["messages"]), time.perf_counter() - start


if __name__ == "__main__":
    print(team.get_graph().draw_mermaid())
    totals = {"team": [0, 0, 0.0, 0.0], "single": [0, 0, 0.0, 0.0]}   # correct, calls, seconds, cost
    for request, checks in REQUESTS:
        print(f"\n{'=' * 78}\nQ: {request}")
        for label, fn in [("team", run_team), ("single", run_single)]:
            answer, booked, route, usage, seconds = fn(request)
            ok = all(check(answer, booked) for check in checks)
            cost = cost_usd(MODEL, sum(i for i, _ in usage), sum(o for _, o in usage))
            t = totals[label]
            t[0] += ok
            t[1] += len(usage)
            t[2] += seconds
            t[3] += cost
            print(f"  [{label:<6}] {'✅' if ok else '❌'} route: {' -> '.join(route) or '(no tools)'}")
            print(f"           {len(usage)} model calls, {seconds:.1f}s, ${cost:.6f}")
            print(f"           A: {answer[:230]}")
            for b in booked:
                print(f"           world: {b}")

    print(f"\n{'=' * 78}\nTOTAL over {len(REQUESTS)} requests")
    for label, (correct, calls, seconds, cost) in totals.items():
        print(f"  {label:<6}: {correct}/{len(REQUESTS)} correct, {calls} model calls, {seconds:.1f}s, ${cost:.6f}")
