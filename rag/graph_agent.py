"""G2: the Calldesk agent as your own LangGraph graph, with a claim-verification step."""
import operator
import sys
import time
from typing import Annotated

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

import agent_scratch
from ask import MODEL
from lc_agent import TOOLS, llm
from request_log import cost_usd

MAX_REVISIONS = 1


class AgentState(MessagesState):          # messages: Annotated[list, add_messages] comes from MessagesState
    revisions: int
    verified: bool | None
    verdicts: Annotated[list, operator.add]
    judge_tokens: Annotated[list, operator.add]   # [(input, output), ...] for the judge calls


class Verdict(BaseModel):
    supported: bool = Field(description="True only if EVERY factual claim is supported by the tool results")
    unsupported_claims: list[str] = Field(description="Each claim in the draft that the tool results do not support")


JUDGE_INSTRUCTIONS = """You are a strict fact-checker for Calldesk answers.
Compare the DRAFT ANSWER with the TOOL RESULTS. List every factual claim in the draft (prices, plans,
features, policies, numbers) that is NOT directly supported by the tool results. Use ONLY the tool
results, not your own knowledge. Arithmetic is supported if its inputs are supported and it is correct.
Greetings, offers of help and "I don't know" are not claims. If there are no unsupported claims, supported=true."""

model_with_tools = llm.bind_tools(TOOLS)
judge = llm.with_structured_output(Verdict, include_raw=True)


# ---------- Nodes ----------
def model(state):
    reply = model_with_tools.invoke([SystemMessage(agent_scratch.AGENT_INSTRUCTIONS)] + state["messages"])
    return {"messages": [reply]}


def verify_claims(state):
    draft = state["messages"][-1].content
    evidence = "\n\n".join(m.content for m in state["messages"] if isinstance(m, ToolMessage)) or "(no tools were called)"
    out = judge.invoke([SystemMessage(JUDGE_INSTRUCTIONS),
                        HumanMessage(f"TOOL RESULTS:\n{evidence}\n\nDRAFT ANSWER:\n{draft}")])
    verdict, usage = out["parsed"], out["raw"].usage_metadata
    return {"verified": verdict.supported, "verdicts": [verdict.model_dump()],
            "judge_tokens": [(usage["input_tokens"], usage["output_tokens"])]}


def request_revision(state):
    claims = state["verdicts"][-1]["unsupported_claims"]
    feedback = (f"A fact-checker found claims in your draft that the tool results do not support: {claims}. "
                "Use the tools to look these up, or remove them. Then give the complete corrected answer.")
    return {"messages": [HumanMessage(feedback)], "revisions": state["revisions"] + 1}


# ---------- Conditional edges ----------
def after_model(state):
    return "tools" if state["messages"][-1].tool_calls else "verify_claims"


def after_verify(state):
    if state["verified"]:
        return END
    return "request_revision" if state["revisions"] < MAX_REVISIONS else END   # out of revisions: END, flagged


graph = StateGraph(AgentState)
graph.add_node("model", model)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("verify_claims", verify_claims)
graph.add_node("request_revision", request_revision)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", after_model, ["tools", "verify_claims"])
graph.add_edge("tools", "model")
graph.add_conditional_edges("verify_claims", after_verify, ["request_revision", END])
graph.add_edge("request_revision", "model")
app = graph.compile()


def run(question, show=True):
    start = time.perf_counter()
    path, final = [], None
    for update in app.stream({"messages": [HumanMessage(question)], "revisions": 0, "verdicts": [], "judge_tokens": []},
                             config={"recursion_limit": 25}, stream_mode=["updates", "values"]):
        kind, data = update
        if kind == "values":
            final = data
            continue
        for node, out in data.items():
            if node == "model":
                msg = out["messages"][-1]
                path.append(f"model[{', '.join(tc['name'] for tc in msg.tool_calls)}]" if msg.tool_calls else "model[draft]")
            elif node == "verify_claims":
                v = out["verdicts"][-1]
                path.append("verify:OK" if v["supported"] else f"verify:REJECT {v['unsupported_claims']}")
            elif node == "tools":
                pass
            else:
                path.append(node)
    seconds = time.perf_counter() - start
    ai = [m for m in final["messages"] if isinstance(m, AIMessage)]
    t_in = sum(m.usage_metadata["input_tokens"] for m in ai) + sum(i for i, _ in final["judge_tokens"])
    t_out = sum(m.usage_metadata["output_tokens"] for m in ai) + sum(o for _, o in final["judge_tokens"])
    stats = {"seconds": seconds, "calls": len(ai) + len(final["judge_tokens"]), "cost": cost_usd(MODEL, t_in, t_out)}
    if show:
        print(f"\n{'=' * 70}\nQ: {question}")
        for p in path:
            print(f"   {p}")
        print(f"A: {final['messages'][-1].content}")
        print(f"   [verified={final['verified']}, revisions={final['revisions']}, {stats['calls']} model calls, "
              f"{seconds:.1f}s, ${stats['cost']:.6f}]")
    return stats


def run_create_agent(question):
    """The A8 prebuilt agent, for comparison (no verification)."""
    start = time.perf_counter()
    result = create_agent(llm, TOOLS, system_prompt=agent_scratch.AGENT_INSTRUCTIONS).invoke(
        {"messages": [{"role": "user", "content": question}]})
    ai = [m for m in result["messages"] if isinstance(m, AIMessage)]
    t_in = sum(m.usage_metadata["input_tokens"] for m in ai)
    t_out = sum(m.usage_metadata["output_tokens"] for m in ai)
    return {"seconds": time.perf_counter() - start, "calls": len(ai), "cost": cost_usd(MODEL, t_in, t_out)}


QUESTIONS = [
    "Does Calldesk make outbound calls, and what is the cheapest plan that includes voicemail transcription "
    "and how much is it per year?",
    "How much is the Growth plan?",
    "If I need 900 extra minutes on the Growth plan, what's my total monthly bill?",
    "Hi there!",
]

if __name__ == "__main__":
    print(app.get_graph().draw_mermaid())
    questions = [" ".join(sys.argv[1:])] if sys.argv[1:] else QUESTIONS
    graph_stats = [run(q) for q in questions]

    print(f"\n{'=' * 70}\nCost of verification: your graph vs A8's create_agent (same questions)")
    print(f"  {'question':<32} {'graph calls':>11} {'agent calls':>11} {'graph s':>8} {'agent s':>8} {'graph $':>10} {'agent $':>10}")
    for q, g in zip(questions, graph_stats):
        a = run_create_agent(q)
        print(f"  {q[:32]:<32} {g['calls']:>11} {a['calls']:>11} {g['seconds']:>8.1f} {a['seconds']:>8.1f} "
              f"{g['cost']:>10.6f} {a['cost']:>10.6f}")
