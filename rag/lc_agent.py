"""A8: the A7 agent rebuilt with LangChain (@tool, bind_tools, create_agent)."""
import json
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import agent_scratch as scratch   # reuse the A7 tool functions, prompt and questions
from ask import MODEL
from request_log import cost_usd

MAX_STEPS = 6


# Part 1: @tool - the docstring becomes the description, the type hints become the JSON schema
@tool
def search_docs(query: str) -> str:
    """Search Calldesk's documentation (features, policies, support, setup, refunds, billing).
    Returns relevant passages with [source #chunk] labels."""
    return scratch.search_docs(query)


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression with numbers and + - * / ( ). Example: '899 + 3 * 100'."""
    return scratch.run_tool("calculate", json.dumps({"expression": expression}))   # safe calculator + error handling


@tool
def get_plan(name: Literal["starter", "growth", "scale"]) -> str:
    """Get exact data for one Calldesk plan: monthly price, included minutes, phone numbers, overage price."""
    return scratch.get_plan(name)


TOOLS = [search_docs, calculate, get_plan]
llm = ChatOpenAI(model=MODEL, timeout=20, max_retries=2)


def show_trace(question, messages):
    model_calls = tool_calls = tokens_in = tokens_out = 0
    print(f"\n{'=' * 70}\nQ: {question}")
    for m in messages:
        if isinstance(m, AIMessage):
            model_calls += 1
            tokens_in += m.usage_metadata["input_tokens"]
            tokens_out += m.usage_metadata["output_tokens"]
            for tc in m.tool_calls:
                tool_calls += 1
                print(f"   call: {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            print(f"     -> {m.content[:90]!r}{'...' if len(m.content) > 90 else ''}")
    print(f"A: {messages[-1].content}")
    print(f"   [{model_calls} model calls, {tool_calls} tool calls, {tokens_in} in / {tokens_out} out tokens, "
          f"${cost_usd(MODEL, tokens_in, tokens_out):.6f}]")


if __name__ == "__main__":
    print("=== Part 1: what @tool generated ===")
    for t in TOOLS:
        print(f"  {t.name}: args={t.args}")
        print(f"    description: {t.description.splitlines()[0]!r}")

    print("\n=== Part 2: bind_tools - the model ASKS for a tool, nothing runs ===")
    request = llm.bind_tools(TOOLS).invoke("How much is the Growth plan?")
    print(f"  content: {request.content!r}")
    print(f"  tool_calls: {request.tool_calls}")

    print("\n=== Part 3: create_agent - LangChain's ready-made loop ===")
    agent = create_agent(llm, TOOLS, system_prompt=scratch.AGENT_INSTRUCTIONS)
    questions = scratch.QUESTIONS + [
        "Does Calldesk make outbound calls, and what is the cheapest plan that includes voicemail "
        "transcription and how much is it per year?"]
    for q in questions:
        result = agent.invoke({"messages": [{"role": "user", "content": q}]},
                              config={"recursion_limit": 2 * MAX_STEPS + 1})
        show_trace(q, result["messages"])

    print(f"\n{'=' * 70}\n=== Part 4: streaming the agent's steps (it's a LangGraph graph inside) ===")
    for update in agent.stream({"messages": [{"role": "user", "content": "How much is the Scale plan?"}]},
                               stream_mode="updates"):
        for node, data in update.items():
            last = data["messages"][-1]
            what = f"tool_calls={[tc['name'] for tc in last.tool_calls]}" if getattr(last, "tool_calls", None) \
                else repr(last.content[:60])
            print(f"  node '{node}': {type(last).__name__} {what}")
