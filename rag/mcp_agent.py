"""M2 part 1: a LangGraph agent whose tools come from the Calldesk MCP server, not Python imports.

langchain-mcp-adapters 0.3.1 doesn't work with mcp 2.x, so this file includes a small bridge
that does what the adapter does: list the server's tools and wrap each as a LangChain tool.
"""
import asyncio
import json
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lc_agent import llm

SERVER = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).parent / "mcp_server.py")],
                               cwd=str(Path(__file__).parent.parent))


async def load_mcp_tools(session):
    """The bridge: every MCP tool becomes a LangChain tool that forwards calls to the server."""
    tools = []
    for t in (await session.list_tools()).tools:
        async def call(_tool_name=t.name, **arguments):
            result = await session.call_tool(_tool_name, arguments)
            text = " ".join(getattr(c, "text", "") for c in result.content)
            return f"ERROR: {text}" if result.is_error else text
        tools.append(StructuredTool.from_function(coroutine=call, name=t.name, description=t.description,
                                                  args_schema=t.input_schema))   # the server's own JSON schema
    return tools


QUESTIONS = [
    # the exact wording that made the G3 agent answer $999 three times out of three
    "On the Growth plan with 900 extra minutes, what is my total monthly bill? Look up the plan first, then calculate.",
    "How much is the Growth plan per year if I pay annually?",
    "Does Calldesk make outbound calls, and which plan includes voicemail transcription?",
]


async def main():
    async with stdio_client(SERVER) as (read, write), ClientSession(read, write) as session:
        info = await session.initialize()
        tools = await load_mcp_tools(session)
        print(f"Loaded {len(tools)} tools from MCP server '{info.server_info.name}': {[t.name for t in tools]}")
        agent = create_agent(llm, tools, system_prompt=(
            f"{info.instructions}\nOnly state facts your tools return. Cite search_docs sources. Be brief."))

        for q in QUESTIONS:
            for run in range(3 if q == QUESTIONS[0] else 1):
                result = await agent.ainvoke({"messages": [HumanMessage(q)]})
                calls = [f"{tc['name']}({json.dumps(tc['args'])})" for m in result["messages"]
                         if isinstance(m, AIMessage) for tc in m.tool_calls]
                print(f"\nQ: {q}" + (f"  [run {run + 1}]" if q == QUESTIONS[0] else ""))
                print(f"   tools: {calls}")
                print(f"   A: {result['messages'][-1].content[:250]}")


asyncio.run(main())
