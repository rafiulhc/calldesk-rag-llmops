"""M1: Calldesk as an MCP server (stdio transport). READ-ONLY tools only.

Run by an MCP client (Claude Desktop, rag/mcp_client_test.py, a LangGraph agent), not by hand.
Never print() to stdout in a stdio server: stdout carries the protocol messages.
"""
import json
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import agent_scratch
import billing

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

server = MCPServer(
    name="calldesk",
    title="Calldesk knowledge base",
    instructions="Tools for answering questions about Calldesk, an AI phone receptionist for home-service "
                 "businesses. Use calculate_bill for any price or bill; never do pricing arithmetic yourself.",
)


@server.tool(annotations=READ_ONLY)
def search_docs(query: str) -> str:
    """Search Calldesk's documentation (features, policies, support, setup, refunds, billing).
    Returns relevant passages with [source #chunk] labels to cite."""
    return agent_scratch.search_docs(query)


@server.tool(annotations=READ_ONLY)
def get_plan(name: Literal["starter", "growth", "scale"]) -> str:
    """Get exact data for one Calldesk plan: monthly price, included minutes and phone numbers.
    For features, use search_docs."""
    return json.dumps({"plan": name, **billing.PLANS[name], "overage": "300 extra minutes for 100 USD"})


@server.tool(annotations=READ_ONLY)
def calculate_bill(plan: Literal["starter", "growth", "scale"], extra_minutes: int = 0,
                   billing_period: Literal["monthly", "annual"] = "monthly") -> str:
    """Calculate a Calldesk bill exactly, in code: plan price plus extra-minute top-ups
    (300 minutes per 100 USD, partial top-ups round up), optionally with annual billing (20% off).
    Use this for ANY bill, total or yearly price."""
    try:
        return json.dumps(billing.calculate_bill(plan, extra_minutes, billing_period))
    except ValueError as e:
        raise ToolError(str(e)) from e     # a clear message the model can act on


@server.tool(annotations=READ_ONLY)
def calculate(expression: str) -> str:
    """Evaluate a plain arithmetic expression (numbers and + - * / ( )). Not for bills: use calculate_bill."""
    try:
        return agent_scratch.calculate(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        raise ToolError(f"invalid expression: {e}") from e


@server.resource("calldesk://plans", name="plans", title="Calldesk plans",
                 description="All Calldesk plans with price, minutes and phone numbers", mime_type="application/json")
def plans_resource() -> str:
    return json.dumps(billing.PLANS, indent=2)


@server.prompt(title="Answer a Calldesk customer question")
def answer_customer_question(question: str) -> str:
    """A ready-made prompt for answering a customer question with the Calldesk tools."""
    return (f"A Calldesk customer asks: {question}\n\nUse the calldesk tools to look up every fact before stating it. "
            "Use calculate_bill for any price. Cite sources from search_docs. If the tools don't have the answer, "
            "say you don't know.")


if __name__ == "__main__":
    server.run("stdio")
