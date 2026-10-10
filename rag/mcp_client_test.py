"""M1: talk to the Calldesk MCP server the way any MCP client does: start it, discover, call."""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).parent / "mcp_server.py")],
                               cwd=str(Path(__file__).parent.parent))


def text_of(result):
    return " ".join(getattr(c, "text", "") for c in result.content)


async def main():
    async with stdio_client(SERVER) as (read, write), ClientSession(read, write) as session:
        info = await session.initialize()
        print(f"Connected to MCP server: {info.server_info.name}")

        print("\n=== Discovery: what does the server offer? ===")
        for t in (await session.list_tools()).tools:
            params = list(t.input_schema.get("properties", {}))
            print(f"  tool      {t.name}({', '.join(params)})  read-only={t.annotations.read_only_hint}")
        for r in (await session.list_resources()).resources:
            print(f"  resource  {r.uri}  ({r.title})")
        for p in (await session.list_prompts()).prompts:
            print(f"  prompt    {p.name}({', '.join(a.name for a in p.arguments or [])})")

        print("\n=== Calling tools ===")
        calls = [("search_docs", {"query": "Does Calldesk make outbound calls?"}),
                 ("get_plan", {"name": "growth"}),
                 ("calculate_bill", {"plan": "growth", "extra_minutes": 900}),
                 ("calculate_bill", {"plan": "growth", "billing_period": "annual"}),
                 ("calculate", {"expression": "__import__('os').system('ls')"}),
                 ("calculate_bill", {"plan": "platinum"})]
        for name, args in calls:
            result = await session.call_tool(name, args)
            flag = "ERROR " if result.is_error else ""
            print(f"  {name}({args})\n    -> {flag}{text_of(result)[:160]}")

        bill = json.loads(text_of(await session.call_tool("calculate_bill", {"plan": "growth", "extra_minutes": 900})))
        assert bill["total_usd_per_month"] == 1199, bill
        print("\n  ✅ calculate_bill(growth, 900) == $1,199 (the agent once said $999)")

        print("\n=== Reading the resource and the prompt ===")
        resource = await session.read_resource("calldesk://plans")
        print(f"  calldesk://plans -> {resource.contents[0].text[:80]!r}...")
        prompt = await session.get_prompt("answer_customer_question", {"question": "Is Calldesk good for plumbers?"})
        print(f"  prompt -> {prompt.messages[0].content.text[:100]!r}...")


asyncio.run(main())
