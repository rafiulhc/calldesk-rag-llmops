"""A7: an agent written from scratch - the raw OpenAI tool-calling loop, no framework."""
import ast
import json
import operator
import sys

from ask import MIN_SCORE, MODEL
from ingest import client
from request_log import cost_usd
from search import search

MAX_STEPS = 6

AGENT_INSTRUCTIONS = """You are Calldesk's assistant.
- Use the tools to look up facts. Never answer questions about Calldesk from memory.
- Use get_plan for exact plan prices, minutes and phone numbers. Use search_docs for everything else.
- Use calculate for ANY arithmetic, even simple sums.
- Cite search results like [pricing.txt #5].
- If the tools don't contain the answer, say you don't know based on the Calldesk docs.
- For greetings or small talk, reply briefly without using tools."""


# ---------- Tool 1: search the docs (your production pgvector search) ----------
def search_docs(query):
    results, _ = search(query, top_k=4)
    hits = [(s, c) for s, c in results if s >= MIN_SCORE]
    if not hits:
        return "No relevant documentation found."
    return "\n\n".join(f"[{c['source']} #{c['chunk_id']}] {c['text']}" for _, c in hits)


# ---------- Tool 2: a SAFE calculator (never eval() model output) ----------
ALLOWED = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.USub: operator.neg}


def calculate(expression):
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED:
            return ALLOWED[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED:
            return ALLOWED[type(node.op)](walk(node.operand))
        raise ValueError(f"not allowed: {type(node).__name__}. Only numbers and + - * / ( ) are allowed")
    return str(walk(ast.parse(expression, mode="eval").body))


# ---------- Tool 3: exact plan data (structured, not searched) ----------
PLANS = {
    "starter": {"price_usd_per_month": 599, "minutes": 600, "phone_numbers": 1},
    "growth": {"price_usd_per_month": 899, "minutes": 1200, "phone_numbers": 2},
    "scale": {"price_usd_per_month": 1299, "minutes": 2400, "phone_numbers": 3},
}


def get_plan(name):
    plan = PLANS.get(name.strip().lower())
    if plan is None:
        return f"Unknown plan {name!r}. Valid plans: {', '.join(PLANS)}"
    return json.dumps({"plan": name.lower(), **plan, "overage": "300 extra minutes for 100 USD"})


# ---------- Tool definitions: what the MODEL sees (name + description + JSON schema) ----------
TOOLS = [
    {"type": "function", "name": "search_docs",
     "description": "Search Calldesk's documentation (features, policies, support, setup, refunds, billing). "
                    "Returns relevant passages with [source #chunk] labels.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A full search question"}},
                    "required": ["query"], "additionalProperties": False}},
    {"type": "function", "name": "calculate",
     "description": "Evaluate an arithmetic expression with numbers and + - * / ( ). Example: '899 + 3 * 100'.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                    "required": ["expression"], "additionalProperties": False}},
    {"type": "function", "name": "get_plan",
     "description": "Get exact data for one Calldesk plan: monthly price, included minutes, phone numbers, overage price.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": ["starter", "growth", "scale"]}},
                    "required": ["name"], "additionalProperties": False}},
]
FUNCTIONS = {"search_docs": search_docs, "calculate": calculate, "get_plan": get_plan}


def run_tool(name, arguments_json):
    """Run one tool. Errors go back to the model as text instead of crashing the agent."""
    try:
        args = json.loads(arguments_json)
        return FUNCTIONS[name](**args)
    except Exception as e:
        return f"ERROR: {e}"


# ---------- The agent loop ----------
def run_agent(question):
    print(f"\n{'=' * 70}\nQ: {question}")
    items = [{"role": "user", "content": question}]
    tokens_in = tokens_out = tool_calls = 0

    for step in range(1, MAX_STEPS + 1):
        response = client.responses.create(model=MODEL, instructions=AGENT_INSTRUCTIONS, input=items, tools=TOOLS)
        tokens_in += response.usage.input_tokens
        tokens_out += response.usage.output_tokens
        calls = [item for item in response.output if item.type == "function_call"]

        if not calls:   # a normal answer: the model has decided it's done
            cost = cost_usd(MODEL, tokens_in, tokens_out)
            print(f"A: {response.output_text}")
            print(f"   [{step} model calls, {tool_calls} tool calls, {tokens_in} in / {tokens_out} out tokens, ${cost:.6f}]")
            return response.output_text

        items += response.output   # the model's tool requests become part of the conversation
        for call in calls:          # there may be several at once (parallel tool calls)
            result = run_tool(call.name, call.arguments)
            tool_calls += 1
            print(f"   step {step}: {call.name}({call.arguments}) -> {result[:90]!r}{'...' if len(result) > 90 else ''}")
            items.append({"type": "function_call_output", "call_id": call.call_id, "output": result})

    print(f"   STOPPED: reached the {MAX_STEPS}-step limit")
    return "Sorry, I couldn't finish that request."


QUESTIONS = [
    "How much is the Growth plan?",
    "If I need 900 extra minutes on the Growth plan, what's my total monthly bill?",
    "Compare the Starter and Scale plans: price and minutes.",
    "Hi there!",
    "What's the weather in Paris?",
]

if __name__ == "__main__":
    for q in (sys.argv[1:] and [" ".join(sys.argv[1:])]) or QUESTIONS:
        run_agent(q)

    print(f"\n{'=' * 70}\nSafety test: the calculator with malicious input")
    print("  calculate('899 + 300')                      ->", run_tool("calculate", '{"expression": "899 + 300"}'))
    print("  calculate(\"__import__('os').system('ls')\") ->",
          run_tool("calculate", json.dumps({"expression": "__import__('os').system('ls')"})))
