"""G3: memory and checkpoints - conversation memory, threads, Postgres persistence,
crash recovery, time travel, long-term memory (Store) and trimming."""
import uuid
from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.postgres import PostgresStore

import agent_scratch
from db import DATABASE_URL
from lc_agent import TOOLS, llm


@dataclass
class Context:
    user_id: str          # who is talking: used for long-term memory


@tool
def remember_customer_fact(fact: str, runtime: ToolRuntime[Context]) -> str:
    """Save a lasting fact the customer shared about themselves or their business
    (industry, location, current plan, needs) so future conversations can use it."""
    runtime.store.put(("customers", runtime.context.user_id), str(uuid.uuid4()), {"fact": fact})
    return f"Saved: {fact}"


INSTRUCTIONS = agent_scratch.AGENT_INSTRUCTIONS + """
- If the customer shares lasting facts about themselves or their business, save each one with remember_customer_fact.
- Use the known customer facts below to personalise answers."""


def build_app(checkpointer, store=None, tools=None, max_messages=None, crash_once=None):
    tools = tools or TOOLS
    model_with_tools = llm.bind_tools(tools)

    def model(state, runtime: Runtime[Context]):
        messages = state["messages"]
        if max_messages:   # trim old turns, always starting cleanly at a human message
            messages = trim_messages(messages, max_tokens=max_messages, token_counter=len,
                                     strategy="last", start_on="human")
        facts = []
        if runtime.store is not None and runtime.context is not None:
            facts = [item.value["fact"] for item in runtime.store.search(("customers", runtime.context.user_id))]
        system = INSTRUCTIONS + "\nKnown customer facts: " + ("; ".join(facts) if facts else "none")
        return {"messages": [model_with_tools.invoke([SystemMessage(system)] + messages)]}

    def after_model(state):
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(MessagesState, context_schema=Context)
    graph.add_node("model", model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=crash_once is None))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=checkpointer, store=store)


def thread(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def ask(app, thread_id, question, user="customer-1"):
    before = len(app.get_state(thread(thread_id)).values.get("messages", []))
    state = app.invoke({"messages": [HumanMessage(question)]}, thread(thread_id), context=Context(user_id=user))
    new = state["messages"][before:]
    calls = [f"{tc['name']}({', '.join(f'{v}' for v in tc['args'].values())})"
             for m in new if isinstance(m, AIMessage) for tc in m.tool_calls]
    last_model_input = [m.usage_metadata["input_tokens"] for m in new if isinstance(m, AIMessage)][-1]
    print(f"  You: {question}")
    print(f"    tools: {calls or 'none'}   (last model call input tokens: {last_model_input})")
    print(f"    Bot: {state['messages'][-1].content[:220]}")
    return last_model_input


if __name__ == "__main__":
    # ---------- Part 1: conversation memory with a checkpointer ----------
    print("=== Part 1: conversation memory (InMemorySaver) ===")
    memory_app = build_app(InMemorySaver())
    t1 = f"customer-1-{uuid.uuid4().hex[:6]}"
    for q in ["How much is the Growth plan?", "And Scale?", "Which one has 24/7 support?"]:
        ask(memory_app, t1, q)

    # ---------- Part 2: threads are isolated ----------
    print("\n=== Part 2: a different thread knows nothing ===")
    ask(memory_app, f"customer-2-{uuid.uuid4().hex[:6]}", "And Scale?", user="customer-2")

    # ---------- Part 3: Postgres persistence survives a restart ----------
    print("\n=== Part 3: PostgresSaver - conversation survives a 'restart' ===")
    t3 = f"pg-{uuid.uuid4().hex[:8]}"
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()                                   # creates the checkpoint tables (once)
        ask(build_app(saver), t3, "How much is the Starter plan?")
    print("  --- process 'restarts': new connection, new saver, new app object ---")
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        app = build_app(saver)
        print(f"  restored {len(app.get_state(thread(t3)).values['messages'])} messages from Postgres")
        ask(app, t3, "How many minutes does it include?")

    # ---------- Part 4: crash and resume ----------
    print("\n=== Part 4: crash halfway, resume from the checkpoint ===")
    counts = {"get_plan": 0, "calculate": 0}

    @tool
    def get_plan(name: str) -> str:
        """Get exact data for one Calldesk plan: monthly price, included minutes, phone numbers, overage price."""
        counts["get_plan"] += 1
        return agent_scratch.get_plan(name)

    crashed = {"done": False}

    @tool
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression with numbers and + - * / ( )."""
        counts["calculate"] += 1
        if not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("simulated server crash during calculate")
        return agent_scratch.calculate(expression)

    crash_app = build_app(InMemorySaver(), tools=[get_plan, calculate], crash_once=True)
    t4 = f"crash-{uuid.uuid4().hex[:6]}"
    q = "On the Growth plan with 900 extra minutes, what is my total monthly bill? Look up the plan first, then calculate."
    try:
        crash_app.invoke({"messages": [HumanMessage(q)]}, thread(t4), context=Context(user_id="customer-4"))
    except RuntimeError as e:
        snapshot = crash_app.get_state(thread(t4))
        print(f"  CRASH: {e}")
        print(f"  saved so far: {len(snapshot.values['messages'])} messages, next node to run: {snapshot.next}")
        print(f"  tool calls so far: {counts}")
    result = crash_app.invoke(None, thread(t4), context=Context(user_id="customer-4"))   # None = resume, no new input
    print(f"  RESUMED -> {result['messages'][-1].content[:150]}")
    print(f"  tool calls after resume: {counts}   (get_plan did NOT run again)")

    # ---------- Part 5: time travel ----------
    print("\n=== Part 5: time travel on the Part 1 thread ===")
    history = list(memory_app.get_state_history(thread(t1)))
    print(f"  {len(history)} checkpoints saved for this 3-question conversation")
    turn_ends = [s for s in history if not s.next and s.values["messages"]
                 and isinstance(s.values["messages"][-1], AIMessage)]
    after_turn_2 = turn_ends[1]          # history is newest first: [turn 3 end, turn 2 end, turn 1 end]
    print(f"  forking from the end of turn 2 ({len(after_turn_2.values['messages'])} messages)...")
    fork = memory_app.invoke({"messages": [HumanMessage("And Starter?")]}, after_turn_2.config,
                             context=Context(user_id="customer-1"))
    print(f"  fork answer: {fork['messages'][-1].content[:150]}")
    print(f"  fork has {len(fork['messages'])} messages; it never saw turn 3 ('Which one has 24/7 support?')")

    # ---------- Part 6: long-term memory across threads (Store) ----------
    print("\n=== Part 6: long-term memory with PostgresStore ===")
    user = f"customer-{uuid.uuid4().hex[:6]}"
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver, PostgresStore.from_conn_string(DATABASE_URL) as store:
        store.setup()
        tools = TOOLS + [remember_customer_fact]
        app = build_app(saver, store=store, tools=tools)
        ask(app, f"{user}-chat-1", "Hi! I run a plumbing business in Texas and I get lots of after-hours calls.", user=user)
        print(f"  stored facts: {[i.value['fact'] for i in store.search(('customers', user))]}")
        print("  --- a NEW conversation (new thread) for the same customer ---")
        ask(app, f"{user}-chat-2", "Does Calldesk work where I am, and is it a good fit for my kind of business?", user=user)
        print("  --- a different customer, new thread ---")
        ask(app, f"other-{uuid.uuid4().hex[:6]}", "Does Calldesk work where I am?", user="someone-else")

    # ---------- Part 7: trimming keeps the prompt small ----------
    print("\n=== Part 7: trimming a long conversation ===")
    questions = ["How much is Starter?", "And Growth?", "And Scale?", "Which has bilingual support?",
                 "How much are extra minutes?", "And how long is the trial?"]
    for label, limit in [("no trimming", None), ("keep last 4 messages", 4)]:
        app = build_app(InMemorySaver(), max_messages=limit)
        tid = f"trim-{uuid.uuid4().hex[:6]}"
        print(f"  [{label}]")
        tokens = [ask(app, tid, q) for q in questions]
        print(f"  -> input tokens per turn: {tokens}\n")
