import uuid

import psycopg
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_postgres import PostgresChatMessageHistory

from ask import INSTRUCTIONS, REFUSAL_ANSWER
from db import DATABASE_URL
from lc_rag import format_context, llm, retrieve

MAX_HISTORY_MESSAGES = 6   # keep only the last 6 messages (3 question/answer pairs)

rewriter = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest message as a standalone question about Calldesk that can be "
               "understood without the chat history. Keep the user's meaning. If it is already standalone "
               "or changes topic, return it unchanged. Return only the question."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
]) | llm | StrOutputParser()

answer_chain = ChatPromptTemplate.from_messages([
    ("system", INSTRUCTIONS),
    MessagesPlaceholder("history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]) | llm


def recent(history):
    """Last N messages, always starting with a human message."""
    return trim_messages(history.messages, max_tokens=MAX_HISTORY_MESSAGES, token_counter=len,
                         strategy="last", start_on="human")


def chat(question, history):
    past = recent(history)
    standalone = rewriter.invoke({"history": past, "question": question}) if past else question   # 1. rewrite
    results = retrieve(standalone)                                                                  # 2. search
    if results:                                                                                     # 3. answer
        message = answer_chain.invoke({"history": past, "context": format_context(results), "question": standalone})
        answer, tokens = message.content, message.usage_metadata["input_tokens"]
    else:
        answer, tokens = REFUSAL_ANSWER, 0
    history.add_messages([HumanMessage(question), AIMessage(answer)])                               # 4. save

    print(f"\n  You:       {question}")
    if standalone != question:
        print(f"  Rewritten: {standalone}")
    print(f"  Searched:  {[f'{c.metadata['source']} #{c.metadata['chunk_id']}' for _, c in results]}")
    print(f"  Bot:       {answer}")
    print(f"  (history sent: {len(past)} messages, input tokens: {tokens})")
    return answer


CONVERSATION = [
    "How much is the Growth plan?",
    "And Scale?",
    "Which one has 24/7 support?",
    "Do you speak Spanish?",          # topic switch
]

if __name__ == "__main__":
    # Part 1: the problem - "And Scale?" on its own
    print("=== Part 1: no memory ===")
    print("  Searching for 'And Scale?' alone finds:",
          [f"{c.metadata['source']} #{c.metadata['chunk_id']} ({s:.2f})" for s, c in retrieve("And Scale?")])

    # Parts 2-4: conversation with in-memory history
    print("\n=== Parts 2-4: conversation with memory (in-memory history) ===")
    memory_history = InMemoryChatMessageHistory()
    for q in CONVERSATION:
        chat(q, memory_history)

    # Part 5: Postgres history survives a "restart"
    print("\n=== Part 5: Postgres history ===")
    session_id = str(uuid.uuid4())
    with psycopg.connect(DATABASE_URL) as conn:
        PostgresChatMessageHistory.create_tables(conn, "chat_history")
        chat("How much is the Starter plan?", PostgresChatMessageHistory("chat_history", session_id, sync_connection=conn))
    print("\n  --- program 'restarts': new connection, new history object, same session id ---")
    with psycopg.connect(DATABASE_URL) as conn:
        restored = PostgresChatMessageHistory("chat_history", session_id, sync_connection=conn)
        print(f"  Restored {len(restored.messages)} messages from Postgres for session {session_id[:8]}...")
        chat("How many minutes does it include?", restored)

    # Part 6: trimming keeps the prompt small as conversations grow
    print("\n=== Part 6: trimming ===")
    all_messages = memory_history.messages
    for n in (len(all_messages), 6, 4, 2):
        kept = trim_messages(all_messages, max_tokens=n, token_counter=len, strategy="last", start_on="human")
        print(f"  keep last {n:>2}: {len(kept)} messages, {llm.get_num_tokens_from_messages(kept)} tokens")
