import time
from typing import Literal

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ask import MODEL
from lc_rag import format_context, retrieve   # reuse the A4 retriever

llm = ChatOpenAI(model=MODEL, timeout=20, max_retries=2)


# Part 1: RunnableLambda - any function becomes a chain step
shout = RunnableLambda(lambda text: text.upper())
greet = ChatPromptTemplate.from_template("Say hello to {name} in 3 words.") | llm | StrOutputParser() | shout
print("1. RunnableLambda:", greet.invoke({"name": "Calldesk customers"}))


# Part 2: routing - classify the question, then send it to the right specialist prompt
class Route(BaseModel):
    topic: Literal["pricing", "support", "other"]


classifier = ChatPromptTemplate.from_messages([
    ("system", "Classify this customer question about Calldesk. "
               "pricing = plans, prices, minutes, billing, discounts. "
               "support = setup, refunds, policies, how the product works. "
               "other = anything not about Calldesk."),
    ("human", "{question}"),
]) | llm.with_structured_output(Route)

SPECIALISTS = {
    "pricing": "You are Calldesk's pricing specialist. Quote exact prices and minutes. Answer in 1-2 sentences.",
    "support": "You are Calldesk's support specialist. Explain the relevant policy clearly, step by step.",
}


def route(inputs):
    topic = classifier.invoke({"question": inputs["question"]}).topic
    if topic == "other":
        return RunnableLambda(lambda _: "[other] Sorry, I can only help with questions about Calldesk.")
    specialist = ChatPromptTemplate.from_messages([
        ("system", SPECIALISTS[topic] + " Use ONLY the context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    add_context = RunnableLambda(lambda x: {**x, "context": format_context(retrieve(x["question"]))})
    return add_context | specialist | llm | StrOutputParser() | RunnableLambda(lambda a: f"[{topic}] {a}")


router = RunnableLambda(route)   # a function that RETURNS a chain: LCEL then runs that chain
print("\n2. Routing:")
for q in ["How much is the Scale plan?", "How do I get a refund?", "What's the weather in Paris?"]:
    print(f"   {q}\n   -> {router.invoke({'question': q})}")


# Part 3: fallbacks - the main model is broken, the backup answers
broken = ChatOpenAI(model="gpt-banana-99", max_retries=0)
safe = broken.with_fallbacks([llm])
reply = safe.invoke("Reply with exactly: fallback worked")
print(f"\n3. Fallback: {reply.content!r}, answered by {reply.response_metadata.get('model_name')}")


# Part 4: retries - fails twice, succeeds on the third try
attempts = {"n": 0}


def flaky(x):
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ConnectionError("simulated network blip")
    return f"succeeded on attempt {attempts['n']}"


print("\n4. Retry:", RunnableLambda(flaky).with_retry(stop_after_attempt=3, wait_exponential_jitter=False).invoke("x"))


# Part 5: caching - the second identical call is instant and free
set_llm_cache(InMemoryCache())
question = "In one sentence, what does an AI phone receptionist do?"
print("\n5. Caching:")
for i in range(2):
    start = time.perf_counter()
    r = llm.invoke(question)
    print(f"   call {i + 1}: {round((time.perf_counter() - start) * 1000)} ms, usage reported: {r.usage_metadata}")
set_llm_cache(None)


# Part 6: configurable fields - turn a dial at run time
flexible = llm.configurable_fields(temperature=ConfigurableField(id="temperature"))
slogan = ChatPromptTemplate.from_template("Give a 5-word slogan for {product}.") | flexible | StrOutputParser()
print("\n6. Configurable temperature:")
for t in (0, 1.2):
    outs = slogan.with_config(configurable={"temperature": t}).batch([{"product": "an AI receptionist"}] * 3)
    print(f"   temperature={t}: {outs}")
