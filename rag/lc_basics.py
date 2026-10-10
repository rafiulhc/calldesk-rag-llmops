import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, timeout=20, max_retries=3)


# Part 1: call the model directly
message = llm.invoke("What is an embedding? Answer in one sentence.")
print("1.", message.content)
print("   usage:", message.usage_metadata)

# Part 2: prompt template + chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise tutor. Answer in at most {max_sentences} sentences."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()
print("\n2.", chain.invoke({"question": "What is RAG?", "max_sentences": 2}))

# Part 3: batch (several inputs in parallel) and stream (token by token)
answers = chain.batch([
    {"question": "What is pgvector?", "max_sentences": 1},
    {"question": "What is cosine similarity?", "max_sentences": 1},
])
print("\n3a.", *answers, sep="\n    ")
print("\n3b. ", end="")
for piece in chain.stream({"question": "Why chunk documents?", "max_sentences": 2}):
    print(piece, end="", flush=True)
print()


# Part 4: structured output, getting a typed Python object instead of text
class PlanInfo(BaseModel):
    name: str = Field(description="The plan's name")
    price_usd: int = Field(description="Monthly price in US dollars")
    minutes: int = Field(description="Minutes included per month")


extractor = llm.with_structured_output(PlanInfo)
plan = extractor.invoke("Growth plan: 899 USD per month. Includes 1,200 minutes per month and 2 phone numbers.")
print("\n4.", plan)
print("   price per minute:", round(plan.price_usd / plan.minutes, 3))
