import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

load_dotenv()
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

response = client.responses.create(
    model=MODEL,
    instructions="You are a concise tutor. Answer in at most two sentences.",
    input="What is an embedding in machine learning?",
)

print(response.output_text)
print(response.usage)
print("MODEL variable:", MODEL)
print("Model OpenAI used:", response.model)