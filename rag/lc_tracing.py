"""A9: observability with LangSmith.

Needs LANGSMITH_TRACING=true, LANGSMITH_API_KEY and LANGSMITH_PROJECT in .env.
Run:  uv run rag/lc_tracing.py          (parts 1-4: traces)
      uv run rag/lc_tracing.py eval     (part 5: dataset + experiments)
"""
import json
import sys

from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai

import ask
import ingest
import search
from eval import QUESTIONS_FILE, check

DATASET = "calldesk-eval-32"


# ---------- Part 2: trace the HAND-WRITTEN RAG without editing it ----------
# wrap_openai records every OpenAI call (prompt, tokens, latency); traceable records our own functions.
traced_client = wrap_openai(ingest.client)
ingest.client = traced_client          # used by embed_texts() inside search()
ask.client = traced_client             # used for the GPT call inside ask()
ask.search = traceable(search.search, name="search (pgvector)", run_type="retriever")
hand_written_ask = traceable(ask.ask, name="hand-written RAG", run_type="chain")


def langchain_ask(question):
    from lc_rag import lc_ask
    return lc_ask(question)


def run_traces():
    from lc_agent import TOOLS, llm
    from lc_rag import rag_chain
    from langchain.agents import create_agent
    import agent_scratch

    # Part 1 + 3: LangChain code traces itself; tags and metadata make runs filterable
    rag_chain.invoke("How much do extra minutes cost?",
                     config={"run_name": "LangChain RAG", "tags": ["rag", "langchain"],
                             "metadata": {"model": ask.MODEL, "top_k": ask.TOP_K}})
    print("1. LangChain RAG chain traced")

    # Part 2: hand-written RAG, traced through the wrappers above
    hand_written_ask("How much do extra minutes cost?",
                     langsmith_extra={"tags": ["rag", "hand-written"], "metadata": {"model": ask.MODEL}})
    print("2. Hand-written RAG traced")

    # Part 4: the agent with the hidden A7 problem - find it in the UI
    agent = create_agent(llm, TOOLS, system_prompt=agent_scratch.AGENT_INSTRUCTIONS)
    agent.invoke({"messages": [{"role": "user", "content":
                  "Does Calldesk make outbound calls, and what is the cheapest plan that includes voicemail "
                  "transcription and how much is it per year?"}]},
                 config={"run_name": "Calldesk agent", "tags": ["agent"], "metadata": {"model": ask.MODEL}})
    print("3. Agent traced")


# ---------- Part 5: the eval as a LangSmith dataset + experiments ----------
def upload_dataset(client):
    if client.has_dataset(dataset_name=DATASET):
        print(f"Dataset {DATASET!r} already exists")
        return
    cases = json.loads(QUESTIONS_FILE.read_text())
    client.create_dataset(DATASET, description="Calldesk RAG eval: 25 answerable + 7 refusal questions")
    client.create_examples(dataset_name=DATASET, examples=[
        {"inputs": {"question": c["question"]}, "outputs": {k: v for k, v in c.items() if k != "question"}}
        for c in cases])
    print(f"Uploaded {len(cases)} examples to dataset {DATASET!r}")


def make_target(answer_fn):
    def target(inputs):
        answer, results = answer_fn(inputs["question"])
        return {"answer": answer, "results": [[score, chunk] for score, chunk in results]}
    return target


def calldesk_checker(inputs, outputs, reference_outputs):
    """Your own eval.check(), reused as a LangSmith evaluator."""
    case = {"question": inputs["question"], **reference_outputs}
    outcome = check(case, outputs["answer"], outputs["results"])
    results = [{"key": "answer_correct", "score": int(outcome["answer"])}]
    if outcome["retrieval"] is not None:
        results.append({"key": "retrieval_hit", "score": int(outcome["retrieval"])})
    return {"results": results}


def run_experiments():
    client = Client()
    upload_dataset(client)
    for name, fn in [("hand-written", hand_written_ask), ("langchain", langchain_ask)]:
        result = client.evaluate(make_target(fn), data=DATASET, evaluators=[calldesk_checker],
                                 experiment_prefix=f"{name}-{ask.MODEL}", max_concurrency=4,
                                 metadata={"impl": name, "model": ask.MODEL})
        print(f"Experiment done: {result.experiment_name}")


if __name__ == "__main__":
    run_experiments() if "eval" in sys.argv else run_traces()
