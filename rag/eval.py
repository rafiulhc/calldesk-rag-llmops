import json
import sys
import time
from pathlib import Path

from ask import ask

QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"
REFUSAL = "don't know"


def normalize(text):
    return text.lower().replace("\u2019", "'")   # treat curly ’ as a straight '


def as_list(value):
    return value if isinstance(value, list) else [value]


def has_fact(answer, fact):
    """A fact is a string, or a list of acceptable alternative wordings."""
    return any(normalize(option) in answer for option in as_list(fact))


def check(case, answer, results):
    answer = normalize(answer)
    refused = REFUSAL in answer
    if case.get("refuse"):
        return {"retrieval": None, "answer": refused}

    sources = {c["source"] for _, c in results}
    retrieval_ok = any(s in sources for s in as_list(case["source"]))
    facts_ok = all(has_fact(answer, f) for f in case["facts"])
    return {"retrieval": retrieval_ok, "answer": facts_ok and not refused}


def http_ask(base_url):
    """Ask the deployed API over HTTP(S), exactly like a real client. Key from EVAL_API_KEY."""
    import os
    import urllib.request

    key = os.environ["EVAL_API_KEY"]

    def _ask(question):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/ask", method="POST",
            data=json.dumps({"question": question}).encode(),
            headers={"X-API-Key": key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.load(response)
        return body["answer"], [(s["score"], s) for s in body["sources"]]
    return _ask


def main():
    if "--langchain" in sys.argv:
        from lc_rag import lc_ask as answer_fn
        print("Evaluating: LangChain RAG\n")
    elif "--graph" in sys.argv:
        from graph_basics import graph_ask as answer_fn
        print("Evaluating: LangGraph self-correcting RAG\n")
    elif url := next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), None):
        answer_fn = http_ask(url)
        print(f"Evaluating: deployed API at {url}\n")
    else:
        answer_fn = ask
        print("Evaluating: hand-written RAG\n")
    cases = json.loads(QUESTIONS_FILE.read_text())
    outcomes = []
    start = time.perf_counter()

    for case in cases:
        answer, results = answer_fn(case["question"])
        outcome = check(case, answer, results)
        outcomes.append((case, outcome))
        print(f"[{'PASS' if outcome['answer'] else 'FAIL'}] {case['question']}")
        if not outcome["answer"]:
            print(f"        retrieved: {[c['source'] for _, c in results]}")
            print(f"        answer:    {answer[:160]!r}")

    answerable = [o for c, o in outcomes if not c.get("refuse")]
    refusals = [o for c, o in outcomes if c.get("refuse")]

    def pct(items, key):
        return 100 * sum(o[key] for o in items) / len(items) if items else 0

    print("\n=== Results ===")
    print(f"Retrieval hit rate: {pct(answerable, 'retrieval'):.0f}%  ({len(answerable)} questions)")
    print(f"Answer accuracy:    {pct(answerable, 'answer'):.0f}%")
    print(f"Refusal accuracy:   {pct(refusals, 'answer'):.0f}%  ({len(refusals)} questions)")
    print(f"Total time:         {time.perf_counter() - start:.1f}s")
    return {"retrieval": pct(answerable, "retrieval"), "answer": pct(answerable, "answer"),
            "refusal": pct(refusals, "answer")}


def gate(scores, minimum):
    """CI eval gate: every metric must reach `minimum` percent. Returns the failing metrics."""
    return [f"{name} {value:.0f}% < {minimum:.0f}%" for name, value in scores.items() if value < minimum]


if __name__ == "__main__":
    scores = main()
    minimum = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--min-accuracy=")), None)
    if minimum is not None:
        failures = gate(scores, minimum)
        print(f"\nEval gate ({minimum:.0f}%): {'FAILED: ' + ', '.join(failures) if failures else 'passed'}")
        sys.exit(1 if failures else 0)