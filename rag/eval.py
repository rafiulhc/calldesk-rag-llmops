import json
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


def main():
    cases = json.loads(QUESTIONS_FILE.read_text())
    outcomes = []
    start = time.perf_counter()

    for case in cases:
        answer, results = ask(case["question"])
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


if __name__ == "__main__":
    main()