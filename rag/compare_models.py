"""A10: run the eval on several models and compare accuracy, latency, tokens and cost.

Each model runs your unchanged production RAG (ask.py): only OPENAI_MODEL changes.
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODELS = ["gpt-4o-mini", "gpt-4.1-nano", "gpt-5-nano", "gpt-5.6-luna", "gpt-5.4-mini"]
RUNS = 2
HERE = Path(__file__).parent
LOG_DIR = HERE / "logs" / "compare"


def run_eval(model, log_file):
    env = {**os.environ, "OPENAI_MODEL": model, "LOG_FILE": str(log_file)}
    out = subprocess.run([sys.executable, str(HERE / "eval.py")], env=env,
                         capture_output=True, text=True, timeout=1800).stdout
    answer = re.search(r"Answer accuracy:\s+(\d+)%", out)
    refusal = re.search(r"Refusal accuracy:\s+(\d+)%", out)
    return {"answer": int(answer.group(1)) if answer else None,
            "refusal": int(refusal.group(1)) if refusal else None,
            "fails": re.findall(r"\[FAIL\] (.+)", out)}


def percentile(values, p):
    return values[min(len(values) - 1, int(p * len(values)))] if values else 0


def evaluate_model(model):
    log_file = LOG_DIR / f"{model}.jsonl"
    log_file.unlink(missing_ok=True)
    runs = [run_eval(model, log_file) for _ in range(RUNS)]
    rows = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    ok = [r for r in rows if "error" not in r]
    answered = [r for r in ok if r.get("input_tokens")]
    latencies = sorted(r["total_ms"] for r in ok)
    costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
    return {
        "model": model, "runs": runs, "errors": len(rows) - len(ok),
        "p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95),
        "out_tokens": round(sum(r["output_tokens"] for r in answered) / len(answered)) if answered else 0,
        "cost": sum(costs) / len(costs) if costs else None,
    }


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running the {RUNS}-run eval on {len(MODELS)} models (models in parallel)...")
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        results = list(pool.map(evaluate_model, MODELS))

    print(f"\n{'Model':<14} {'Answers':<11} {'Refusals':<11} {'p50':>7} {'p95':>7} {'Out tok':>8} "
          f"{'$/answer':>10} {'$/1k':>7} {'Err':>4}")
    for r in results:
        answers = " / ".join(f"{run['answer']}%" for run in r["runs"])
        refusals = " / ".join(f"{run['refusal']}%" for run in r["runs"])
        cost = f"{r['cost']:.6f}" if r["cost"] is not None else "?"
        per_1k = f"{r['cost'] * 1000:.2f}" if r["cost"] is not None else "?"
        print(f"{r['model']:<14} {answers:<11} {refusals:<11} {r['p50']:>5}ms {r['p95']:>5}ms "
              f"{r['out_tokens']:>8} {cost:>10} {per_1k:>7} {r['errors']:>4}")

    print("\nFailed questions:")
    for r in results:
        fails = sorted({f for run in r["runs"] for f in run["fails"]})
        print(f"  {r['model']}: {fails if fails else 'none'}")
