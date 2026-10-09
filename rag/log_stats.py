import json

from request_log import LOG_FILE

rows = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]
ok = [r for r in rows if "error" not in r]
latencies = sorted(r["total_ms"] for r in ok)
costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]


def percentile(values, p):
    return values[min(len(values) - 1, int(p * len(values)))] if values else 0


print(f"Requests:        {len(rows)}  (errors: {len(rows) - len(ok)})")
print(f"Latency p50/p95: {percentile(latencies, 0.50)} ms / {percentile(latencies, 0.95)} ms")
if costs:
    print(f"Cost per answer: ${sum(costs) / len(costs):.6f} average, ${sum(costs):.4f} total")
