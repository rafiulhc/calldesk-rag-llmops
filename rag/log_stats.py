import json
from collections import defaultdict

from request_log import LOG_FILE

rows = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]
groups = defaultdict(list)
for r in rows:
    groups[r.get("impl", "hand-written")].append(r)


def percentile(values, p):
    return values[min(len(values) - 1, int(p * len(values)))] if values else 0


for impl, items in groups.items():
    ok = [r for r in items if "error" not in r]
    latencies = sorted(r["total_ms"] for r in ok)
    costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
    print(f"\n[{impl}]")
    print(f"  Requests:        {len(items)}  (errors: {len(items) - len(ok)})")
    print(f"  Latency p50/p95: {percentile(latencies, 0.50)} ms / {percentile(latencies, 0.95)} ms")
    if costs:
        print(f"  Cost per answer: ${sum(costs) / len(costs):.6f} average")
