import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(os.getenv("LOG_FILE", Path(__file__).parent / "logs" / "requests.jsonl"))

# USD per 1 million tokens. Check current prices at openai.com/api/pricing.
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def cost_usd(model, input_tokens, output_tokens=0):
    price = PRICES.get(model)
    if price is None:
        return None   # unknown model: show "unknown", never a fake 0
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def log_request(record):
    record = {"id": str(uuid.uuid4()), "time": datetime.now(timezone.utc).isoformat(), **record}
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")