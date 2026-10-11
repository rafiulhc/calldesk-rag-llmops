import os
import tempfile
from pathlib import Path

# Set fake config BEFORE any app module is imported, so tests never use
# real keys, never touch the real database, never write to the real log,
# and never send traces to LangSmith.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"   # rate-limit tests use their own limiter
os.environ["OPENAI_API_KEY"] = "sk-test-not-real"
os.environ["APP_API_KEY"] = "test-api-key"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
os.environ["LOG_FILE"] = str(Path(tempfile.mkdtemp()) / "requests.jsonl")