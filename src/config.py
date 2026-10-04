import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "sales.duckdb")))
CACHE_DIR = ROOT / ".cache" / "llm"
CHROMA_DIR = ROOT / ".cache" / "chroma"
LOG_PATH = ROOT / "logs" / "queries.jsonl"
FEEDBACK_PATH = ROOT / "logs" / "feedback.jsonl"
SCHEMA_DOCS_PATH = ROOT / "src" / "schema_docs.yaml"
GOLD_PATH = ROOT / "eval" / "gold.yaml"
EVAL_DIR = ROOT / "eval"

MAX_ROWS = 1000
QUERY_TIMEOUT_S = 10.0
