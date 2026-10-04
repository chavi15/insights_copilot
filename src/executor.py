from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, QUERY_TIMEOUT_S


@dataclass
class ExecResult:
    ok: bool
    frame: pd.DataFrame | None
    error: str | None
    latency_ms: float


def dialect_for(db_path):
    return "duckdb" if str(db_path).endswith(".duckdb") else "sqlite"


def _run_duckdb(sql, db_path, timeout_s):
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
        timer = threading.Timer(timeout_s, con.interrupt)
        timer.start()
        try:
            return con.execute(sql).df()
        finally:
            timer.cancel()
    finally:
        con.close()


def _run_sqlite(sql, db_path, timeout_s):
    con = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        deadline = time.monotonic() + timeout_s
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def run_query(sql, db_path=DB_PATH, timeout_s=QUERY_TIMEOUT_S):
    start = time.perf_counter()
    runner = _run_duckdb if dialect_for(db_path) == "duckdb" else _run_sqlite
    try:
        frame = runner(sql, db_path, timeout_s)
        return ExecResult(True, frame, None, (time.perf_counter() - start) * 1000)
    except Exception as error:
        return ExecResult(False, None, f"{type(error).__name__}: {str(error)[:300]}", (time.perf_counter() - start) * 1000)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
