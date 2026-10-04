from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from src import guardrails
from src.config import DB_PATH, LOG_PATH
from src.executor import ExecResult, append_jsonl, dialect_for, run_query
from src.nl2sql import generate_sql


@dataclass
class Answer:
    question: str
    variant: str
    raw_sql: str
    verdict: guardrails.Verdict | None
    result: ExecResult | None
    error: str | None
    latency_ms: float

    @property
    def status(self):
        if self.error:
            return "generation_error"
        if not self.verdict.ok:
            return "blocked"
        if not self.result.ok:
            return "execution_error"
        return "ok"


def answer_question(question, variant, llm, retriever=None, db_path=DB_PATH, log_path=LOG_PATH, validator: Callable | None = None):
    start = time.perf_counter()
    validator = validator or guardrails.validate
    raw_sql, verdict, result, error = "", None, None, None
    try:
        raw_sql = generate_sql(question, variant, llm, retriever)
        verdict = validator(raw_sql, dialect=dialect_for(db_path))
        if verdict.ok:
            result = run_query(verdict.sql, db_path)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    answer = Answer(question, variant, raw_sql, verdict, result, error, (time.perf_counter() - start) * 1000)
    if log_path:
        append_jsonl(
            log_path,
            {
                "question": question,
                "variant": variant,
                "sql": raw_sql,
                "status": answer.status,
                "reason": verdict.reason if verdict else error,
                "rows": 0 if not (result and result.ok) else len(result.frame),
                "latency_ms": round(answer.latency_ms, 1),
            },
        )
    return answer
