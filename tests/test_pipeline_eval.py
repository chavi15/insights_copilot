from functools import partial

import pandas as pd
import pytest

from src import guardrails
from src.config import GOLD_PATH
from src.evaluation import (
    classify_failure,
    failure_breakdown,
    load_gold,
    normalize_value,
    results_match,
    run_evaluation,
    summarize,
    write_report,
)
from src.executor import dialect_for, run_query
from src.llm import ScriptedLLM
from src.nl2sql import VARIANTS
from src.pipeline import answer_question
from src.retrieval import KeywordRetriever, TableRetriever

light = partial(guardrails.validate, ast=False)


@pytest.fixture(scope="module")
def gold():
    return load_gold(GOLD_PATH)


@pytest.fixture(scope="module")
def retriever():
    return TableRetriever(KeywordRetriever(), k=3)


def test_sqlite_connection_is_read_only(sqlite_db):
    assert not run_query("DELETE FROM dim_hcp", sqlite_db).ok
    assert not run_query("DROP TABLE dim_hcp", sqlite_db).ok
    assert run_query("SELECT COUNT(*) AS n FROM dim_hcp", sqlite_db).frame["n"].iloc[0] == 2000


def test_query_timeout(sqlite_db):
    outcome = run_query(
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r", sqlite_db, timeout_s=0.5
    )
    assert not outcome.ok and outcome.latency_ms < 5000


def test_dialect_detection():
    assert dialect_for("x/sales.duckdb") == "duckdb" and dialect_for("x/sales.sqlite") == "sqlite"


def test_normalization_rules():
    assert normalize_value(5) == normalize_value(5.0) == "5.00"
    assert normalize_value(None) == normalize_value(float("nan")) == "NULL"
    assert normalize_value(pd.Timestamp("2024-04-01")) == "2024-04-01" == normalize_value("2024-04-01")
    a = pd.DataFrame({"x": ["a", "b"], "y": [1.001, 2.0]})
    b = pd.DataFrame({"n": [2, 1], "c": ["b", "a"]})
    assert results_match(a, b)
    assert not results_match(a, pd.DataFrame({"x": ["a", "b"], "y": [1.5, 2.0]}))
    assert not results_match(a, a.iloc[:1])
    assert not results_match(None, a)


def test_pipeline_blocks_bad_sql_and_logs(sqlite_db, retriever, tmp_path):
    log = tmp_path / "q.jsonl"
    llm = ScriptedLLM({"wipe": "DROP TABLE dim_hcp", "ok": "SELECT COUNT(*) AS n FROM dim_hcp", "bad col": "SELECT nope FROM dim_hcp"})
    blocked = answer_question("wipe", "few_shot_full_schema", llm, retriever, sqlite_db, log, light)
    good = answer_question("ok", "retrieval_few_shot", llm, retriever, sqlite_db, log, light)
    broken = answer_question("bad col", "zero_shot_full_schema", llm, retriever, sqlite_db, log, light)
    assert (blocked.status, good.status, broken.status) == ("blocked", "ok", "execution_error")
    assert run_query("SELECT COUNT(*) AS n FROM dim_hcp", sqlite_db).frame["n"].iloc[0] == 2000
    assert len(log.read_text().strip().splitlines()) == 3


def test_oracle_llm_scores_100_percent_on_every_variant(gold, sqlite_db, retriever, tmp_path):
    llm = ScriptedLLM({g["question"]: g["sql"] for g in gold})
    records = run_evaluation(gold, llm, retriever, VARIANTS, sqlite_db, validator=light)
    assert len(records) == 120 and records["correct"].all()
    summary = summarize(records)
    assert (summary["execution_accuracy"] == 100.0).all() and (summary["guardrail_block_rate"] == 0).all()
    meta = {"generated": "t", "model": "scripted", "questions": 40, "easy": 15, "medium": 15, "hard": 10}
    write_report(records, tmp_path / "REPORT.md", tmp_path / "results.csv", meta)
    assert "Results by variant" in (tmp_path / "REPORT.md").read_text()
    assert len(pd.read_csv(tmp_path / "results.csv")) == 120


def test_sabotaged_llm_is_scored_and_classified(gold, sqlite_db, retriever):
    by_id = {g["id"]: g for g in gold}
    answers = {g["question"]: g["sql"] for g in gold}
    answers[by_id["m02"]["question"]] = "SELECT 'x' AS region, ROUND(SUM(revenue), 2) AS total_revenue FROM fact_sales"
    answers[by_id["m08"]["question"]] = by_id["m08"]["sql"].replace("SUM(s.revenue)", "COUNT(s.revenue)")
    answers[by_id["m01"]["question"]] = by_id["m01"]["sql"].replace("d.year = 2025", "d.year = 2024")
    answers[by_id["e05"]["question"]] = "SELECT SUM(revenue_usd) FROM fact_sales"
    answers[by_id["e01"]["question"]] = "DROP TABLE dim_product"
    llm = ScriptedLLM(answers)
    records = run_evaluation(gold, llm, retriever, ["few_shot_full_schema"], sqlite_db, validator=light).set_index("id")
    assert not records.loc[["m02", "m08", "m01", "e05", "e01"], "correct"].any()
    assert records.loc["m02", "failure_category"] == "wrong join"
    assert records.loc["m08", "failure_category"] == "wrong aggregation"
    assert records.loc["m01", "failure_category"] == "wrong filter or date logic"
    assert records.loc["e05", "failure_category"] == "hallucinated column or table"
    assert records.loc["e01", "status"] == "blocked"
    assert records["correct"].sum() == 35
    breakdown = failure_breakdown(records.reset_index())
    assert breakdown.loc["few_shot_full_schema"].sum() == 5


def test_classify_failure_direct():
    gold_sql = "SELECT region, SUM(x) FROM a JOIN b ON a.k = b.k WHERE y = 1 GROUP BY region"
    assert classify_failure("ok", None, "SELECT region, SUM(x) FROM a GROUP BY region", gold_sql) == "wrong join"
    assert classify_failure("execution_error", "Catalog Error: Table with name zz does not exist", "x", gold_sql) == "hallucinated column or table"
    assert classify_failure("ok", None, gold_sql.replace("y = 1", "y = 2"), gold_sql) == "wrong filter or date logic"
