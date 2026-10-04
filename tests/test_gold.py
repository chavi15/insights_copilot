import pandas as pd
import pytest

from src.config import GOLD_PATH
from src.evaluation import load_gold
from src.examples import FEW_SHOT_EXAMPLES
from src.executor import run_query
from src.guardrails import precheck


@pytest.fixture(scope="module")
def gold():
    return load_gold(GOLD_PATH)


def test_gold_shape(gold):
    assert len(gold) == 40
    counts = pd.Series([g["difficulty"] for g in gold]).value_counts()
    assert counts["easy"] == 15 and counts["medium"] == 15 and counts["hard"] == 10
    assert len({g["id"] for g in gold}) == 40
    assert len({g["question"] for g in gold}) == 40


def test_gold_not_in_few_shot(gold):
    shots = {e["question"].lower() for e in FEW_SHOT_EXAMPLES}
    shot_sql = {e["sql"].strip().lower() for e in FEW_SHOT_EXAMPLES}
    for item in gold:
        assert item["question"].lower() not in shots
        assert item["sql"].strip().lower() not in shot_sql


def test_gold_sql_passes_guardrails(gold):
    for item in gold:
        assert precheck(item["sql"]) is None, item["id"]


def test_gold_sql_runs_and_returns_rows(gold, sqlite_db):
    for item in gold:
        outcome = run_query(item["sql"], sqlite_db)
        assert outcome.ok, (item["id"], outcome.error)
        assert len(outcome.frame) > 0, item["id"]


def test_gold_results_are_deterministic(gold, sqlite_db):
    for item in gold:
        first = run_query(item["sql"], sqlite_db).frame
        second = run_query(item["sql"], sqlite_db).frame
        pd.testing.assert_frame_equal(first, second)


def test_hard_questions_use_advanced_sql(gold):
    hard = [g["sql"].lower() for g in gold if g["difficulty"] == "hard"]
    advanced = sum(1 for s in hard if "over (" in s or "with " in s or "exists" in s or "case when" in s)
    assert advanced == len(hard)
