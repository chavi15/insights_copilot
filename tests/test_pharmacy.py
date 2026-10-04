import itertools

import pandas as pd
import pytest

from src import guardrails, retrieval
from src.datasets import DATASETS, PHARMA, PHARMACY, RETAIL
from src.evaluation import EvaluationStopped, load_gold, results_match, run_evaluation
from src.executor import run_query
from src.llm import RetryingLLM, ScriptedLLM, is_quota_error
from src.pharmacy_load import ATC_CODES, EXPECTED_COLUMNS, RAW_PATH, build_tables, read_raw, write_db
from src.pipeline import answer_question
from tests.pharmacy_reference import answers, load_wide

needs_raw = pytest.mark.skipif(not RAW_PATH.exists(), reason="copy salesdaily.csv into data/raw/ first")

# Table names that two datasets both use. Each lives in its own database file, so a shared name is not a leak,
# but the list is fixed here so a new overlap fails the test instead of slipping in.
KNOWN_SHARED_NAMES = {
    frozenset({"pharma", "retail"}): {"dim_product"},
    frozenset({"pharma", "pharmacy"}): {"dim_date"},
}


def tiny_wide():
    rows = [
        ["1/2/2014", 0, 3.67, 3.4, 32.4, 7, 0, 0, 2, 2014, 1, 248, "Thursday"],
        ["1/4/2014", 2, 1, 6.5, 61.85, 10, 0, 9, 1, 2014, 1, 276, "Saturday"],
        ["1/5/2014", 0, 0, 0, 0, 0, 0, 0, 0, 2014, 1, 276, "Sunday"],
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_tiny_melt_and_dates():
    tables, report = build_tables(tiny_wide())
    assert report.table_rows == {"dim_date": 3, "dim_atc": 8, "fact_sales_daily": 24}
    fact = tables["fact_sales_daily"]
    assert fact.groupby("atc_code")["quantity"].sum().round(2).to_dict() == {
        "M01AB": 2.0, "M01AE": 4.67, "N02BA": 9.9, "N02BE": 94.25, "N05B": 17.0, "N05C": 0.0, "R03": 9.0, "R06": 3.0,
    }
    assert not fact.duplicated(["date", "atc_code"]).any()
    dates = tables["dim_date"].set_index("date")
    saturday = pd.Timestamp("2014-01-04").date()
    assert dates.loc[saturday, "weekday"] == "Saturday" and dates.loc[saturday, "is_weekend"]
    assert not dates.loc[pd.Timestamp("2014-01-02").date(), "is_weekend"]
    assert dates["quarter"].tolist() == [1, 1, 1]
    assert any("zero sales in every category: 1" in note for note in report.notes)


def test_tiny_dim_atc():
    tables, _ = build_tables(tiny_wide())
    atc = tables["dim_atc"].set_index("atc_code")
    assert list(atc.index) == ATC_CODES
    assert atc.loc["N05B", "description"] == "Anxiolytics"
    assert atc.loc["N02BE", "therapeutic_group"] == "Analgesics"
    assert atc["therapeutic_group"].nunique() == 5


def test_loader_rejects_unexpected_columns(tmp_path):
    path = tmp_path / "salesdaily.csv"
    tiny_wide().rename(columns={"datum": "date"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unexpected columns"):
        read_raw(path)


def test_loader_rejects_wrong_date_format():
    wide = tiny_wide()
    wide.loc[0, "Weekday Name"] = "Monday"
    with pytest.raises(ValueError, match="Weekday Name"):
        build_tables(wide)


# ---------------------------------------------------------------- the real file


@pytest.fixture(scope="module")
def pharmacy_db(tmp_path_factory):
    if not RAW_PATH.exists():
        pytest.skip("salesdaily.csv not in data/raw/")
    tables, report = build_tables(read_raw(RAW_PATH))
    return write_db(tables, tmp_path_factory.mktemp("pharmacy") / "pharmacy.duckdb"), report


@pytest.fixture(scope="module")
def reference():
    if not RAW_PATH.exists():
        pytest.skip("salesdaily.csv not in data/raw/")
    return answers(load_wide(RAW_PATH))


@needs_raw
def test_real_row_counts(pharmacy_db):
    db, report = pharmacy_db
    assert report.raw_rows == 2_106
    assert report.table_rows == {"dim_date": 2_106, "dim_atc": 8, "fact_sales_daily": 16_848}
    span = run_query("SELECT MIN(date) AS lo, MAX(date) AS hi FROM dim_date", db).frame
    assert (str(span["lo"][0])[:10], str(span["hi"][0])[:10]) == ("2014-01-02", "2019-10-08")


@needs_raw
def test_real_keys_and_exact_totals(pharmacy_db):
    db, _ = pharmacy_db
    keys = run_query(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT (date, atc_code)) AS k, typeof(ANY_VALUE(quantity)) AS t FROM fact_sales_daily", db
    ).frame
    assert keys["n"][0] == keys["k"][0] == 16_848 and keys["t"][0] == "DECIMAL(18,9)"
    orphans = run_query(
        "SELECT COUNT(*) AS n FROM fact_sales_daily f LEFT JOIN dim_date d ON f.date = d.date "
        "LEFT JOIN dim_atc a ON f.atc_code = a.atc_code WHERE d.date IS NULL OR a.atc_code IS NULL", db
    ).frame
    assert orphans["n"][0] == 0
    wide = load_wide(RAW_PATH)
    totals = run_query("SELECT atc_code, SUM(quantity) AS q FROM fact_sales_daily GROUP BY atc_code", db).frame
    for row in totals.itertuples():
        assert abs(float(row.q) - float(wide[row.atc_code].sum())) < 1e-6, row.atc_code


# ---------------------------------------------------------------- gold set


@pytest.fixture(scope="module")
def gold():
    return load_gold(PHARMACY.gold_path)


def test_pharmacy_gold_shape(gold):
    assert len(gold) == 10
    assert pd.Series([g["difficulty"] for g in gold]).value_counts().to_dict() == {"easy": 4, "medium": 4, "hard": 2}
    assert len({g["id"] for g in gold}) == len({g["question"] for g in gold}) == 10


def test_pharmacy_gold_not_in_few_shot(gold):
    shots = {e["question"].lower() for e in PHARMACY.examples}
    shot_sql = {e["sql"].strip().lower() for e in PHARMACY.examples}
    for item in gold:
        assert item["question"].lower() not in shots and item["sql"].strip().lower() not in shot_sql


def test_pharmacy_gold_passes_guardrails(gold):
    for item in gold:
        assert guardrails.validate(item["sql"], allowed_tables=PHARMACY.allowed_tables).ok, item["id"]


@needs_raw
def test_pharmacy_gold_matches_pandas(gold, pharmacy_db, reference):
    db, _ = pharmacy_db
    for item in gold:
        outcome = run_query(item["sql"], db)
        assert outcome.ok, (item["id"], outcome.error)
        assert len(outcome.frame) > 0, item["id"]
        assert results_match(outcome.frame, reference[item["id"]]), item["id"]


@needs_raw
def test_pharmacy_examples_run(pharmacy_db):
    db, _ = pharmacy_db
    for example in PHARMACY.examples:
        assert guardrails.validate(example["sql"], allowed_tables=PHARMACY.allowed_tables).ok
        outcome = run_query(example["sql"], db)
        assert outcome.ok and len(outcome.frame) > 0, example["question"]


@needs_raw
def test_pharmacy_evaluation_with_oracle_llm(gold, pharmacy_db):
    db, _ = pharmacy_db
    llm = ScriptedLLM({g["question"]: g["sql"] for g in gold})
    records = run_evaluation(gold, llm, None, ["few_shot_full_schema"], db, dataset=PHARMACY)
    assert len(records) == 10 and records["correct"].all()


# ---------------------------------------------------------------- dataset separation


def test_retail_and_pharmacy_share_no_tables():
    assert RETAIL.allowed_tables.isdisjoint(PHARMACY.allowed_tables)


@pytest.mark.parametrize("a, b", list(itertools.combinations(sorted(DATASETS), 2)))
def test_shared_table_names_are_only_the_known_ones(a, b):
    shared = DATASETS[a].allowed_tables & DATASETS[b].allowed_tables
    assert shared == KNOWN_SHARED_NAMES.get(frozenset({a, b}), set())


@pytest.mark.parametrize("key", sorted(DATASETS))
def test_allowlist_is_exactly_own_schema(key):
    dataset = DATASETS[key]
    assert dataset.allowed_tables == frozenset(dataset.schema().table_names)


@pytest.mark.parametrize("own, other", list(itertools.permutations(sorted(DATASETS), 2)))
def test_allowlist_blocks_other_datasets_tables(own, other):
    foreign = DATASETS[other].allowed_tables - DATASETS[own].allowed_tables
    assert foreign
    for table in foreign:
        verdict = guardrails.validate(f"SELECT * FROM {table}", allowed_tables=DATASETS[own].allowed_tables)
        assert not verdict.ok and "table not allowed" in verdict.reason, (own, table)


@pytest.mark.parametrize("table", ["fact_sales", "invoices", "dim_customer", "targets"])
def test_pipeline_blocks_foreign_tables_in_pharmacy_mode(table, tmp_path):
    llm = ScriptedLLM({"q": f"SELECT COUNT(*) FROM {table}"})
    answer = answer_question("q", "few_shot_full_schema", llm, None, tmp_path / "x.duckdb", None, dataset=PHARMACY)
    assert answer.status == "blocked" and table in answer.verdict.reason


def test_each_dataset_gets_its_own_chroma_collection(monkeypatch):
    seen = {}

    class FakeChroma:
        name = "chroma"

        def __init__(self, schema, collection_name="schema_tables"):
            seen[tuple(sorted(schema.table_names))] = collection_name

    monkeypatch.setattr(retrieval, "ChromaRetriever", FakeChroma)
    for dataset in DATASETS.values():
        retrieval.get_retriever("chroma", dataset=dataset)
    assert sorted(seen.values()) == ["schema_tables", "schema_tables_pharmacy", "schema_tables_retail"]
    assert seen[tuple(sorted(PHARMACY.allowed_tables))] == "schema_tables_pharmacy"
    assert seen[tuple(sorted(PHARMA.allowed_tables))] == "schema_tables"


def test_pharmacy_prompt_has_only_pharmacy_schema():
    from src.nl2sql import build_prompt

    prompt = build_prompt(
        "q", "few_shot_full_schema", schema=PHARMACY.schema(), examples=list(PHARMACY.examples), domain=PHARMACY.domain
    )
    assert "TABLE fact_sales_daily" in prompt and "not revenue" in prompt
    for foreign in ("fact_sales ", "dim_hcp", "invoice_lines", "dim_customer", "territory"):
        assert foreign not in prompt


# ---------------------------------------------------------------- stopping on quota errors


class QuotaLLM:
    name, model = "fake", "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED: You exceeded your current quota")


def test_quota_error_is_not_retried_when_disabled():
    sleeps = []
    inner = QuotaLLM()
    with pytest.raises(RuntimeError):
        RetryingLLM(inner, sleep=sleeps.append, retry_quota=False).complete("p")
    assert inner.calls == 1 and sleeps == []
    assert is_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED")) and not is_quota_error(RuntimeError("503 UNAVAILABLE"))


def test_evaluation_stops_at_first_quota_error(gold, tmp_path):
    db = write_db(build_tables(tiny_wide())[0], tmp_path / "tiny.duckdb")
    inner = QuotaLLM()
    stop = lambda answer: "quota" if answer.error and is_quota_error(answer.error) else None
    with pytest.raises(EvaluationStopped) as stopped:
        run_evaluation(gold, inner, None, ["few_shot_full_schema"], db, dataset=PHARMACY, stop_if=stop)
    assert inner.calls == 1 and len(stopped.value.records) == 1
