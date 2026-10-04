import pandas as pd
import pytest

from src import guardrails
from src.datasets import DATASETS, PHARMA, RETAIL
from src.evaluation import load_gold, results_match, run_evaluation
from src.executor import run_query
from src.llm import ScriptedLLM
from src.nl2sql import build_prompt
from src.pipeline import answer_question
from src.retail_load import RAW_PATH, build_tables, read_raw, write_db
from src.retrieval import get_retriever
from tests.retail_reference import answers, clean_raw

needs_raw = pytest.mark.skipif(not RAW_PATH.exists(), reason="run python -m src.retail_load to download the raw file")


def _row(invoice, code, qty, price, customer, country="United Kingdom", when="2011-01-03 10:00", desc="MUG"):
    return {
        "InvoiceNo": invoice, "StockCode": code, "Description": desc, "Quantity": qty,
        "InvoiceDate": pd.Timestamp(when), "UnitPrice": price, "CustomerID": customer, "Country": country,
    }


@pytest.fixture(scope="module")
def tiny():
    raw = pd.DataFrame(
        [
            _row("100", "85123A", 6, 2.5, 17850.0, desc=" MUG "),
            _row("100", "85123A", 6, 2.5, 17850.0, desc=" MUG "),  # exact duplicate
            _row("100", "85123a", 2, 2.5, 17850.0, when="2011-01-03 10:01"),  # lower-case code, later timestamp
            _row("101", "22197", 10, 0.85, None, country="France", desc="POPCORN HOLDER"),  # missing customer
            _row("C102", "85123A", -3, 2.5, 17850.0, desc="MUG"),  # cancellation
            _row("103", "22197", -20, 0.0, None, desc="damaged"),  # write-off
            _row("104", "22197", 5, 0.0, 12345.0, desc="check"),  # free item
            _row("A105", "B", 1, -11062.06, None, desc="Adjust bad debt"),  # adjustment
            _row("106", "22197", 1, 0.85, 17850.0, country="EIRE", desc="POPCORN HOLDER"),
        ]
    )
    return build_tables(raw)


def test_tiny_cleaning_steps(tiny):
    tables, report = tiny
    steps = {decision.split(" (")[0]: count for decision, count, _, _ in report.steps}
    assert report.raw_rows == 9
    assert steps["exact duplicate rows"] == 1
    assert steps["bad-debt adjustment invoices"] == 1
    assert steps["zero or negative unit price"] == 2
    assert steps["cancellation lines"] == 1
    assert steps["lines with a missing CustomerID"] == 1
    assert steps["invoices with a missing CustomerID"] == 1
    assert steps["stock codes with lower-case letters"] == 1
    assert report.table_rows == {"dim_customer": 1, "dim_product": 2, "invoices": 4, "invoice_lines": 5}


def test_tiny_tables(tiny):
    tables, _ = tiny
    invoices = tables["invoices"].set_index("invoice_no")
    assert set(invoices.index) == {"100", "101", "C102", "106"}
    assert invoices.loc["C102", "is_cancelled"] and not invoices.loc["100", "is_cancelled"]
    assert invoices.loc["100", "invoice_date"] == pd.Timestamp("2011-01-03 10:00")
    assert pd.isna(invoices.loc["101", "customer_id"])
    assert tables["dim_product"].set_index("stock_code")["description"].to_dict() == {"22197": "POPCORN HOLDER", "85123A": "MUG"}
    # 17850 bought twice from the UK and once from EIRE, so the UK wins.
    assert tables["dim_customer"].to_dict("records") == [{"customer_id": 17850, "country": "United Kingdom"}]
    lines = tables["invoice_lines"]
    assert set(lines["stock_code"]) == {"85123A", "22197"}
    assert lines.loc[lines["invoice_no"] == "C102", "line_total"].tolist() == [-7.5]
    assert (lines["unit_price"] > 0).all()


def test_tiny_rejects_negative_quantity_outside_cancellations():
    raw = pd.DataFrame([_row("200", "22197", -1, 0.85, 1.0)])
    with pytest.raises(ValueError, match="negative quantities"):
        build_tables(raw)


# ---------------------------------------------------------------- the real file


@pytest.fixture(scope="session")
def raw():
    if not RAW_PATH.exists():
        pytest.skip("raw Online Retail file not downloaded")
    return read_raw(RAW_PATH)


@pytest.fixture(scope="session")
def built(raw):
    return build_tables(raw)


@pytest.fixture(scope="session")
def retail_db(built, tmp_path_factory):
    return write_db(built[0], tmp_path_factory.mktemp("retail") / "retail.duckdb")


@pytest.fixture(scope="session")
def reference(raw):
    return answers(clean_raw(raw))


@needs_raw
def test_real_row_counts(built):
    tables, report = built
    assert report.raw_rows == 541_909
    assert report.table_rows == {"dim_customer": 4_371, "dim_product": 3_827, "invoices": 23_795, "invoice_lines": 534_128}
    removed = sum(count for decision, count, unit, action in report.steps if action.startswith("removed"))
    assert report.raw_rows - removed == report.table_rows["invoice_lines"]


@needs_raw
def test_real_no_duplicate_keys(retail_db):
    for table, key in [("dim_customer", "customer_id"), ("dim_product", "stock_code"), ("invoices", "invoice_no")]:
        frame = run_query(f"SELECT COUNT(*) AS n, COUNT(DISTINCT {key}) AS d, COUNT({key}) AS nn FROM {table}", retail_db).frame
        assert frame["n"][0] == frame["d"][0] == frame["nn"][0], table


@needs_raw
def test_real_cancellations_flagged(retail_db):
    checks = {
        "flag matches C prefix": "SELECT COUNT(*) FROM invoices WHERE is_cancelled <> starts_with(invoice_no, 'C')",
        "cancelled lines are negative": "SELECT COUNT(*) FROM invoice_lines l JOIN invoices i USING (invoice_no) WHERE i.is_cancelled AND l.quantity >= 0",
        "other lines are positive": "SELECT COUNT(*) FROM invoice_lines l JOIN invoices i USING (invoice_no) WHERE NOT i.is_cancelled AND l.quantity <= 0",
        "prices above zero": "SELECT COUNT(*) FROM invoice_lines WHERE unit_price <= 0",
        "lines have invoices": "SELECT COUNT(*) FROM invoice_lines l LEFT JOIN invoices i USING (invoice_no) WHERE i.invoice_no IS NULL",
        "lines have products": "SELECT COUNT(*) FROM invoice_lines l LEFT JOIN dim_product p USING (stock_code) WHERE p.stock_code IS NULL",
        "customers exist": "SELECT COUNT(*) FROM invoices i LEFT JOIN dim_customer c USING (customer_id) WHERE i.customer_id IS NOT NULL AND c.customer_id IS NULL",
    }
    for name, sql in checks.items():
        assert run_query(sql, retail_db).frame.iloc[0, 0] == 0, name
    counts = run_query(
        "SELECT COUNT(*) FILTER (WHERE is_cancelled) AS c, COUNT(*) FILTER (WHERE customer_id IS NULL) AS n FROM invoices", retail_db
    ).frame
    assert (counts["c"][0], counts["n"][0]) == (3_836, 1_609)


# ---------------------------------------------------------------- retail gold set


@pytest.fixture(scope="module")
def gold():
    return load_gold(RETAIL.gold_path)


def test_retail_gold_shape(gold):
    assert len(gold) == 10
    assert pd.Series([g["difficulty"] for g in gold]).value_counts().to_dict() == {"easy": 4, "medium": 4, "hard": 2}
    assert len({g["id"] for g in gold}) == len({g["question"] for g in gold}) == 10


def test_retail_gold_not_in_few_shot(gold):
    shots = {e["question"].lower() for e in RETAIL.examples}
    shot_sql = {e["sql"].strip().lower() for e in RETAIL.examples}
    for item in gold:
        assert item["question"].lower() not in shots and item["sql"].strip().lower() not in shot_sql


def test_retail_gold_passes_guardrails(gold):
    for item in gold:
        assert guardrails.validate(item["sql"], allowed_tables=RETAIL.allowed_tables).ok, item["id"]


@needs_raw
def test_retail_gold_matches_pandas(gold, retail_db, reference):
    for item in gold:
        outcome = run_query(item["sql"], retail_db)
        assert outcome.ok, (item["id"], outcome.error)
        assert len(outcome.frame) > 0, item["id"]
        assert results_match(outcome.frame, reference[item["id"]]), item["id"]


@needs_raw
def test_retail_examples_run(retail_db):
    for example in RETAIL.examples:
        outcome = run_query(example["sql"], retail_db)
        assert outcome.ok and len(outcome.frame) > 0, example["question"]


@needs_raw
def test_retail_evaluation_with_oracle_llm(gold, retail_db):
    llm = ScriptedLLM({g["question"]: g["sql"] for g in gold})
    records = run_evaluation(gold, llm, None, ["few_shot_full_schema"], retail_db, dataset=RETAIL)
    assert len(records) == 10 and records["correct"].all()


# ---------------------------------------------------------------- dataset separation


def test_allowlists_are_disjoint_except_dim_product():
    assert PHARMA.allowed_tables & RETAIL.allowed_tables == {"dim_product"}
    assert sorted(RETAIL.schema().table_names) == sorted(RETAIL.allowed_tables)
    assert sorted(PHARMA.schema().table_names) == sorted(PHARMA.allowed_tables)


@pytest.mark.parametrize("table", sorted(PHARMA.allowed_tables - {"dim_product"}))
def test_retail_allowlist_blocks_pharma_tables(table):
    verdict = guardrails.validate(f"SELECT * FROM {table}", allowed_tables=RETAIL.allowed_tables)
    assert not verdict.ok and "table not allowed" in verdict.reason


@pytest.mark.parametrize("table", sorted(RETAIL.allowed_tables - {"dim_product"}))
def test_pharma_allowlist_blocks_retail_tables(table):
    verdict = guardrails.validate(f"SELECT * FROM {table}", allowed_tables=PHARMA.allowed_tables)
    assert not verdict.ok and "table not allowed" in verdict.reason


def test_pipeline_uses_dataset_allowlist(tmp_path):
    llm = ScriptedLLM({"pharma table": "SELECT COUNT(*) FROM fact_sales"})
    answer = answer_question("pharma table", "few_shot_full_schema", llm, None, tmp_path / "x.duckdb", None, dataset=RETAIL)
    assert answer.status == "blocked" and "fact_sales" in answer.verdict.reason


def test_retail_prompt_has_only_retail_schema_and_examples():
    prompt = build_prompt("q", "few_shot_full_schema", schema=RETAIL.schema(), examples=list(RETAIL.examples), domain=RETAIL.domain)
    assert "TABLE invoice_lines" in prompt and "UK online gift retailer" in prompt
    for pharma_word in ("fact_sales", "dim_hcp", "pharma", "therapeutic"):
        assert pharma_word not in prompt
    assert build_prompt("q", "few_shot_full_schema") == build_prompt(
        "q", "few_shot_full_schema", schema=PHARMA.schema(), examples=list(PHARMA.examples), domain=PHARMA.domain
    )


@pytest.mark.parametrize("key", sorted(DATASETS))
def test_keyword_retriever_stays_in_its_dataset(key):
    dataset = DATASETS[key]
    retriever = get_retriever("keyword", dataset=dataset)
    for question in ("total revenue by country", "cancelled invoices per customer", "calls per HCP tier"):
        assert set(retriever.tables_for(question)) <= dataset.allowed_tables
