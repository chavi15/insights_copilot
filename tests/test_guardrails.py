import pytest

from src.guardrails import precheck, validate

BLOCKED = [
    "DROP TABLE fact_sales",
    "DELETE FROM dim_hcp",
    "UPDATE fact_sales SET revenue = 0",
    "INSERT INTO dim_product VALUES (9, 'x', 'y', '2025-01-01')",
    "CREATE TABLE evil AS SELECT * FROM dim_hcp",
    "ALTER TABLE dim_hcp ADD COLUMN x INTEGER",
    "TRUNCATE TABLE fact_sales",
    "SELECT * FROM dim_hcp; DROP TABLE dim_hcp",
    "SELECT 1; SELECT 2",
    "COPY dim_hcp TO 'out.csv'",
    "ATTACH 'other.db' AS other",
    "INSTALL httpfs",
    "LOAD httpfs",
    "PRAGMA table_info('dim_hcp')",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('x.parquet')",
    "SELECT * FROM '/etc/passwd'",
    "SELECT * FROM glob('*')",
    "SELECT getenv('HOME')",
    "SELECT * FROM information_schema.tables",
    "WITH x AS (SELECT 1) INSERT INTO dim_hcp SELECT * FROM x",
    "SELECT * INTO backup FROM dim_hcp",
    "/* sneaky */ DROP TABLE dim_hcp",
    "SELECT 1 -- ; \n; DROP TABLE dim_hcp",
    "",
    "   ",
    "SET enable_external_access = true",
    "SELECT * FROM dim_hcp WHERE name = 'unterminated",
]

AST_ONLY_BLOCKED = [
    "SELECT * FROM secret_table",
    "SELECT * FROM main.dim_hcp",
    "SELECT * FROM customers",
]

VALID = [
    "SELECT COUNT(*) AS n FROM dim_hcp",
    "SELECT region, COUNT(*) AS n FROM dim_territory GROUP BY region ORDER BY region",
    "SELECT * FROM dim_product WHERE product_name = 'drop table'",
    "SELECT p.product_name, SUM(s.revenue) FROM fact_sales s JOIN dim_product p ON s.product_id = p.product_id GROUP BY p.product_name",
    "WITH m AS (SELECT date_id, SUM(revenue) AS r FROM fact_sales GROUP BY date_id) SELECT date_id, r, LAG(r) OVER (ORDER BY date_id) FROM m",
    "SELECT tier, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct FROM dim_hcp GROUP BY tier",
    "SELECT hcp_id FROM fact_sales UNION SELECT hcp_id FROM fact_calls",
    "-- monthly\nSELECT date_id FROM dim_date ORDER BY date_id;",
    "SELECT d.year, d.quarter, SUM(s.units) FROM fact_sales s JOIN dim_date d ON s.date_id = d.date_id WHERE d.year = 2025 GROUP BY d.year, d.quarter",
    "SELECT territory_name FROM dim_territory t WHERE EXISTS (SELECT 1 FROM dim_hcp h WHERE h.territory_id = t.territory_id)",
    "SELECT product_name FROM dim_product LIMIT 3",
    "(SELECT product_name FROM dim_product)",
]


@pytest.mark.parametrize("sql", BLOCKED)
def test_precheck_blocks_without_parser(sql):
    verdict = validate(sql, ast=False)
    assert not verdict.ok, sql


@pytest.mark.parametrize("sql", [v for v in VALID if not v.startswith("(")])
def test_precheck_allows_valid_without_parser(sql):
    assert precheck(sql) is None


def test_precheck_reports_reasons():
    assert "multiple" in precheck("SELECT 1; SELECT 2")
    assert "function" in precheck("SELECT * FROM read_csv('x')")
    assert "SELECT" in precheck("DROP TABLE x")


@pytest.mark.parametrize("sql", BLOCKED + AST_ONLY_BLOCKED)
def test_full_validator_blocks(sql):
    pytest.importorskip("sqlglot")
    assert not validate(sql).ok, sql


@pytest.mark.parametrize("sql", VALID)
def test_full_validator_allows(sql):
    pytest.importorskip("sqlglot")
    verdict = validate(sql)
    assert verdict.ok, (sql, verdict.reason)


def test_full_validator_adds_limit():
    pytest.importorskip("sqlglot")
    verdict = validate("SELECT product_name FROM dim_product")
    assert verdict.ok and "LIMIT 1000" in verdict.sql.upper()


def test_full_validator_caps_large_limit():
    pytest.importorskip("sqlglot")
    verdict = validate("SELECT product_name FROM dim_product LIMIT 999999")
    assert verdict.ok and "999999" not in verdict.sql and "1000" in verdict.sql


def test_full_validator_keeps_small_limit():
    pytest.importorskip("sqlglot")
    verdict = validate("SELECT product_name FROM dim_product LIMIT 3")
    assert verdict.ok and "LIMIT 3" in verdict.sql.upper()


def test_full_validator_wraps_set_operations():
    pytest.importorskip("sqlglot")
    verdict = validate("SELECT hcp_id FROM fact_sales UNION SELECT hcp_id FROM fact_calls")
    assert verdict.ok and "LIMIT 1000" in verdict.sql.upper()


def test_full_validator_allows_cte_names():
    pytest.importorskip("sqlglot")
    assert validate("WITH tmp AS (SELECT 1 AS a) SELECT a FROM tmp").ok


def test_full_validator_rejects_unknown_table():
    pytest.importorskip("sqlglot")
    verdict = validate("SELECT * FROM customers")
    assert not verdict.ok and "not allowed" in verdict.reason


def test_full_validator_rejects_schema_qualified_table():
    pytest.importorskip("sqlglot")
    assert not validate("SELECT * FROM main.dim_hcp").ok
