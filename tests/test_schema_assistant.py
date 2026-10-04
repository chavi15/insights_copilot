import pytest

from src import schema_assistant as sa
from src.llm import CachedLLM

ORACLE_DDL = """
-- HR schema, internal
CREATE TABLE hr.employees (
    employee_id NUMBER(6) PRIMARY KEY,
    full_name   VARCHAR2(100) DEFAULT 'top, secret' NOT NULL, /* salary band 7 */
    dept_id     NUMBER(4),
    salary      NUMBER(10, 2) CHECK (salary > 0),
    CONSTRAINT emp_dept_fk FOREIGN KEY (dept_id) REFERENCES hr.departments (dept_id)
) TABLESPACE users STORAGE (INITIAL 64K);

CREATE TABLE IF NOT EXISTS "Departments" ("Dept Id" INT, `name` TEXT COMMENT 'hidden note', UNIQUE (name));
"""


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.text


def test_parse_create_table_keeps_names_only():
    tables = sa.parse_schema(ORACLE_DDL)
    assert tables == {
        "hr.employees": ["employee_id", "full_name", "dept_id", "salary"],
        '"Departments"': ['"Dept Id"', "`name`"],
    }
    rendered = sa.render_schema(tables)
    for leaked in ("NUMBER", "VARCHAR2", "secret", "salary band", "hidden note", "TABLESPACE", "64K", "internal"):
        assert leaked not in rendered


@pytest.mark.parametrize(
    "paste",
    [
        "orders: order_id, customer_id, total\ncustomers: customer_id, region",
        "orders(order_id, customer_id, total)\ncustomers(customer_id, region)",
        "orders.order_id\norders.customer_id\norders.total\ncustomers.customer_id\ncustomers.region",
    ],
)
def test_parse_column_lists(paste):
    assert sa.parse_schema(paste) == {
        "orders": ["order_id", "customer_id", "total"],
        "customers": ["customer_id", "region"],
    }


@pytest.mark.parametrize(
    "paste, message",
    [
        ("", "at least one"),
        ("   ", "at least one"),
        ("x" * (sa.MAX_SCHEMA_CHARS + 1), "too long"),
        ("just some prose about my data", "No tables"),
        ("CREATE TABLE t (a INT DEFAULT 'oops)", "unterminated"),
    ],
)
def test_parse_rejects_bad_input(paste, message):
    with pytest.raises(ValueError, match=message):
        sa.parse_schema(paste)


def test_schema_at_limit_is_accepted():
    paste = "t: a\n" + " " * (sa.MAX_SCHEMA_CHARS - 5)
    assert len(paste) == sa.MAX_SCHEMA_CHARS
    assert sa.parse_schema(paste) == {"t": ["a"]}


def test_dialect_options():
    assert list(sa.DIALECTS) == ["Oracle", "PostgreSQL", "MySQL", "DuckDB"]


@pytest.mark.parametrize("label", list(sa.DIALECTS))
def test_prompt_names_dialect(label):
    prompt = sa.build_prompt("How many orders?", "orders(order_id)", label)
    assert f"Write SQL for {label}." in prompt
    assert "orders(order_id)" in prompt
    assert prompt.endswith("Question: How many orders?")


@pytest.mark.parametrize(
    "sql, dialect",
    [
        ("SELECT dept_id, COUNT(*) AS n FROM hr.employees GROUP BY dept_id ORDER BY n DESC FETCH FIRST 5 ROWS ONLY", "oracle"),
        ("SELECT e.full_name FROM employees e WHERE ROWNUM <= 10", "oracle"),
        ("SELECT region, SUM(total)::numeric(12,2) AS revenue FROM orders GROUP BY region", "postgres"),
        ("SELECT `region`, SUM(total) AS revenue FROM orders GROUP BY `region` LIMIT 10", "mysql"),
        ("WITH t AS (SELECT 1 AS x) SELECT x FROM t", "duckdb"),
        ("SELECT a FROM t1 UNION ALL SELECT a FROM t2", "postgres"),
    ],
)
def test_check_allows_selects(sql, dialect):
    assert sa.check_sql(sql, dialect) == sa.SqlCheck("ok", "ok")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE employees",
        "DELETE FROM employees",
        "UPDATE employees SET salary = 0",
        "SELECT * FROM employees; DROP TABLE employees",
        "SELECT * FROM employees FOR UPDATE",
        "SELECT DBMS_LOCK.SLEEP(10) FROM dual",
        "SELECT UTL_HTTP.REQUEST('http://x') FROM dual",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT LOAD_FILE('/etc/passwd')",
        "SELECT pg_sleep(10)",
        "SELECT * INTO backup FROM employees",
        "",
    ],
)
@pytest.mark.parametrize("dialect", list(sa.DIALECTS.values()))
def test_check_blocks_dangerous_sql(sql, dialect):
    assert sa.check_sql(sql, dialect).status == "blocked"


def test_check_reports_parse_failure_without_blocking():
    check = sa.check_sql("SELECT a FROM t WHERE (a = 1", "oracle")
    assert check.status == "unparsed"
    assert check.reason


def test_write_sql_sends_names_only_and_returns_explanation():
    llm = FakeLLM(
        "```sql\nSELECT dept_id, AVG(salary) AS avg_salary FROM hr.employees GROUP BY dept_id\n```\n"
        "Explanation: Averages salary per department."
    )
    answer = sa.write_sql("Average salary per department?", ORACLE_DDL, "Oracle", llm)
    assert answer.sql.startswith("SELECT dept_id")
    assert answer.explanation == "Averages salary per department."
    assert answer.check.status == "ok"
    (prompt,) = llm.prompts
    assert "Write SQL for Oracle." in prompt
    assert "hr.employees(employee_id, full_name, dept_id, salary)" in prompt
    for leaked in ("NUMBER", "secret", "salary band", "hidden note", "TABLESPACE"):
        assert leaked not in prompt


def test_write_sql_keeps_unparseable_sql_visible():
    llm = FakeLLM("```sql\nSELECT a FROM t WHERE (a = 1\n```\nExplanation: Broken on purpose.")
    answer = sa.write_sql("q", "t: a", "PostgreSQL", llm)
    assert answer.check.status == "unparsed"
    assert answer.sql == "SELECT a FROM t WHERE (a = 1"


def test_write_sql_validates_before_calling_model():
    llm = FakeLLM("unused")
    with pytest.raises(ValueError):
        sa.write_sql("q", "t: a", "SQLite", llm)
    with pytest.raises(ValueError):
        sa.write_sql("  ", "t: a", "Oracle", llm)
    with pytest.raises(ValueError):
        sa.write_sql("q", "x" * (sa.MAX_SCHEMA_CHARS + 1), "Oracle", llm)
    assert llm.prompts == []


def test_write_sql_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import src.executor

    def fail(*args, **kwargs):
        raise AssertionError("schema mode must not log")

    monkeypatch.setattr(src.executor, "append_jsonl", fail)
    llm = FakeLLM("```sql\nSELECT a FROM t\n```\nExplanation: x")
    sa.write_sql("q", "t: a", "DuckDB", llm)
    assert list(tmp_path.iterdir()) == []


def test_schema_llm_is_uncached_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    llm = sa.build_schema_llm()
    assert not isinstance(llm, CachedLLM)
    assert llm.name == "gemini"
    assert llm.model.startswith("gemini")


def test_explanation_fallback_strips_code():
    assert sa.extract_explanation("Here you go.\n```sql\nSELECT 1\n```") == "Here you go."


# ---------------------------------------------------------------- multiple schemas

HR_DDL = "CREATE TABLE employees (emp_id NUMBER, name VARCHAR2(50), dept_id NUMBER);\nCREATE TABLE departments (dept_id NUMBER, city VARCHAR2(30));"
SALES_LIST = "employees: emp_id, region\norders: order_id, emp_id, amount"


def test_parse_schemas_qualifies_and_keeps_same_named_tables_apart():
    tables = sa.parse_schemas([("HR", HR_DDL), ("SALES", SALES_LIST)])
    assert tables == {
        "HR.employees": ["emp_id", "name", "dept_id"],
        "HR.departments": ["dept_id", "city"],
        "SALES.employees": ["emp_id", "region"],
        "SALES.orders": ["order_id", "emp_id", "amount"],
    }


def test_parse_schemas_box_name_replaces_existing_qualifier():
    assert sa.parse_schemas([("HR", "CREATE TABLE old_schema.employees (emp_id INT)")]) == {"HR.employees": ["emp_id"]}


def test_single_unnamed_schema_stays_unqualified():
    assert sa.parse_schemas([("", "t: a, b"), ("", "")]) == {"t": ["a", "b"]}


@pytest.mark.parametrize(
    "entries, message",
    [
        ([("HR", "t: a"), ("hr", "u: b")], "used twice"),
        ([("HR", "t: a"), ("", "u: b")], "Give every schema a name"),
        ([("HR", "t: a"), ("SALES", "")], "Schema SALES: Paste at least one"),
        ([("HR", "prose only")], "Schema HR: No tables"),
        ([("1HR", "t: a")], "must start with a letter"),
        ([("HR.X", "t: a")], "must start with a letter"),
        ([(f"S{i}", "t: a") for i in range(sa.MAX_SCHEMAS + 1)], "at most 5"),
        ([], "at least one"),
        ([("", ""), ("  ", "  ")], "at least one"),
    ],
)
def test_parse_schemas_rejects(entries, message):
    with pytest.raises(ValueError, match=message):
        sa.parse_schemas(entries)


def test_parse_schemas_total_limit():
    half = sa.MAX_SCHEMA_CHARS // 2
    at_limit = [("HR", "t: a\n" + " " * (half - 5)), ("SALES", "u: b\n" + " " * (half - 5))]
    assert len(sa.parse_schemas(at_limit)) == 2
    over = [("HR", at_limit[0][1] + " "), ("SALES", at_limit[1][1])]
    with pytest.raises(ValueError, match="too long together"):
        sa.parse_schemas(over)


def test_write_sql_multi_schema_prompt_and_check():
    llm = FakeLLM(
        "```sql\nSELECT e.name, o.amount FROM HR.employees e JOIN SALES.orders o ON o.emp_id = e.emp_id\n```\n"
        "Explanation: Joins HR employees to their sales orders."
    )
    answer = sa.write_sql("Order amounts per employee?", [("HR", HR_DDL), ("SALES", SALES_LIST)], "Oracle", llm)
    (prompt,) = llm.prompts
    assert "Write SQL for Oracle." in prompt
    assert "SCHEMA.table" in prompt
    assert "HR.employees(emp_id, name, dept_id)" in prompt and "SALES.employees(emp_id, region)" in prompt
    assert "VARCHAR2" not in prompt and "NUMBER" not in prompt
    assert answer.schema_sent.splitlines()[0] == "HR.employees(emp_id, name, dept_id)"
    assert answer.check.status == "ok"


def test_single_string_prompt_has_no_qualified_rule():
    llm = FakeLLM("```sql\nSELECT a FROM t\n```")
    sa.write_sql("q", "t: a", "DuckDB", llm)
    assert "SCHEMA.table" not in llm.prompts[0]


@pytest.mark.parametrize(
    "sql, dialect",
    [
        ("SELECT e.name FROM HR.employees e JOIN SALES.orders o ON o.emp_id = e.emp_id", "oracle"),
        ('SELECT "e"."name" FROM "HR"."employees" "e"', "oracle"),
        ("SELECT e.name FROM hr.employees AS e", "postgres"),
        ("SELECT e.name FROM `hr`.`employees` e", "mysql"),
        ("SELECT name FROM hr.employees WHERE dept_id IN (SELECT dept_id FROM sales.orders)", "duckdb"),
    ],
)
def test_check_accepts_schema_qualified_tables(sql, dialect):
    assert sa.check_sql(sql, dialect).status == "ok"
