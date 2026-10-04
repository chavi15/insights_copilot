import pytest

from src import sql_optimizer as so
from src.schema_assistant import DIALECTS


class FakeLLM:
    def __init__(self, text="- Add an index on employees(dept_id)."):
        self.text = text
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.text


def rules_for(sql, dialect="oracle"):
    analysis = so.analyze(sql, dialect)
    assert analysis.parse_error is None, analysis.parse_error
    return {f.rule for f in analysis.findings}


# (rule, query that triggers it, similar query that does not)
RULE_CASES = [
    (
        "SELECT *",
        "SELECT * FROM employees",
        "SELECT e.emp_id FROM employees e WHERE EXISTS (SELECT * FROM departments d WHERE d.dept_id = e.dept_id)",
    ),
    (
        "SELECT *",
        "SELECT e.* FROM employees e",
        "SELECT COUNT(*) AS n FROM employees",
    ),
    (
        "Function on a column in WHERE",
        "SELECT emp_id FROM employees WHERE UPPER(name) = 'SMITH'",
        "SELECT emp_id FROM employees WHERE name = UPPER('smith')",
    ),
    (
        "Function on a column in WHERE",
        "SELECT emp_id FROM employees WHERE TRUNC(hire_date) = DATE '2024-01-01'",
        "SELECT emp_id FROM employees WHERE hire_date >= TRUNC(SYSDATE) - 30",
    ),
    (
        "Leading-wildcard LIKE",
        "SELECT emp_id FROM employees WHERE name LIKE '%son'",
        "SELECT emp_id FROM employees WHERE name LIKE 'son%'",
    ),
    (
        "Missing join condition",
        "SELECT e.name, d.dept_name FROM employees e, departments d",
        "SELECT e.name, d.dept_name FROM employees e, departments d WHERE d.dept_id = e.dept_id",
    ),
    (
        "Missing join condition",
        "SELECT e.name, d.dept_name FROM employees e JOIN departments d ON d.dept_id = e.dept_id, locations l",
        "SELECT e.name, l.city FROM employees e CROSS JOIN locations l",
    ),
    (
        "NOT IN with a subquery",
        "SELECT emp_id FROM employees WHERE dept_id NOT IN (SELECT dept_id FROM closed_departments)",
        "SELECT emp_id FROM employees WHERE dept_id NOT IN (10, 20)",
    ),
    (
        "Correlated subquery",
        "SELECT e.name, (SELECT MAX(s.amount) FROM sales s WHERE s.emp_id = e.emp_id) AS top_sale FROM employees e",
        "SELECT e.name FROM employees e WHERE e.dept_id IN (SELECT d.dept_id FROM departments d WHERE d.city = 'Pune')",
    ),
    (
        "Correlated subquery",
        "SELECT e.name FROM employees e WHERE e.salary > (SELECT AVG(x.salary) FROM employees x WHERE x.dept_id = e.dept_id)",
        "SELECT e.name FROM employees e WHERE NOT EXISTS (SELECT 1 FROM closed c WHERE c.dept_id = e.dept_id)",
    ),
    (
        "DISTINCT with joins",
        "SELECT DISTINCT d.dept_name FROM departments d JOIN employees e ON e.dept_id = d.dept_id",
        "SELECT DISTINCT dept_id FROM employees",
    ),
    (
        "ORDER BY in a subquery without LIMIT/FETCH",
        "SELECT name FROM (SELECT name FROM employees ORDER BY salary DESC) x",
        "SELECT name FROM (SELECT name FROM employees ORDER BY salary DESC FETCH FIRST 5 ROWS ONLY) x",
    ),
    (
        "ORDER BY in a subquery without LIMIT/FETCH",
        "WITH ranked AS (SELECT name FROM employees ORDER BY name) SELECT name FROM ranked",
        "SELECT name FROM (SELECT name FROM employees ORDER BY salary DESC) WHERE ROWNUM <= 5",
    ),
    (
        "Implicit type conversion",
        "SELECT name FROM employees WHERE emp_id = '42'",
        "SELECT name FROM employees WHERE emp_id = 42",
    ),
    (
        "Implicit type conversion",
        "SELECT name FROM employees WHERE hire_date > '2024-01-01'",
        "SELECT name FROM employees WHERE hire_date > DATE '2024-01-01'",
    ),
]


@pytest.mark.parametrize("rule, bad, good", RULE_CASES)
def test_rule_triggers(rule, bad, good):
    assert rule in rules_for(bad)


@pytest.mark.parametrize("rule, bad, good", RULE_CASES)
def test_rule_does_not_trigger(rule, bad, good):
    assert rule not in rules_for(good)


def test_every_rule_has_cases():
    names = {
        "SELECT *", "Function on a column in WHERE", "Leading-wildcard LIKE", "Missing join condition",
        "NOT IN with a subquery", "Correlated subquery", "DISTINCT with joins",
        "ORDER BY in a subquery without LIMIT/FETCH", "Implicit type conversion",
    }
    assert {case[0] for case in RULE_CASES} == names
    assert len(so.RULES) == len(names)


def test_clean_query_has_no_findings():
    sql = (
        "SELECT e.emp_id, e.name FROM employees e JOIN departments d ON d.dept_id = e.dept_id "
        "WHERE e.hire_date >= DATE '2024-01-01' AND e.name LIKE 'Sm%' "
        "AND NOT EXISTS (SELECT 1 FROM closed c WHERE c.dept_id = e.dept_id) ORDER BY e.name"
    )
    assert rules_for(sql) == set()


@pytest.mark.parametrize("dialect", list(DIALECTS.values()))
def test_rules_run_in_every_dialect(dialect):
    assert "SELECT *" in rules_for("SELECT * FROM employees", dialect)


def test_findings_are_sorted_and_complete():
    analysis = so.analyze(
        "SELECT * FROM employees e, departments d WHERE e.name LIKE '%x' AND e.dept_id NOT IN (SELECT dept_id FROM c)",
        "oracle",
    )
    severities = [f.severity for f in analysis.findings]
    assert severities == sorted(severities, key=so.SEVERITY_ORDER.get)
    for finding in analysis.findings:
        assert finding.severity in so.SEVERITY_ORDER
        assert finding.reason and finding.suggestion and finding.snippet


def test_parse_failure_is_reported_not_raised():
    analysis = so.analyze("SELECT name FROM employees WHERE (salary > 1", "oracle")
    assert analysis.findings == []
    assert analysis.parse_error


def test_analyze_never_calls_a_model():
    # analyze takes no llm at all; this guards against someone adding one.
    import inspect

    assert "llm" not in inspect.signature(so.analyze).parameters


# ---------------------------------------------------------------- literal masking


@pytest.mark.parametrize(
    "dialect, sql, expected",
    [
        ("oracle", "SELECT 1 FROM t WHERE name = 'O''Brien' AND city = 'Pune'", "SELECT 1 FROM t WHERE name = ? AND city = ?"),
        ("oracle", "SELECT q'[it's secret]' FROM dual", "SELECT ? FROM dual"),
        ("oracle", "SELECT \"Weird'Name\" FROM t WHERE x = 'a'", "SELECT \"Weird'Name\" FROM t WHERE x = ?"),
        ("postgres", "SELECT $$secret$$, $tag$also$tag$, E'it\\'s' FROM t", "SELECT ?, ?, E? FROM t"),
        ("postgres", "SELECT * FROM t WHERE id = $1", "SELECT * FROM t WHERE id = $1"),
        ("mysql", "SELECT * FROM t WHERE a = \"secret\" AND b = 'it\\'s'", "SELECT * FROM t WHERE a = ? AND b = ?"),
        ("mysql", "SELECT `it's col` FROM t", "SELECT `it's col` FROM t"),
        ("duckdb", "SELECT 1 -- note: account 1234\nFROM t /* client: ACME */ WHERE k = 'v'", "SELECT 1  \nFROM t   WHERE k = ?"),
    ],
)
def test_mask_literals(dialect, sql, expected):
    assert so.mask_literals(sql, dialect) == expected


def test_mask_keeps_numbers():
    assert so.mask_literals("SELECT * FROM t WHERE id = 42", "oracle") == "SELECT * FROM t WHERE id = 42"


@pytest.mark.parametrize("sql", ["SELECT 'open", "SELECT 1 /* open", "SELECT q'[open"])
def test_mask_rejects_unterminated(sql):
    with pytest.raises(ValueError, match="unterminated"):
        so.mask_literals(sql, "oracle")


def test_plan_masking_is_lenient_and_keeps_layout():
    plan = "|* 2 | TABLE ACCESS FULL | EMPLOYEES |\n------------\n 2 - filter(\"E\".\"NAME\"='Smith')\n odd 'tail"
    masked = so.mask_literals(plan, None, strict=False)
    assert "Smith" not in masked and "tail" not in masked
    assert "------------" in masked and '"E"."NAME"=?' in masked


# ---------------------------------------------------------------- length limits


def test_query_length_limit():
    at_limit = "SELECT 1 FROM t" + " " * (so.MAX_QUERY_CHARS - 15)
    so.analyze(at_limit, "oracle")
    with pytest.raises(ValueError, match="too long"):
        so.analyze(at_limit + " ", "oracle")
    with pytest.raises(ValueError, match="Paste a SQL query"):
        so.analyze("   ", "oracle")


def test_plan_length_limit():
    llm = FakeLLM()
    so.suggest("SELECT 1 FROM t", "Oracle", llm, plan="x" * so.MAX_PLAN_CHARS)
    with pytest.raises(ValueError, match="plan text is too long"):
        so.suggest("SELECT 1 FROM t", "Oracle", llm, plan="x" * (so.MAX_PLAN_CHARS + 1))
    assert len(llm.prompts) == 1


def test_query_limit_applies_before_model_call():
    llm = FakeLLM()
    with pytest.raises(ValueError):
        so.suggest("SELECT 1" + " " * so.MAX_QUERY_CHARS, "Oracle", llm)
    assert llm.prompts == []


# ---------------------------------------------------------------- model suggestions


def test_suggest_sends_only_masked_query_names_and_plan():
    llm = FakeLLM()
    schemas = [
        ("HR", "CREATE TABLE employees (emp_id NUMBER(6), name VARCHAR2(100) DEFAULT 'n/a', dept_id NUMBER)"),
        ("SALES", "employees: emp_id, region"),
    ]
    result = so.suggest(
        "SELECT name FROM HR.employees WHERE name = 'Smith' -- VIP client",
        "Oracle",
        llm,
        schemas=schemas,
        plan="filter(\"NAME\"='Smith')",
    )
    (prompt,) = llm.prompts
    assert "Write SQL for Oracle." in prompt
    assert "HR.employees(emp_id, name, dept_id)" in prompt
    assert "SALES.employees(emp_id, region)" in prompt
    assert "WHERE name = ?" in prompt
    for leaked in ("Smith", "VIP", "NUMBER", "VARCHAR2", "n/a"):
        assert leaked not in prompt
    assert result.query_sent == "SELECT name FROM HR.employees WHERE name = ?"
    assert result.plan_sent == 'filter("NAME"=?)'
    assert result.text == llm.text


def test_suggest_without_schema_or_plan():
    llm = FakeLLM()
    result = so.suggest("SELECT * FROM t", "PostgreSQL", llm, schemas=[("", "")])
    assert result.schema_sent == "" and result.plan_sent == ""
    assert "TABLES" not in llm.prompts[0] and "EXECUTION PLAN" not in llm.prompts[0]


def test_suggest_rejects_unknown_dialect():
    with pytest.raises(ValueError):
        so.suggest("SELECT 1", "SQLite", FakeLLM())


def test_warning_and_label_text():
    assert "Gemini" in so.OPTIMIZER_WARNING and "?" in so.OPTIMIZER_WARNING
    assert so.SUGGESTIONS_LABEL == "Unverified suggestions: compare plans on your own database before using."
