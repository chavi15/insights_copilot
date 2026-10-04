import re
from dataclasses import dataclass

from src.config import MAX_ROWS

ALLOWED_TABLES = frozenset(
    {"dim_product", "dim_territory", "dim_hcp", "dim_date", "fact_sales", "fact_calls", "targets"}
)

BLOCKED_WORDS = frozenset(
    {
        "attach", "detach", "copy", "pragma", "install", "load", "export", "import", "insert", "update",
        "delete", "drop", "create", "alter", "truncate", "merge", "call", "checkpoint", "vacuum", "grant",
        "revoke", "begin", "commit", "rollback", "use", "set", "reset", "describe", "show", "explain",
        "summarize", "force", "into", "returning", "analyze", "prepare", "execute", "install_extension",
    }
)

BLOCKED_FUNCTIONS = frozenset(
    {
        "glob", "getenv", "current_setting", "query", "query_table", "sniff_csv", "httpfs", "load_extension",
        "read_blob", "read_text", "write_csv", "write_parquet", "system", "shell", "copy_to", "to_parquet",
        "to_csv", "list_files", "which_secret",
    }
)
SYSTEM_CATALOGS = frozenset({"information_schema", "pg_catalog", "sqlite_master", "sqlite_schema", "sqlite_temp_master"})
BLOCKED_FUNCTION_PREFIXES = ("read_", "duckdb_", "pragma_", "sqlite_", "glob_", "parquet_", "csv_", "json_scan")
BLOCKED_FUNCTION_SUFFIXES = ("_scan",)

_WORD = re.compile(r"[a-z_][a-z0-9_]*")
_CALL = re.compile(r"([a-z_][a-z0-9_]*)\s*\(")
_FILE_SOURCE = re.compile(r"\b(from|join)\s+(?:'|\"[^\"]*[/\\.]|\$|\[)")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    sql: str
    reason: str


def _is_blocked_function(name):
    return (
        name in BLOCKED_FUNCTIONS
        or name.startswith(BLOCKED_FUNCTION_PREFIXES)
        or name.endswith(BLOCKED_FUNCTION_SUFFIXES)
    )


def _sanitize(sql):
    out = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    break
                i += 1
            if i >= n:
                return None
            i += 1
            out.append("''")
        elif ch == "-" and sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end == -1 else end
            out.append(" ")
        elif ch == "/" and sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            if end == -1:
                return None
            i = end + 2
            out.append(" ")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def precheck(sql):
    if not isinstance(sql, str) or not sql.strip():
        return "empty query"
    if len(sql) > 6000:
        return "query too long"
    if "\x00" in sql:
        return "invalid characters"
    cleaned = _sanitize(sql)
    if cleaned is None:
        return "unterminated string or comment"
    bare = cleaned.lower().strip()
    if _FILE_SOURCE.search(bare):
        return "file or external sources are not allowed"
    bare = bare.rstrip(";").strip()
    if ";" in bare:
        return "multiple statements are not allowed"
    if not (bare.startswith("select") or bare.startswith("with") or bare.startswith("(")):
        return "only SELECT queries are allowed"
    words = set(_WORD.findall(bare))
    if words & SYSTEM_CATALOGS:
        return "system catalogs are not allowed"
    hit = words & BLOCKED_WORDS
    if hit:
        return f"blocked keyword: {sorted(hit)[0]}"
    for name in _CALL.findall(bare):
        if _is_blocked_function(name):
            return f"blocked function: {name}"
    return None


def _forbidden_node_types(exp):
    names = [
        "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Copy", "Pragma", "Attach",
        "Detach", "Set", "Use", "Merge", "TruncateTable", "Transaction", "Commit", "Rollback", "Into",
        "ReadCSV", "Export", "Install", "LoadData", "Show", "Describe",
    ]
    return tuple(getattr(exp, n) for n in names if hasattr(exp, n))


def validate(sql, dialect="duckdb", allowed_tables=ALLOWED_TABLES, max_rows=MAX_ROWS, ast=True):
    reason = precheck(sql)
    if reason:
        return Verdict(False, sql, reason)
    if not ast:
        return Verdict(True, sql.strip().rstrip(";"), "ok")

    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as error:
        return Verdict(False, sql, f"parse error: {str(error)[:200]}")
    if len(statements) != 1:
        return Verdict(False, sql, "exactly one statement is required")

    root = statements[0]
    while isinstance(root, exp.Subquery):
        root = root.this
    set_operation = getattr(exp, "SetOperation", exp.Union)
    if not isinstance(root, (exp.Select, set_operation)):
        return Verdict(False, sql, "only SELECT queries are allowed")

    forbidden = _forbidden_node_types(exp)
    for node in root.find_all(*forbidden):
        return Verdict(False, sql, f"blocked statement type: {type(node).__name__}")

    for node in root.find_all(exp.Anonymous):
        if _is_blocked_function(str(node.name).lower()):
            return Verdict(False, sql, f"blocked function: {node.name}")

    cte_names = {cte.alias.lower() for cte in root.find_all(exp.CTE) if cte.alias}
    allowed = {t.lower() for t in allowed_tables}
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return Verdict(False, sql, "table functions and file sources are not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            return Verdict(False, sql, "schema-qualified tables are not allowed")
        name = table.name.lower()
        if name not in allowed and name not in cte_names:
            return Verdict(False, sql, f"table not allowed: {table.name}")

    if isinstance(root, exp.Select):
        limit = root.args.get("limit")
        capped = None
        if limit is None:
            capped = root.limit(max_rows)
        else:
            try:
                value = int(limit.expression.name)
            except (AttributeError, TypeError, ValueError):
                value = None
            if value is None or value > max_rows:
                capped = root.limit(max_rows)
        final = (capped or root).sql(dialect=dialect, comments=False)
    else:
        inner = root.sql(dialect=dialect, comments=False)
        final = f"SELECT * FROM ({inner}) AS limited_query LIMIT {max_rows}"
    return Verdict(True, final, "ok")
