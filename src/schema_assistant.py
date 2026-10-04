"""Schema-only assistant: write SQL from a pasted schema without touching any database.

Nothing here logs, caches or persists the pasted schema or the generated SQL.
Only table and column names are extracted from the paste and sent to the model.
"""

import os
import re
from dataclasses import dataclass

from src import guardrails
from src.llm import DEFAULT_MODELS, GeminiLLM, RetryingLLM
from src.nl2sql import extract_sql

DIALECTS = {"Oracle": "oracle", "PostgreSQL": "postgres", "MySQL": "mysql", "DuckDB": "duckdb"}
MAX_SCHEMA_CHARS = 20_000
PRIVACY_WARNING = "Only table and column names are sent to Google's Gemini API. Don't paste confidential schemas."

_IDENT = r'(?:"[^"\n]{1,128}"|`[^`\n]{1,128}`|\[[^\]\n]{1,128}\]|[A-Za-z_][\w$#]{0,127})'
_NAME = re.compile(rf"{_IDENT}(?:\s*\.\s*{_IDENT}){{0,2}}")
_CREATE = re.compile(
    rf"\bcreate\s+(?:or\s+replace\s+)?(?:(?:global\s+|local\s+)?(?:temporary|temp)\s+|unlogged\s+)?table\s+"
    rf"(?:if\s+not\s+exists\s+)?({_NAME.pattern})\s*\(",
    re.I,
)
_LIST_LINE = re.compile(rf"^\s*({_NAME.pattern})\s*(?::|\((?=.*\)\s*;?\s*$))\s*(.*?)\)?\s*;?\s*$")
_DOTTED_LINE = re.compile(rf"^\s*({_IDENT}(?:\s*\.\s*{_IDENT})?)\s*\.\s*({_IDENT})\s*$")
_CONSTRAINT_WORDS = frozenset(
    {"constraint", "primary", "foreign", "unique", "check", "key", "index", "fulltext", "spatial",
     "exclude", "period", "like", "supplemental", "references"}
)
# Dangerous routines in Oracle, PostgreSQL and MySQL that the DuckDB-focused lists do not cover.
_EXTRA_BLOCKED_PREFIXES = ("dbms_", "utl_")
_EXTRA_BLOCKED_FUNCTIONS = frozenset(
    {"load_file", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_sleep", "pg_terminate_backend",
     "lo_import", "lo_export", "dblink", "dblink_exec", "sleep", "benchmark", "xp_cmdshell"}
)

INSTRUCTIONS = """You are an expert analytics engineer. The user will copy your SQL and run it themselves.
Write SQL for {dialect}.

Rules:
- Return exactly one read-only SELECT statement (a WITH ... SELECT is allowed) inside a single ```sql fenced block.
- Use only the tables and columns listed below. Never invent a table or a column.
- Column data types are not provided; infer them from the names and say so if an assumption matters.
- If the question cannot be fully answered from these tables, write the closest query and explain the gap.
- After the fenced block, write "Explanation:" followed by two to four short sentences on what the query does."""


@dataclass(frozen=True)
class SqlCheck:
    status: str  # "ok", "blocked" or "unparsed"
    reason: str


@dataclass(frozen=True)
class SchemaAnswer:
    sql: str
    explanation: str
    check: SqlCheck
    schema_sent: str


def _clean(name):
    return re.sub(r"\s*\.\s*", ".", name.strip())


def _split_top_level(body):
    items, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(body[start:i])
            start = i + 1
    items.append(body[start:])
    return items


def _balanced_body(text, open_index):
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i + 1
    return text[open_index + 1 :], len(text)


def _add(tables, table, columns):
    known = tables.setdefault(_clean(table), [])
    known.extend(c for c in columns if c not in known)


def _column_names(items):
    columns = []
    for item in items:
        match = re.match(rf"\s*({_IDENT})", item)
        if not match:
            continue
        name = match.group(1)
        if name.lower() in _CONSTRAINT_WORDS:
            continue
        if name not in columns:
            columns.append(name)
    return columns


def parse_schema(text):
    """Return {table: [columns]} holding names only. Types, defaults, comments and literals are dropped."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Paste at least one table definition.")
    if len(text) > MAX_SCHEMA_CHARS:
        raise ValueError(f"The schema is too long. The limit is {MAX_SCHEMA_CHARS:,} characters.")
    cleaned = guardrails._sanitize(text.replace("\x00", ""))
    if cleaned is None:
        raise ValueError("The schema has an unterminated quote or comment.")

    tables = {}
    creates = list(_CREATE.finditer(cleaned))
    if creates:
        for match in creates:
            body, _ = _balanced_body(cleaned, match.end() - 1)
            columns = _column_names(_split_top_level(body))
            if columns:
                _add(tables, match.group(1), columns)
    else:
        for line in cleaned.splitlines():
            dotted = _DOTTED_LINE.match(line)
            listed = _LIST_LINE.match(line)
            if dotted:
                _add(tables, dotted.group(1), [dotted.group(2)])
            elif listed:
                columns = _column_names(listed.group(2).split(","))
                if columns:
                    _add(tables, listed.group(1), columns)
    if not tables:
        raise ValueError(
            "No tables were found. Paste CREATE TABLE statements, or one table per line such as "
            "`orders: order_id, customer_id, total`."
        )
    return tables


def render_schema(tables):
    return "\n".join(f"{table}({', '.join(columns)})" for table, columns in tables.items())


def build_prompt(question, schema_text, dialect_label):
    return "\n\n".join(
        [INSTRUCTIONS.format(dialect=dialect_label), "TABLES (table(columns)):\n" + schema_text, f"Question: {question}"]
    )


def extract_explanation(text):
    if not text:
        return ""
    marker = re.search(r"explanation\s*:\s*", text, re.I)
    rest = text[marker.end() :] if marker else re.sub(r"```.*?```", " ", text, flags=re.S)
    return rest.strip()[:1500]


def _blocked_extra(bare_sql):
    words = set(re.findall(r"[a-z_][a-z0-9_$#]*", bare_sql))
    for word in words:
        if word.startswith(_EXTRA_BLOCKED_PREFIXES):
            return f"blocked package: {word}"
    for name in re.findall(r"([a-z_][a-z0-9_]*)\s*\(", bare_sql):
        if name in _EXTRA_BLOCKED_FUNCTIONS:
            return f"blocked function: {name}"
    return None


def check_sql(sql, dialect="duckdb"):
    """Sanity-check the SQL text only. It is never executed."""
    reason = guardrails.precheck(sql)
    if reason:
        return SqlCheck("blocked", reason)
    reason = _blocked_extra(guardrails._sanitize(sql).lower())
    if reason:
        return SqlCheck("blocked", reason)

    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as error:
        return SqlCheck("unparsed", str(error)[:200])
    if len(statements) != 1:
        return SqlCheck("blocked", "exactly one statement is required")
    root = statements[0]
    while isinstance(root, exp.Subquery):
        root = root.this
    if not isinstance(root, (exp.Select, getattr(exp, "SetOperation", exp.Union))):
        return SqlCheck("blocked", "only SELECT queries are allowed")
    for node in root.find_all(*guardrails._forbidden_node_types(exp)):
        return SqlCheck("blocked", f"blocked statement type: {type(node).__name__}")
    for node in root.find_all(exp.Anonymous):
        name = str(node.name).lower()
        if guardrails._is_blocked_function(name) or name in _EXTRA_BLOCKED_FUNCTIONS:
            return SqlCheck("blocked", f"blocked function: {node.name}")
    return SqlCheck("ok", "ok")


def write_sql(question, schema_paste, dialect_label, llm):
    if dialect_label not in DIALECTS:
        raise ValueError(f"Unsupported dialect: {dialect_label}")
    if not question or not question.strip():
        raise ValueError("Ask a question.")
    schema_text = render_schema(parse_schema(schema_paste))
    text = llm.complete(build_prompt(question.strip(), schema_text, dialect_label))
    sql = extract_sql(text)
    return SchemaAnswer(sql, extract_explanation(text), check_sql(sql, DIALECTS[dialect_label]), schema_text)


def build_schema_llm():
    """Gemini without the disk cache, so neither prompts nor generated SQL are written anywhere."""
    configured = os.getenv("LLM_MODEL", "")
    model = os.getenv("GEMINI_MODEL") or (configured if configured.startswith("gemini") else DEFAULT_MODELS["gemini"])
    return RetryingLLM(GeminiLLM(model=model))
