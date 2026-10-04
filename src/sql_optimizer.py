"""Query optimizer for the schema-only assistant.

Rule-based findings come from sqlglot alone, with no model call. Model suggestions send only the
query with string literals masked, the table and column names, and the plan text with quoted values
masked. Nothing here logs, caches or runs anything.
"""

import re
from dataclasses import dataclass

from src.schema_assistant import DIALECTS, parse_schemas, render_schema

MAX_QUERY_CHARS = 20_000
MAX_PLAN_CHARS = 10_000
OPTIMIZER_WARNING = (
    "The pasted query and plan text are sent to Google's Gemini API, together with table and column names. "
    "String literals in the query, and quoted values in the plan, are replaced with ? before sending."
)
SUGGESTIONS_LABEL = "Unverified suggestions: compare plans on your own database before using."
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

_PSEUDO_COLUMNS = frozenset({"rownum", "rowid", "level", "sysdate", "systimestamp", "user"})
_NUMERIC_STRING = re.compile(r"[+-]?\d+(?:\.\d+)?")
_DATE_STRING = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?")
_Q_QUOTE_CLOSERS = {"[": "]", "(": ")", "{": "}", "<": ">"}

PROMPT = """You are a database performance expert. Write SQL for {dialect}.
The user wants to make the query below faster. You cannot run anything. String literals were replaced with ?.

Rules:
- Give at most six concrete suggestions as a short bulleted list: indexes to consider (with CREATE INDEX statements), query rewrites, and what to look for in the plan.
- For each suggestion, say why it should help and how confident you are.
- If you propose a rewritten query, put it in one ```sql block and keep its results identical to the original.
- Do not invent tables or columns beyond those listed or used in the query."""


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    reason: str
    suggestion: str
    snippet: str


@dataclass(frozen=True)
class Analysis:
    findings: list
    parse_error: str | None


@dataclass(frozen=True)
class Suggestions:
    text: str
    schema_sent: str
    query_sent: str
    plan_sent: str


def _check_lengths(sql, plan=""):
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("Paste a SQL query.")
    if len(sql) > MAX_QUERY_CHARS:
        raise ValueError(f"The query is too long. The limit is {MAX_QUERY_CHARS:,} characters.")
    if plan and len(plan) > MAX_PLAN_CHARS:
        raise ValueError(f"The plan text is too long. The limit is {MAX_PLAN_CHARS:,} characters.")


# ---------------------------------------------------------------- literal masking


def mask_literals(text, dialect=None, strict=True):
    """Replace string literals with ? and drop comments.

    dialect=None masks single-quoted values only and keeps everything else, which suits plan text.
    strict=False masks an unterminated string to the end instead of raising.
    """
    out, i, n = [], 0, len(text)
    sql_mode = dialect is not None

    def unterminated():
        if strict:
            raise ValueError("The query has an unterminated string or comment, so it was not sent.")
        out.append("?")
        return n

    while i < n:
        ch = text[i]
        prev = text[i - 1] if i else ""
        if sql_mode and text.startswith("--", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
            out.append(" ")
        elif sql_mode and text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                i = unterminated()
            else:
                i = end + 2
                out.append(" ")
        elif (
            dialect == "oracle" and ch in "qQ" and text.startswith("'", i + 1) and i + 2 < n
            and not (prev.isalnum() and prev not in "nN") and not prev == "_"
        ):
            close = _Q_QUOTE_CLOSERS.get(text[i + 2], text[i + 2]) + "'"
            end = text.find(close, i + 3)
            if end == -1:
                i = unterminated()
            else:
                out.append("?")
                i = end + 2
        elif dialect == "postgres" and ch == "$" and re.match(r"\$(?:[A-Za-z_]\w*)?\$", text[i:]):
            tag = re.match(r"\$(?:[A-Za-z_]\w*)?\$", text[i:]).group(0)
            end = text.find(tag, i + len(tag))
            if end == -1:
                i = unterminated()
            else:
                out.append("?")
                i = end + len(tag)
        elif ch == "'" or (dialect == "mysql" and ch == '"'):
            backslash = dialect == "mysql" or (dialect == "postgres" and prev in "eE")
            j = i + 1
            while j < n:
                if backslash and text[j] == "\\":
                    j += 2
                    continue
                if text[j] == ch:
                    if j + 1 < n and text[j + 1] == ch:
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                i = unterminated()
            else:
                out.append("?")
                i = j + 1
        elif sql_mode and ch in '"`[':
            close = {"[": "]"}.get(ch, ch)
            end = text.find(close, i + 1)
            end = n if end == -1 else end + 1
            out.append(text[i:end])
            i = end
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# ---------------------------------------------------------------- rules


def _snippet(node, dialect):
    try:
        text = node.sql(dialect=dialect)
    except Exception:
        text = str(node)
    return text if len(text) <= 200 else text[:197] + "..."


def _unwrap(node):
    from sqlglot import exp

    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _real_columns(node):
    from sqlglot import exp

    return [c for c in node.find_all(exp.Column) if c.name.lower() not in _PSEUDO_COLUMNS]


def _rule_select_star(root, dialect, exp):
    for select in root.find_all(exp.Select):
        if isinstance(select.parent, exp.Exists):
            continue
        for item in select.expressions:
            if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
                yield Finding(
                    "SELECT *", "low",
                    "SELECT * reads every column, which stops covering indexes from being used, moves more data "
                    "and breaks silently when columns are added.",
                    "List only the columns you need, for example SELECT e.emp_id, e.name instead of SELECT *.",
                    _snippet(item, dialect),
                )
                break


def _function_advice(func, exp):
    name = type(func).__name__.lower()
    if isinstance(func, (exp.Upper, exp.Lower)):
        return ("Compare against the column as stored, or create an expression index (Oracle: function-based "
                f"index) on {name.upper()}(column).")
    if isinstance(func, (exp.Year, exp.Month, exp.Day, exp.Extract, exp.DateTrunc, exp.TimeToStr)) or (
        isinstance(func, exp.Anonymous) and func.name.lower() in {"trunc", "to_char", "date_format"}
    ):
        return ("Compare the bare column with a range instead, for example "
                "hire_date >= DATE '2024-01-01' AND hire_date < DATE '2025-01-01'.")
    return ("Move the function to the other side of the comparison so the bare column is compared, or create "
            "an expression index that matches it exactly.")


def _rule_function_on_column(root, dialect, exp):
    seen = set()
    for where in root.find_all(exp.Where):
        for pred in where.find_all(exp.Predicate):
            if id(pred) in seen:
                continue
            seen.add(id(pred))
            sides = [pred.this, pred.args.get("expression"), pred.args.get("low"), pred.args.get("high")]
            for side in sides:
                side = _unwrap(side) if side is not None else None
                if (
                    isinstance(side, exp.Func)
                    and not isinstance(side, (exp.AggFunc, exp.Exists))
                    and not side.find(exp.Select)
                    and _real_columns(side)
                ):
                    yield Finding(
                        "Function on a column in WHERE", "medium",
                        "Wrapping a column in a function means a normal index on that column cannot be used for "
                        "this filter, so the database may scan the whole table.",
                        _function_advice(side, exp),
                        _snippet(pred, dialect),
                    )
                    break


def _rule_leading_wildcard(root, dialect, exp):
    for like in root.find_all(exp.Like, exp.ILike):
        pattern = _unwrap(like.expression)
        if isinstance(pattern, exp.Literal) and pattern.is_string and pattern.this.startswith(("%", "_")):
            yield Finding(
                "Leading-wildcard LIKE", "medium",
                "A pattern that starts with % or _ cannot use a normal B-tree index, so every row is checked.",
                "Anchor the pattern at the start (LIKE 'abc%') if you can. Otherwise use a text index: Oracle Text, "
                "PostgreSQL pg_trgm, a MySQL FULLTEXT index, or a reversed-column index for suffix searches.",
                _snippet(like, dialect),
            )


def _links(where, alias, exp):
    if where is None:
        return False
    alias = alias.lower()
    for eq in where.find_all(exp.EQ):
        left = {c.table.lower() for c in _real_columns(eq.this)}
        right = {c.table.lower() for c in _real_columns(eq.expression)}
        if not left or not right:
            continue
        if "" in left or "" in right:
            return True  # unqualified columns: cannot tell, so do not flag
        if (alias in left or alias in right) and left != right:
            return True
    return False


def _rule_missing_join_condition(root, dialect, exp):
    for select in root.find_all(exp.Select):
        for join in select.args.get("joins") or []:
            if join.args.get("on") or join.args.get("using") or join.args.get("method") or join.args.get("kind") == "CROSS":
                continue
            target = join.this
            if isinstance(target, exp.Table) and not isinstance(target.this, exp.Identifier):
                continue
            if not isinstance(target, (exp.Table, exp.Subquery)):
                continue
            if _links(select.args.get("where"), target.alias_or_name, exp):
                continue
            yield Finding(
                "Missing join condition", "high",
                f"{target.alias_or_name} is joined without an ON condition or a matching WHERE condition, which "
                "produces a Cartesian product (every row times every row).",
                "Add the join keys, for example JOIN departments d ON d.dept_id = e.dept_id. If you really want "
                "every combination, write CROSS JOIN so the intent is explicit.",
                _snippet(join, dialect),
            )


def _rule_not_in_subquery(root, dialect, exp):
    for node in root.find_all(exp.Not):
        inner = _unwrap(node.this)
        if isinstance(inner, exp.In) and inner.args.get("query"):
            yield Finding(
                "NOT IN with a subquery", "high",
                "If the subquery returns any NULL, NOT IN returns no rows at all. It can also stop the optimizer "
                "from using an efficient anti-join.",
                "Use NOT EXISTS: WHERE NOT EXISTS (SELECT 1 FROM other o WHERE o.key = t.key).",
                _snippet(inner.this, dialect) + " NOT IN " + _snippet(inner.args["query"], dialect),
            )


def _rule_correlated_subquery(root, dialect, exp):
    from sqlglot.optimizer.scope import traverse_scope

    try:
        scopes = traverse_scope(root)
    except Exception:
        return
    for scope in scopes:
        if not scope.is_subquery:
            continue
        own = {name.lower() for name in scope.sources}
        # Only qualified columns count: sqlglot treats every unqualified column in a subquery as external.
        if not any(c.table and c.table.lower() not in own for c in scope.columns):
            continue
        node = scope.expression
        parent = node.parent
        while isinstance(parent, exp.Subquery):
            parent = parent.parent
        if isinstance(parent, exp.Exists):
            continue  # EXISTS / NOT EXISTS correlation is the recommended pattern
        yield Finding(
            "Correlated subquery", "medium",
            "This subquery refers to the outer query, so it may run once per outer row.",
            "Rewrite it as a JOIN to a pre-aggregated derived table (GROUP BY the join key), or use a window "
            "function such as MAX(...) OVER (PARTITION BY key).",
            _snippet(node, dialect),
        )


def _rule_distinct_with_joins(root, dialect, exp):
    for select in root.find_all(exp.Select):
        if select.args.get("distinct") and select.args.get("joins"):
            yield Finding(
                "DISTINCT with joins", "medium",
                "DISTINCT on a joined result often hides row multiplication from a one-to-many join, and the "
                "database has to sort or hash every row to remove the duplicates.",
                "If the join only filters rows, use WHERE EXISTS (SELECT 1 FROM ...) instead. Otherwise check "
                "for a missing join key, or aggregate before joining.",
                _snippet(select, dialect)[:120],
            )


def _uses_rownum(select, exp):
    where = select.args.get("where")
    return bool(where) and any(c.name.lower() == "rownum" for c in where.find_all(exp.Column))


def _rule_order_without_limit(root, dialect, exp):
    set_operation = getattr(exp, "SetOperation", exp.Union)
    for select in root.find_all(exp.Select):
        if select is root or not select.args.get("order"):
            continue
        if select.args.get("limit") or select.args.get("offset") or select.args.get("fetch"):
            continue
        ancestors, parent = [], select.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        if all(isinstance(a, set_operation) for a in ancestors):
            continue  # ORDER BY of the outermost query
        if any(isinstance(a, exp.Array) for a in ancestors):
            continue  # ARRAY(SELECT ... ORDER BY) keeps the order
        outer = select.parent_select
        if outer is not None and _uses_rownum(outer, exp):
            continue  # Oracle top-N with ROWNUM needs the inner ORDER BY
        yield Finding(
            "ORDER BY in a subquery without LIMIT/FETCH", "low",
            "The order of a subquery or CTE is not guaranteed to survive into the outer query, so this sort is "
            "wasted work (some databases ignore it, others still pay for it).",
            "Remove the inner ORDER BY and sort in the outermost query, or add LIMIT / FETCH FIRST n ROWS ONLY if "
            "you meant a top-N.",
            _snippet(select.args["order"], dialect),
        )


def _rule_implicit_conversion(root, dialect, exp):
    for pred in root.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE):
        for col, lit in ((_unwrap(pred.this), _unwrap(pred.expression)), (_unwrap(pred.expression), _unwrap(pred.this))):
            if not (isinstance(col, exp.Column) and isinstance(lit, exp.Literal) and lit.is_string):
                continue
            if _NUMERIC_STRING.fullmatch(lit.this):
                yield Finding(
                    "Implicit type conversion", "low",
                    f"{col.sql(dialect=dialect)} is compared with the quoted number '{lit.this}'. If the column is "
                    "numeric, the database converts types, and in some cases that conversion stops an index from "
                    "being used.",
                    f"If the column is numeric, write {lit.this} without quotes. If it is text, keep the quotes.",
                    _snippet(pred, dialect),
                )
            elif _DATE_STRING.fullmatch(lit.this):
                yield Finding(
                    "Implicit type conversion", "medium" if dialect == "oracle" else "low",
                    f"{col.sql(dialect=dialect)} is compared with the string '{lit.this}'. If the column is a "
                    "date or timestamp, the string is converted using session settings (in Oracle, NLS_DATE_FORMAT), "
                    "which can fail or compare differently.",
                    f"Use a typed literal such as DATE '{lit.this[:10]}' or TIMESTAMP '{lit.this}', or "
                    "TO_DATE('...', 'YYYY-MM-DD') in Oracle.",
                    _snippet(pred, dialect),
                )
            break


RULES = (
    _rule_missing_join_condition,
    _rule_not_in_subquery,
    _rule_function_on_column,
    _rule_leading_wildcard,
    _rule_correlated_subquery,
    _rule_distinct_with_joins,
    _rule_implicit_conversion,
    _rule_order_without_limit,
    _rule_select_star,
)


def analyze(sql, dialect="duckdb"):
    """Rule-based findings from sqlglot only. Nothing is sent anywhere and nothing is run."""
    _check_lengths(sql)
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as error:
        return Analysis([], str(error)[:300])
    findings = []
    for root in statements:
        for rule in RULES:
            findings.extend(rule(root, dialect, exp))
    unique = list(dict.fromkeys(findings))
    unique.sort(key=lambda f: SEVERITY_ORDER[f.severity])
    return Analysis(unique, None)


# ---------------------------------------------------------------- model suggestions


def build_prompt(query_sent, dialect_label, schema_sent="", plan_sent=""):
    parts = [PROMPT.format(dialect=dialect_label)]
    if schema_sent:
        parts.append("TABLES (table(columns)):\n" + schema_sent)
    parts.append(f"QUERY:\n```sql\n{query_sent}\n```")
    if plan_sent:
        parts.append("EXECUTION PLAN:\n" + plan_sent)
    return "\n\n".join(parts)


def suggest(sql, dialect_label, llm, schemas=(), plan=""):
    """Ask the model for suggestions. schemas is a list of (schema_name, paste) pairs and may be empty."""
    if dialect_label not in DIALECTS:
        raise ValueError(f"Unsupported dialect: {dialect_label}")
    plan = plan or ""
    _check_lengths(sql, plan)
    query_sent = mask_literals(sql, DIALECTS[dialect_label]).strip()
    plan_sent = mask_literals(plan, None, strict=False).strip()
    has_schema = any((name or "").strip() or (paste or "").strip() for name, paste in schemas)
    schema_sent = render_schema(parse_schemas(schemas)) if has_schema else ""
    text = llm.complete(build_prompt(query_sent, dialect_label, schema_sent, plan_sent))
    return Suggestions((text or "").strip()[:6000], schema_sent, query_sent, plan_sent)
