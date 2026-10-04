import datetime as dt
import numbers
import re
from collections import Counter

import pandas as pd
import yaml

from src.executor import run_query
from src.pipeline import answer_question

CATEGORIES = [
    "wrong join",
    "wrong aggregation",
    "wrong filter or date logic",
    "hallucinated column or table",
    "other",
]
DIFFICULTIES = ["easy", "medium", "hard"]
_HALLUCINATION = re.compile(
    r"does not exist|no such column|no such table|not found in from|referenced column|catalog error|binder error|table not allowed|unknown column",
    re.I,
)


def load_gold(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_value(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "NULL"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Number):
        return f"{round(float(value), 2):.2f}"
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return str(value)[:10]
    return str(value).strip()


def normalize_frame(frame):
    rows = Counter()
    for row in frame.itertuples(index=False, name=None):
        rows[tuple(sorted(normalize_value(v) for v in row))] += 1
    return rows


def results_match(generated, gold):
    if generated is None or gold is None:
        return False
    if generated.shape != gold.shape:
        return False
    return normalize_frame(generated) == normalize_frame(gold)


def _tables(sql):
    lowered = sql.lower()
    ctes = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s+as\s*\(", lowered))
    return set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", lowered)) - ctes


def _aggregates(sql):
    lowered = sql.lower()
    return (
        frozenset(re.findall(r"\b(sum|count|avg|min|max)\s*\(", lowered)),
        "group by" in lowered,
        "distinct" in lowered,
        "over (" in lowered or "over(" in lowered,
    )


def _where_clause(sql):
    match = re.search(r"\bwhere\b(.*?)(?:\bgroup by\b|\border by\b|\blimit\b|\bhaving\b|$)", sql.lower(), re.S)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def classify_failure(status, error, generated_sql, gold_sql):
    text = error or ""
    if _HALLUCINATION.search(text):
        return "hallucinated column or table"
    if status in ("generation_error", "blocked", "execution_error"):
        return "other"
    if _tables(generated_sql) != _tables(gold_sql):
        return "wrong join"
    if _aggregates(generated_sql) != _aggregates(gold_sql):
        return "wrong aggregation"
    if _where_clause(generated_sql) != _where_clause(gold_sql):
        return "wrong filter or date logic"
    return "other"


class EvaluationStopped(RuntimeError):
    """Raised when stop_if fires; .records holds the rows finished so far, including the one that stopped it."""

    def __init__(self, reason, records):
        super().__init__(reason)
        self.records = records


def run_evaluation(
    gold, llm, retriever, variants, db_path, validator=None, log_path=None, progress=None, dataset=None,
    stop_if=None, pause=None,
):
    gold_frames = {}
    for item in gold:
        outcome = run_query(item["sql"], db_path)
        if not outcome.ok:
            raise RuntimeError(f"gold query {item['id']} failed: {outcome.error}")
        gold_frames[item["id"]] = outcome.frame
    records = []
    for variant in variants:
        for index, item in enumerate(gold, 1):
            if pause and records:
                pause()
            answer = answer_question(
                item["question"], variant, llm, retriever, db_path=db_path, log_path=log_path, validator=validator,
                dataset=dataset,
            )
            frame = answer.result.frame if answer.result and answer.result.ok else None
            correct = answer.status == "ok" and results_match(frame, gold_frames[item["id"]])
            error = answer.error or (answer.result.error if answer.result and not answer.result.ok else None)
            if answer.status == "blocked":
                error = answer.verdict.reason
            category = None if correct else classify_failure(answer.status, error, answer.raw_sql, item["sql"])
            records.append(
                {
                    "variant": variant,
                    "id": item["id"],
                    "difficulty": item["difficulty"],
                    "question": item["question"],
                    "status": answer.status,
                    "correct": bool(correct),
                    "failure_category": category,
                    "generated_sql": answer.raw_sql,
                    "gold_sql": item["sql"],
                    "error": error,
                    "latency_ms": round(answer.latency_ms, 1),
                }
            )
            if progress:
                progress(variant, index, len(gold), correct)
            reason = stop_if(answer) if stop_if else None
            if reason:
                raise EvaluationStopped(reason, pd.DataFrame(records))
    return pd.DataFrame(records)


def summarize(records):
    rows = []
    for variant, group in records.groupby("variant", sort=False):
        for label, subset in [(d, group[group["difficulty"] == d]) for d in DIFFICULTIES] + [("overall", group)]:
            if subset.empty:
                continue
            n = len(subset)
            rows.append(
                {
                    "variant": variant,
                    "difficulty": label,
                    "questions": n,
                    "execution_accuracy": round(subset["correct"].mean() * 100, 1),
                    "valid_sql_rate": round((subset["status"] == "ok").mean() * 100, 1),
                    "guardrail_block_rate": round((subset["status"] == "blocked").mean() * 100, 1),
                    "avg_latency_ms": round(subset["latency_ms"].mean(), 0),
                }
            )
    return pd.DataFrame(rows)


def failure_breakdown(records):
    failed = records[~records["correct"]]
    table = pd.crosstab(failed["variant"], failed["failure_category"]).reindex(columns=CATEGORIES, fill_value=0)
    return table.reindex(records["variant"].unique(), fill_value=0)


def _markdown_table(frame):
    headers = list(frame.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(records, md_path, csv_path, meta):
    records.to_csv(csv_path, index=False)
    summary = summarize(records)
    breakdown = failure_breakdown(records).reset_index()
    lines = [
        f"# Evaluation report{meta.get('title_suffix', '')}",
        "",
        f"Generated {meta['generated']} with model `{meta['model']}` on {meta['questions']} gold questions "
        f"({meta['easy']} easy, {meta['medium']} medium, {meta['hard']} hard).",
        "",
        "Execution accuracy counts a query as correct when it returns the same rows as the gold query. Column names and "
        "row order are ignored, columns may appear in any order, and numbers are compared to 2 decimals. Failure categories "
        "are assigned by simple rules on the SQL text, so treat them as a first triage, not a verdict.",
        "",
        "## Results by variant and difficulty",
        "",
        _markdown_table(summary),
        "",
        "## Failure categories (count of failed questions)",
        "",
        _markdown_table(breakdown),
        "",
        "## Example failures",
    ]
    for variant, group in records[~records["correct"]].groupby("variant", sort=False):
        lines += ["", f"### {variant}"]
        for item in group.head(3).itertuples():
            lines += [
                "",
                f"**{item.id}: {item.question}** (category: {item.failure_category}, status: {item.status})",
                "",
                "Generated:",
                "```sql",
                str(item.generated_sql),
                "```",
                "Gold:",
                "```sql",
                str(item.gold_sql),
                "```",
            ]
            if item.error:
                lines.append(f"Error: `{str(item.error)[:200]}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
