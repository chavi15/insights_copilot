# Commercial Insights Copilot

A natural-language analytics assistant over a synthetic pharma commercial sales warehouse. A user asks a business question in English; a language model writes SQL; safety checks validate it; it runs read-only; the app shows the SQL, a table and a chart. A separate evaluation harness measures how often the generated SQL returns the right answer, comparing three prompting strategies.

## Architecture

```mermaid
flowchart LR
    Q[Business question] --> R[Schema retrieval<br/>ChromaDB + join-path expansion]
    R --> P[Prompt builder<br/>schema + notes + few-shot examples]
    P --> L[LLM<br/>Anthropic or Gemini<br/>disk-cached]
    L --> G[Guardrails<br/>keyword pre-check + sqlglot AST<br/>allowlist + LIMIT]
    G -->|blocked| B[Reason shown to user]
    G -->|allowed| X[Executor<br/>DuckDB read-only<br/>external access off, 10s timeout]
    X --> O[SQL + table + chart]
    X --> J[(JSONL query log)]
    E[Evaluation harness<br/>40 gold questions] -. scores .-> L
```

## Setup (macOS, Python 3.10 or newer)

```bash
cd commercial-insights-copilot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set one key. Never commit it; `.gitignore` already excludes it.

```text
ANTHROPIC_API_KEY=your_key_here
LLM_MODEL=claude-sonnet-5-5
```

For Gemini instead, set `GEMINI_API_KEY` and `LLM_MODEL` (for example `gemini-3.8-flash`). If both keys exist, set `LLM_PROVIDER=anthropic` or `gemini`.

## Run it

```bash
python -m src.data_gen
pytest
python eval/run_eval.py --limit 6
python eval/run_eval.py
streamlit run app.py
```

1. `python -m src.data_gen` builds `data/sales.duckdb` (about 123k sales rows, 32k calls, 6.6k targets).
2. `pytest` runs the tests. Tests that need `sqlglot` run once it is installed.
3. `eval/run_eval.py --limit 6` is a cheap smoke test. The full run makes up to 120 model calls (40 questions x 3 variants); answers are cached in `.cache/llm`, so reruns cost nothing.
4. `eval/REPORT.md` and `eval/results.csv` are written by the evaluation run.

The first ChromaDB run downloads a small embedding model, so it needs internet. If ChromaDB cannot start, the app prints a notice and falls back to a built-in keyword retriever. Set `RETRIEVER=keyword` to force that.

## Schema-only assistant

Pick **Schema-only assistant** in the sidebar to get SQL for your own database without connecting to it. Paste `CREATE TABLE` statements or lines such as `orders: order_id, customer_id, total` (up to 20,000 characters), choose Oracle, PostgreSQL, MySQL or DuckDB, and ask a question. Only table and column names are extracted and sent to Google's Gemini API; types, defaults, comments and constraints are dropped. This mode always uses Gemini (`GEMINI_API_KEY`, optional `GEMINI_MODEL`), skips the disk cache and query log, and never runs the SQL. The SQL is sanity-checked with the guardrails (SELECT only, blocked keywords and functions) using the chosen sqlglot dialect; if sqlglot cannot parse it, the SQL is shown with a warning.

## Data model

Star schema: `dim_product`, `dim_territory`, `dim_hcp`, `dim_date`, `fact_sales`, `fact_calls`, `targets`. All data is synthetic and generated from a fixed seed, with seasonality, launch ramps for newer products, higher volume for top-tier doctors, and monthly targets set so that some territories miss target in some quarters.

## Safety design

- Only one `SELECT` (or `WITH ... SELECT`) statement is accepted.
- A keyword and function pre-check blocks DDL, DML, `COPY`, `ATTACH`, `PRAGMA`, file readers such as `read_csv`, system catalogs and multiple statements.
- A `sqlglot` parse enforces the statement type, a table allowlist (CTE names allowed), no schema-qualified or file-based sources, and adds or caps `LIMIT 1000`.
- DuckDB is opened `read_only=True`, with external access disabled and configuration locked, plus a 10 second timeout.
- Every question, SQL, verdict and latency is appended to `logs/queries.jsonl`.

## Evaluation

`eval/gold.yaml` holds 40 questions with reference SQL: 15 easy (single table), 15 medium (joins) and 10 hard (window functions, CTEs, anti-joins). A generated query is correct when it returns the same rows as the reference query. Column names and row order are ignored, columns may appear in any order, and numbers are compared to 2 decimals. The runner reports execution accuracy, valid-SQL rate, guardrail-block rate and average latency per variant and difficulty, plus a failure-category breakdown.

The three variants:

| Variant | Schema in prompt | Examples |
| --- | --- | --- |
| `zero_shot_full_schema` | all tables | none |
| `few_shot_full_schema` | all tables | 3 worked examples |
| `retrieval_few_shot` | top 3 retrieved tables plus tables needed to join them | 3 worked examples |

**Results:** no measured results are included in this repository yet. Run the evaluation, then copy the numbers from `eval/REPORT.md`. Do not quote figures you have not measured.

## Limitations

- The data is synthetic, so questions are cleaner than real client data would be.
- The gold set has 40 questions, so accuracy differences of a few points are within noise.
- The reference SQL was written by a model and checked by independent pandas calculations on a sample of 12 questions, not all 40. Hand-check more before relying on it.
- Failure categories come from simple rules on the SQL text; they are a first triage.
- There is no per-user access control, and ambiguous questions are not clarified back to the user.
- Top-N questions with tied values can fail on tie-breaking even when the logic is right.
- The guardrails were developed with the pre-check tested directly; the `sqlglot` layer is tested by the tests that run when `sqlglot` is installed.

## Deploying

See `DEPLOY.md`.
