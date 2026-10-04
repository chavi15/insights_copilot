# Explainer: every file in plain English

Read this once from top to bottom, then open each file and find the functions named here. You should be able to explain any of them without looking.

## The idea in one paragraph

A business user asks a question. We show a language model the database layout and ask for SQL. We never trust that SQL: a checker decides whether it is allowed, the database runs it in read-only mode, and the result goes back to the user. Separately, a test harness asks the same model 40 questions with known answers and counts how often it gets them right.

## src/config.py

Holds paths (database, cache, logs) and limits (1000 rows, 10 second timeout). It loads the `.env` file with `python-dotenv` so API keys come from the environment, never from code.

## src/data_gen.py

Builds the fake warehouse with a fixed random seed, so everyone gets the same data.

- `build_frames()` returns seven pandas tables. Sales volume depends on the doctor's tier (A sells most), a seasonal curve, a territory multiplier, and a launch ramp (no sales before a product launches, then a six month climb). Targets are the actual revenue times a random factor per territory, product and quarter, so some territories miss target.
- `write_duckdb()` and `write_sqlite()` save the tables. SQLite exists only so tests can run without DuckDB; the real app uses DuckDB.
- Run it with `python -m src.data_gen`.

## src/schema_docs.yaml and src/schema.py

The YAML describes every table and column in plain English, lists the join keys and gives the model notes such as "a sale reaches a territory only through the doctor".

- `SchemaDocs.render()` turns chosen tables into the text placed in the prompt.
- `SchemaDocs.expand_join_paths()` adds missing tables. If the question needs `fact_sales` and `dim_territory`, it finds the shortest join path (through `dim_hcp`) with a breadth-first search and adds the table in between.

## src/retrieval.py

Chooses which tables to show the model.

- `ChromaRetriever` stores one text document per table in a local ChromaDB and returns the top matches for a question by embedding similarity.
- `KeywordRetriever` is a simple keyword scorer used as a fallback and in tests.
- `TableRetriever` wraps either one, takes the top 3 tables and then calls `expand_join_paths`, so a retrieved set is always joinable.

## src/examples.py and src/nl2sql.py

- `examples.py` holds three worked question and SQL examples. None of them is in the gold set.
- `build_prompt()` assembles the instructions, the schema text, the notes, optional examples and the question. The three variants differ only in schema scope and whether examples are included.
- `extract_sql()` pulls the SQL out of the model's answer, preferring a fenced code block.
- `generate_sql()` ties it together.

## src/llm.py

- `AnthropicLLM` and `GeminiLLM` call the two providers at temperature 0.
- `RetryingLLM` retries rate-limit errors with growing waits and re-raises other errors immediately.
- `CachedLLM` stores each answer on disk keyed by a hash of provider, model and prompt, so reruns of the evaluation are free and repeatable.
- `ScriptedLLM` returns canned answers; tests use it so they need no API key.

## src/guardrails.py

The safety layer. `validate()` runs two checks.

1. `precheck()` works on text. It strips comments and string contents with a small scanner, rejects multiple statements, requires the statement to start with `SELECT` or `WITH`, and blocks dangerous words, file-reading functions and system catalogs.
2. The `sqlglot` step parses the SQL into a tree. It requires one `SELECT`, rejects dangerous node types, rejects table functions and file sources, rejects schema-qualified tables, enforces the table allowlist (CTE names are allowed), and adds `LIMIT 1000` or lowers a larger limit.

It returns a `Verdict` with `ok`, the cleaned SQL and a reason.

## src/executor.py

- `run_query()` opens DuckDB with `read_only=True`, switches off external access, locks the configuration, and starts a timer that interrupts the query after 10 seconds.
- It returns an `ExecResult` that never raises, so errors become data the app can show.
- `append_jsonl()` writes the query log.

## src/pipeline.py

`answer_question()` is the whole flow: generate SQL, validate, execute, log. It returns an `Answer` whose `status` is `ok`, `blocked`, `execution_error` or `generation_error`. The app and the evaluation both use it, so what you test is what you ship.

## src/charts.py

`chart_spec()` picks a line chart when there is a time column and a number, a bar chart for a category and a number, and nothing otherwise.

## src/evaluation.py and eval/run_eval.py

- `results_match()` compares two result tables the way an execution-accuracy benchmark does: same shape and same set of rows, ignoring column names, row order and column order, with numbers rounded to 2 decimals.
- `classify_failure()` labels a wrong answer as hallucinated column or table, wrong join, wrong aggregation, wrong filter or date logic, or other, using simple rules on the SQL text.
- `run_evaluation()` runs every variant over every gold question; `summarize()` and `write_report()` produce the tables in `eval/REPORT.md`.
- `eval/gold.yaml` holds the 40 questions and reference SQL.

## app.py

The Streamlit page: sample questions, a strategy dropdown, the generated SQL, the result table, an automatic chart and a button that flags a wrong answer to `logs/feedback.jsonl`. It creates the database on first start, reads secrets from Streamlit when deployed, and limits live model calls per session in demo mode.

## tests/

- `test_data_gen.py` checks keys, determinism, launch ramps and that some targets are missed.
- `test_guardrails.py` has about 30 blocked queries and 12 valid ones.
- `test_gold.py` runs all 40 reference queries and checks they are deterministic and not copies of the few-shot examples.
- `test_nl2sql_retrieval.py` and `test_pipeline_eval.py` check prompts, retrieval, caching, retries, read-only execution, timeouts, and that a scripted perfect model scores 100% while a deliberately broken one is scored and categorised correctly.
