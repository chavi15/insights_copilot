# Interview Q&A for this project

The answers below describe what the code actually does. Wherever a number would help, it says to read it from `eval/REPORT.md` after you run the evaluation. Do not quote a figure you have not measured, and do not claim anything the code does not do.

## 1. Give me a 30-second summary.

I built a natural-language analytics assistant over a synthetic pharma sales warehouse. A model writes SQL from an English question, a validator blocks anything unsafe, the query runs read-only, and the app shows the SQL, a table and a chart. I also built a 40-question evaluation set and measured accuracy for three prompting strategies. The headline numbers are in `eval/REPORT.md`.

## 2. Why a synthetic dataset?

Real commercial data is confidential and I would not put it through an external model. Synthetic data from a fixed seed means no privacy risk, repeatable results, and I know the true patterns because I generated them: seasonality, launch ramps, tier effects and targets that some territories miss. The cost is that it is cleaner than real data, which I state as a limitation.

## 3. Why a star schema?

Analytics questions are mostly "measure by dimension over time", which is what a star schema is designed for: facts (`fact_sales`, `fact_calls`, `targets`) joined to dimensions (product, territory, HCP, date). It keeps joins simple and predictable for the model. One subtlety I documented for the model is that sales and calls reach a territory only through the doctor.

## 4. Walk me through what happens when a user asks a question.

`answer_question()` in `pipeline.py` does four steps. It retrieves the relevant tables and builds a prompt, the model returns SQL, `guardrails.validate()` checks it, and `run_query()` executes it on a read-only DuckDB connection. Every attempt, including blocked ones, is logged to a JSONL file with the status and latency.

## 5. What do the three prompting variants test?

They isolate two questions. Does showing worked examples help (zero-shot against few-shot, same full schema)? Does showing fewer, relevant tables help (few-shot full schema against retrieval few-shot)? Which variant won on which difficulty is in `eval/REPORT.md`; I would report it per difficulty rather than only overall, because retrieval can help easy questions and hurt hard ones that need more tables.

## 6. Why do you use retrieval over the schema, and how does it work?

A real client schema has hundreds of tables, which will not fit in a prompt and would distract the model. I embed one text document per table in ChromaDB and fetch the top 3 matches. A weakness of plain similarity search is that it can miss a bridging table, so I add the tables on the shortest join path between the retrieved ones with a breadth-first search. For example, revenue by region needs `fact_sales` and `dim_territory`, and the expansion adds `dim_hcp`. If ChromaDB is unavailable the code falls back to a keyword retriever.

## 7. How do you stop the model from running something destructive or leaking data?

Defence in depth, with four layers.

1. A text pre-check requires one statement starting with `SELECT` or `WITH` and blocks dangerous keywords, file-reading functions and system catalogs.
2. A `sqlglot` parse checks the statement type, rejects table functions and file sources, enforces a table allowlist and caps `LIMIT`.
3. DuckDB is opened read-only with external access disabled and its configuration locked.
4. A 10 second timeout interrupts runaway queries.

No single layer is the whole answer; each assumes the others might miss something.

## 8. What about prompt injection?

The model's output is untrusted input to my system, whatever caused it. A malicious question might try to make it emit `DROP TABLE` or read a file, and the validator and read-only connection stop that even if the model complies. What I do not defend against is the model being talked into returning wrong but harmless queries, or leaking data the user is already allowed to see. In a real deployment I would add per-user row-level permissions, and I list the missing access control as a limitation.

## 9. Why measure execution accuracy instead of comparing SQL text?

Many different queries are correct. Text matching punishes valid rewrites and rewards near-misses. Execution accuracy runs both queries and compares the results. My comparison ignores row order, column names and column order and rounds numbers to 2 decimals. Its weakness is false positives: a wrong query can coincide with the right answer on a small or lopsided dataset, and ties in top-N queries can fail a correct query. I mention both.

## 10. How do you know your gold answers are right?

They were written by a model, so I did not trust them. Every gold query must run, return rows and be deterministic, which the tests check. I also recomputed a sample of 12 of them independently in pandas and they matched. I replaced one question that returned zero rows because it could not distinguish a good model from a bad one. The honest gap is that 28 were not independently recomputed, which I would close before quoting accuracy externally.

## 11. What fails most, and why?

Read the failure breakdown in `eval/REPORT.md`. Where I would expect the model to struggle is on questions that need the join path through the doctor, on fan-out traps where revenue is divided by calls from a second fact table, and on date logic. I would pick two real failures from the report, show the generated and reference SQL side by side and explain the cause. The failure categories are assigned by simple text rules, so I treat them as triage.

## 12. How would you scale this to a real client schema?

I would add a curated semantic layer (metric and dimension definitions) so the model picks business terms rather than raw columns, retrieve at column level as well as table level, and add per-user permissions with row-level filters. I would also add a human-review step for sensitive queries, a larger and more varied gold set drawn from real questions, cost and latency monitoring and a feedback loop from the flagged answers.

## 13. How did you control cost and repeatability?

Temperature is 0, every model call is cached on disk by a hash of the provider, model and prompt, and rate-limit errors are retried with growing delays. After a first run, repeating the evaluation makes no new calls. In the deployed demo I limit live calls per session so a public link cannot run up a bill.

## 14. What would you do differently with more time?

I would grow the gold set and add paraphrases of each question to test robustness, add self-correction (feed the error back to the model for one retry) and measure whether it helps, calibrate a confidence signal so the app can say "I am not sure" on ambiguous questions, and test against a real warehouse with messy data.

## 15. How did you use AI tools to build this, and how did you verify the output?

I used an AI assistant to scaffold code and drafts, and I treated its output as a draft from a fast junior colleague. I verified by running it. The tests cover data integrity, guardrail behaviour, the evaluation logic and the pipeline, including a scripted model that is perfect (it must score 100%) and one that is deliberately broken (it must be scored and categorised correctly). I also cross-checked gold queries independently in pandas. I only claim what I can explain and what the tests and the report show.
