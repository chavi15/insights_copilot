# Evaluation report

Generated 2026-10-04 01:22 with model `gemini:gemini-3.8-flash` on 6 gold questions (6 easy, 0 medium, 0 hard).

Execution accuracy counts a query as correct when it returns the same rows as the gold query. Column names and row order are ignored, columns may appear in any order, and numbers are compared to 2 decimals. Failure categories are assigned by simple rules on the SQL text, so treat them as a first triage, not a verdict.

## Results by variant and difficulty

| variant | difficulty | questions | execution_accuracy | valid_sql_rate | guardrail_block_rate | avg_latency_ms |
| --- | --- | --- | --- | --- | --- | --- |
| zero_shot_full_schema | easy | 6 | 100.0 | 100.0 | 0.0 | 29.0 |
| zero_shot_full_schema | overall | 6 | 100.0 | 100.0 | 0.0 | 29.0 |
| few_shot_full_schema | easy | 6 | 83.3 | 83.3 | 0.0 | 20587.0 |
| few_shot_full_schema | overall | 6 | 83.3 | 83.3 | 0.0 | 20587.0 |
| retrieval_few_shot | easy | 6 | 16.7 | 16.7 | 0.0 | 102917.0 |
| retrieval_few_shot | overall | 6 | 16.7 | 16.7 | 0.0 | 102917.0 |

## Failure categories (count of failed questions)

| variant | wrong join | wrong aggregation | wrong filter or date logic | hallucinated column or table | other |
| --- | --- | --- | --- | --- | --- |
| zero_shot_full_schema | 0 | 0 | 0 | 0 | 0 |
| few_shot_full_schema | 0 | 0 | 0 | 0 | 1 |
| retrieval_few_shot | 0 | 0 | 0 | 0 | 5 |

## Example failures

### few_shot_full_schema

**e06: What is the total number of units sold?** (category: other, status: generation_error)

Generated:
```sql

```
Gold:
```sql
SELECT SUM(units) AS total_units FROM fact_sales
```
Error: `ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https`

### retrieval_few_shot

**e01: How many products are in the portfolio?** (category: other, status: generation_error)

Generated:
```sql

```
Gold:
```sql
SELECT COUNT(*) AS product_count FROM dim_product
```
Error: `ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https`

**e02: List the product names and their therapeutic areas ordered by product name.** (category: other, status: generation_error)

Generated:
```sql

```
Gold:
```sql
SELECT product_name, therapeutic_area FROM dim_product ORDER BY product_name
```
Error: `ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https`

**e03: How many territories are in each region?** (category: other, status: generation_error)

Generated:
```sql

```
Gold:
```sql
SELECT region, COUNT(*) AS territory_count FROM dim_territory GROUP BY region ORDER BY region
```
Error: `ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https`
