import pytest

from src.llm import CachedLLM, RetryingLLM, ScriptedLLM
from src.nl2sql import VARIANTS, build_prompt, extract_sql, generate_sql
from src.retrieval import KeywordRetriever, TableRetriever
from src.schema import load_schema


@pytest.fixture(scope="module")
def retriever():
    return TableRetriever(KeywordRetriever(), k=3)


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Here you go:\n```\nSELECT 2\n```\nDone") == "SELECT 2"
    assert extract_sql("SELECT 3 FROM t;") == "SELECT 3 FROM t"
    assert extract_sql("Sure! with a as (select 1) select * from a") == "with a as (select 1) select * from a"
    assert extract_sql("") == ""


def test_schema_join_path_expansion():
    schema = load_schema()
    assert set(schema.expand_join_paths(["fact_sales", "dim_territory"])) == {"fact_sales", "dim_hcp", "dim_territory"}
    assert set(schema.expand_join_paths(["fact_calls", "targets"])) >= {"dim_date"} or "dim_hcp" in schema.expand_join_paths(
        ["fact_calls", "targets"]
    )


@pytest.mark.parametrize(
    "question, must_have",
    [
        ("What was total revenue by region in 2025?", {"fact_sales", "dim_territory", "dim_hcp", "dim_date"}),
        ("Which territories missed their target?", {"targets", "dim_territory"}),
        ("Average call duration by channel", {"fact_calls"}),
        ("How many products launched after 2024?", {"dim_product"}),
        ("Calls made to tier A doctors", {"fact_calls", "dim_hcp"}),
    ],
)
def test_keyword_retrieval_finds_needed_tables(retriever, question, must_have):
    assert must_have <= set(retriever.tables_for(question))


def test_prompts_differ_by_variant(retriever):
    q = "What is the total revenue by region?"
    zero = build_prompt(q, "zero_shot_full_schema")
    few = build_prompt(q, "few_shot_full_schema")
    rag = build_prompt(q, "retrieval_few_shot", retriever)
    assert "EXAMPLES:" not in zero and "EXAMPLES:" in few and "EXAMPLES:" in rag
    assert zero.rstrip().endswith(f"Question: {q}")
    assert "TABLE fact_calls" in few and "TABLE fact_calls" not in rag
    assert len(rag) < len(few)
    assert "TABLE fact_sales" in rag and "TABLE dim_territory" in rag


def test_unknown_variant_and_missing_retriever():
    with pytest.raises(ValueError):
        build_prompt("q", "nope")
    with pytest.raises(ValueError):
        build_prompt("q", "retrieval_few_shot")


def test_generate_sql_with_scripted_llm(retriever):
    llm = ScriptedLLM({"How many products?": "SELECT COUNT(*) FROM dim_product"})
    for variant in VARIANTS:
        assert generate_sql("How many products?", variant, llm, retriever) == "SELECT COUNT(*) FROM dim_product"


def test_cached_llm_calls_inner_once(tmp_path):
    inner = ScriptedLLM({"q": "SELECT 1"})
    cached = CachedLLM(inner, tmp_path)
    prompt = "Question: q"
    assert cached.complete(prompt) == cached.complete(prompt)
    assert inner.calls == 1 and cached.hits == 1 and cached.misses == 1


def test_retrying_llm_retries_rate_limits_only():
    class Flaky:
        name, model = "flaky", "m"

        def __init__(self, failures, message):
            self.failures, self.message, self.calls = failures, message, 0

        def complete(self, prompt):
            self.calls += 1
            if self.calls <= self.failures:
                raise RuntimeError(self.message)
            return "ok"

    sleeps = []
    flaky = Flaky(2, "Error 429 rate limit")
    assert RetryingLLM(flaky, sleep=sleeps.append).complete("x") == "ok"
    assert flaky.calls == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0]
    broken = Flaky(5, "invalid api key")
    with pytest.raises(RuntimeError):
        RetryingLLM(broken, sleep=sleeps.append).complete("x")
    assert broken.calls == 1
