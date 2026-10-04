from types import SimpleNamespace

import pytest
from google.genai import types

from src.datasets import RETAIL
from src.llm import (
    GEMINI_MAX_OUTPUT_TOKENS,
    AnthropicLLM,
    CachedLLM,
    GeminiLLM,
    LLMError,
    RetryingLLM,
    check_gemini_finish,
)
from src.pipeline import answer_question

# What Gemini actually returned for "Which 10 products earned the most net revenue?" with 900 tokens and default thinking.
TRUNCATED_TEXT = "breaker:\n`ORDER BY net_revenue DESC, p.stock_code`\n\nColumns to select:\n`p.stock_code`, `p.description`,"


def gemini_response(reason, text="```sql\nSELECT 1\n```", thoughts=None, message=None):
    candidate = SimpleNamespace(finish_reason=reason, finish_message=message)
    usage = SimpleNamespace(thoughts_token_count=thoughts)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage, text=text)


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(config)
        return self.response


def fake_gemini(response, model="gemini-3.8-flash"):
    llm = GeminiLLM.__new__(GeminiLLM)
    llm.model = model
    llm.client = SimpleNamespace(models=FakeModels(response))
    return llm


def test_max_tokens_raises_clear_error():
    response = gemini_response(types.FinishReason.MAX_TOKENS, TRUNCATED_TEXT, thoughts=866)
    with pytest.raises(LLMError, match=r"max_output_tokens limit of 2048 \(866 of them spent on thinking\).*not run as SQL"):
        check_gemini_finish(response, GEMINI_MAX_OUTPUT_TOKENS)


@pytest.mark.parametrize("reason", [types.FinishReason.STOP, "STOP", None])
def test_stop_is_accepted(reason):
    check_gemini_finish(gemini_response(reason), GEMINI_MAX_OUTPUT_TOKENS)


def test_other_finish_reasons_raise():
    with pytest.raises(LLMError, match="SAFETY"):
        check_gemini_finish(gemini_response(types.FinishReason.SAFETY, message="blocked"), GEMINI_MAX_OUTPUT_TOKENS)
    with pytest.raises(LLMError, match="no answer"):
        check_gemini_finish(SimpleNamespace(candidates=[], usage_metadata=None, text=""), GEMINI_MAX_OUTPUT_TOKENS)


def test_gemini_complete_raises_on_truncation_and_sends_low_thinking():
    llm = fake_gemini(gemini_response(types.FinishReason.MAX_TOKENS, TRUNCATED_TEXT, thoughts=866))
    with pytest.raises(LLMError, match="cut off"):
        llm.complete("prompt")
    (config,) = llm.client.models.calls
    assert config["max_output_tokens"] >= 2048
    assert config["thinking_config"] == {"thinking_level": "low"}


@pytest.mark.parametrize(
    "model, thinking",
    [
        ("gemini-3.8-flash", {"thinking_level": "low"}),
        ("gemini-2.5-flash", {"thinking_budget": 0}),
        ("gemini-2.5-pro", {"thinking_budget": 128}),
    ],
)
def test_thinking_config_per_model(model, thinking):
    assert fake_gemini(None, model).config()["thinking_config"] == thinking


def test_gemini_complete_returns_text_on_stop():
    llm = fake_gemini(gemini_response(types.FinishReason.STOP, "```sql\nSELECT 1\n```"))
    assert llm.complete("prompt") == "```sql\nSELECT 1\n```"


def test_truncated_answer_is_not_run_as_sql(tmp_path):
    llm = fake_gemini(gemini_response(types.FinishReason.MAX_TOKENS, TRUNCATED_TEXT, thoughts=866))
    answer = answer_question(
        "Which 10 products earned the most net revenue?", "few_shot_full_schema", llm, None,
        tmp_path / "missing.duckdb", None, dataset=RETAIL,
    )
    assert answer.status == "generation_error"
    assert answer.verdict is None and answer.result is None
    assert "max_output_tokens" in answer.error


def test_truncated_answer_is_not_cached_or_retried(tmp_path):
    sleeps = []
    inner = fake_gemini(gemini_response(types.FinishReason.MAX_TOKENS, TRUNCATED_TEXT))
    llm = CachedLLM(RetryingLLM(inner, sleep=sleeps.append), cache_dir=tmp_path)
    with pytest.raises(LLMError):
        llm.complete("prompt")
    assert list(tmp_path.iterdir()) == []
    assert sleeps == [] and len(inner.client.models.calls) == 1


def test_anthropic_max_tokens_raises():
    response = SimpleNamespace(stop_reason="max_tokens", content=[SimpleNamespace(type="text", text="SELECT stock_")])
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model = "claude-sonnet-5-5"
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
    with pytest.raises(LLMError, match="cut off"):
        llm.complete("prompt")
