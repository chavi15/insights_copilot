import hashlib
import json
import os
import re
import time
from pathlib import Path

from src.config import CACHE_DIR

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "gemini": "gemini-3.8-flash"}


class LLMError(RuntimeError):
    pass


def _is_rate_limit(error):
    text = f"{type(error).__name__} {error}".lower()
    return any(token in text for token in ("429", "rate", "overloaded", "quota", "resource_exhausted", "529", "503", "unavailable", "high demand", "502", "504"))


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model=None, api_key=None):
        import anthropic

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to the .env file in the project root.")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS["anthropic"]

    def _call(self, prompt, with_temperature):
        kwargs = {"model": self.model, "max_tokens": 900, "messages": [{"role": "user", "content": prompt}]}
        if with_temperature:
            kwargs["temperature"] = 0
        response = self.client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

    def complete(self, prompt):
        try:
            return self._call(prompt, True)
        except Exception as error:
            if "temperature" in str(error).lower():
                return self._call(prompt, False)
            raise


class GeminiLLM:
    name = "gemini"

    def __init__(self, model=None, api_key=None):
        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to the .env file in the project root.")
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS["gemini"]

    def complete(self, prompt):
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config={"temperature": 0, "max_output_tokens": 900}
        )
        return response.text or ""


class ScriptedLLM:
    name = "scripted"
    model = "scripted"

    def __init__(self, answers, default="SELECT 1"):
        self.answers = dict(answers)
        self.default = default
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        match = re.findall(r"^Question: (.*)$", prompt, flags=re.M)
        question = match[-1].strip() if match else ""
        return f"```sql\n{self.answers.get(question, self.default)}\n```"


class RetryingLLM:
    def __init__(self, inner, attempts=6, base_delay=4.0, sleep=time.sleep):
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay
        self.sleep = sleep
        self.name = inner.name
        self.model = inner.model

    def complete(self, prompt):
        for attempt in range(self.attempts):
            try:
                return self.inner.complete(prompt)
            except Exception as error:
                if attempt == self.attempts - 1 or not _is_rate_limit(error):
                    raise
                self.sleep(min(60.0, self.base_delay * 2**attempt))
        raise LLMError("unreachable")


class CachedLLM:
    def __init__(self, inner, cache_dir=CACHE_DIR):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = inner.name
        self.model = inner.model
        self.hits = 0
        self.misses = 0

    def _path(self, prompt):
        digest = hashlib.sha256(f"{self.name}|{self.model}|{prompt}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def complete(self, prompt):
        path = self._path(prompt)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self.inner.complete(prompt)
        self.misses += 1
        path.write_text(json.dumps({"model": self.model, "text": text}), encoding="utf-8")
        return text


def detect_provider():
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("anthropic", "gemini"):
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    raise LLMError("No API key found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY in the .env file.")


def build_llm(provider=None, cache=True):
    provider = provider or detect_provider()
    inner = AnthropicLLM() if provider == "anthropic" else GeminiLLM()
    llm = RetryingLLM(inner)
    return CachedLLM(llm) if cache else llm
