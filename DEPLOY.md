# Deploying the demo

These steps target Streamlit Community Cloud. I have not verified its current memory, sleep or usage limits, so check its documentation and test the deployed app once before you rely on it in an interview.

1. Push the project to a GitHub repository. Make sure `.env` is not committed.
2. Sign in to Streamlit Community Cloud with GitHub, create an app from the repository and set `app.py` as the main file.
3. In the app's Secrets settings, add:

```toml
ANTHROPIC_API_KEY = "your_key_here"
LLM_MODEL = "claude-sonnet-5-5"
DEMO_MODE = "1"
MAX_LIVE_CALLS = "10"
RETRIEVER = "keyword"
GEMINI_API_KEY = "your_gemini_key"  # only needed for the Schema-only assistant
```

4. Deploy. On first start the app generates the synthetic DuckDB database by itself.

Notes:

- `RETRIEVER = "keyword"` skips the ChromaDB model download, which keeps start-up fast and memory low on a free host. Remove it to use ChromaDB.
- `DEMO_MODE` limits live model calls per session so a public link cannot run up your bill. Anyone with the link can still use your key, so keep the limit low and consider a Streamlit viewer allow-list.
- Free apps can sleep. Open the link a few minutes before the interview.
- Keep a local copy running and a short screen recording as a backup.
- If you commit `.cache/llm` (the cached answers) the demo can answer the sample questions without calling the model. Only do this if the cache contains nothing sensitive; it holds prompts and generated SQL, never your key.
