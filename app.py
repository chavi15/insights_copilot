import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st


def load_streamlit_secrets():
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LLM_MODEL", "LLM_PROVIDER", "GEMINI_MODEL", "RETRIEVER", "DEMO_MODE", "MAX_LIVE_CALLS"):
        try:
            if key in st.secrets:
                os.environ.setdefault(key, str(st.secrets[key]))
        except Exception:
            return


load_streamlit_secrets()

from src.charts import chart_spec
from src.config import DB_PATH, FEEDBACK_PATH
from src.data_gen import build_frames, write_db
from src.executor import append_jsonl
from src.llm import LLMError, build_llm
from src.nl2sql import VARIANTS
from src.pipeline import answer_question
from src.retrieval import get_retriever
from src.schema import load_schema
from src.schema_assistant import DIALECTS, MAX_SCHEMA_CHARS, PRIVACY_WARNING, build_schema_llm, write_sql

SAMPLE_QUESTIONS = [
    "What was the total revenue for each product in 2025?",
    "What is the total revenue for each region?",
    "Which 5 territories had the highest total revenue?",
    "Show total revenue by month together with the change from the previous month.",
    "Which territories missed their total target revenue in Q4 2025? Show actual revenue and target.",
    "What are the top 3 territories by total revenue within each region?",
    "What percentage of total revenue does each region contribute?",
    "How many calls were made to HCPs in each tier?",
]
DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")
MAX_LIVE_CALLS = int(os.getenv("MAX_LIVE_CALLS", "10" if DEMO_MODE else "1000"))

PASTELS = ["#CDB8FF", "#BFDBFE", "#C7F5E6"]
BADGES = ["Read-only DuckDB", "SQL guardrails", "Synthetic data"]
PIPELINE_STEPS = ["Question", "Schema retrieval", "LLM", "Guardrails", "Read-only execution", "Result"]
MODES = ("Warehouse demo", "Schema-only assistant")
QUOTA_MESSAGE = "Free-tier limit reached. Try a sample question, which may be cached."

st.set_page_config(page_title="Commercial Insights Copilot", layout="wide")

# Contrast on #0B0B14: text #EDEDF7 16.8:1, muted #9A9AB8 7.2:1, dark text on pastels 11:1 or more.
# The purple accent (3.1:1) is never used as text; the header gradient is translucent over the card colour.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #0B0B14; --card: #14142B; --purple: #5B2EFF; --blue: #3B82F6;
  --text: #EDEDF7; --muted: #9A9AB8; --line: rgba(237, 237, 247, 0.08);
  --body-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --head-font: 'Bebas Neue', 'Arial Narrow', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
html, body, .stApp, .stMarkdown, p, label, input, textarea, button, [data-testid="stWidgetLabel"] {
  font-family: var(--body-font);
}
.stApp { background: var(--bg); color: var(--text); }
h1, h2, h3, .cic-title { font-family: var(--head-font) !important; letter-spacing: 0.04em; font-weight: 400 !important; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* Header */
.cic-header {
  background:
    linear-gradient(135deg, rgba(91, 46, 255, 0.55), rgba(59, 130, 246, 0.35)),
    var(--card);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.6rem 1.8rem 1.4rem;
  margin-bottom: 1.2rem;
  box-shadow: 0 10px 40px rgba(91, 46, 255, 0.18);
  animation: cic-fade 0.3s ease-out;
}
.cic-title { font-size: 3rem; line-height: 1; color: var(--text); margin: 0; }
.cic-sub { color: #DCDCEC; margin: 0.5rem 0 0.9rem; max-width: 52rem; }
.cic-badge {
  display: inline-block; margin: 0 0.4rem 0.3rem 0; padding: 0.22rem 0.75rem;
  border-radius: 999px; font-size: 0.78rem; font-weight: 500; color: var(--text);
  background: rgba(11, 11, 20, 0.35); border: 1px solid rgba(237, 237, 247, 0.18);
}

/* Pipeline diagram */
.cic-flow { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; padding: 0.3rem 0; }
.cic-step {
  padding: 0.4rem 0.8rem; border-radius: 12px; background: rgba(91, 46, 255, 0.16);
  border: 1px solid rgba(91, 46, 255, 0.35); color: var(--text); font-size: 0.85rem;
}
.cic-arrow { color: var(--muted); }

/* Cards and containers */
[data-testid="stExpander"] details, [data-testid="stTabs"], [data-testid="stAlert"] {
  border-radius: 14px !important; border: 1px solid var(--line) !important;
}
[data-testid="stExpander"] details { background: var(--card); transition: border-color 0.25s ease; }
[data-testid="stExpander"] details:hover { border-color: rgba(91, 46, 255, 0.4) !important; }
[data-testid="stTabs"] { background: var(--card); padding: 0.6rem 1.1rem 1.1rem; animation: cic-fade 0.35s ease-out; }
[data-testid="stTabs"] button[role="tab"] { transition: color 0.2s ease; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: var(--text); font-weight: 600; }
[data-testid="stTabs"] button[role="tab"] p { color: var(--muted); }
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--purple), var(--blue)) !important; }
[data-baseweb="tab-border"] { background: var(--line) !important; }
[data-testid="stDataFrame"], [data-testid="stCode"], .stCodeBlock { border-radius: 12px; overflow: hidden; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }

/* Question box */
.stTextArea textarea {
  background: var(--card) !important; color: var(--text) !important; font-size: 1.05rem;
  border-radius: 14px !important; border: 1px solid var(--line) !important;
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.stTextArea textarea:focus { border-color: var(--blue) !important; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25) !important; }
.stTextArea [data-baseweb="textarea"] { border: none !important; background: transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .stButton button { width: 100%; justify-content: flex-start; text-align: left; }
[data-testid="stSidebar"] .stButton button p { text-align: left; font-size: 0.85rem; }
[data-baseweb="select"] > div { border-radius: 12px !important; transition: border-color 0.2s ease; }

/* Buttons: pastel fills, dark text, soft lift on hover */
.stButton button, [data-testid="stDownloadButton"] button {
  background: #CDB8FF; color: #0B0B14 !important; border: none; border-radius: 12px;
  font-weight: 600; transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}
.stButton button p, [data-testid="stDownloadButton"] button p { color: #0B0B14 !important; }
.stButton button:hover, [data-testid="stDownloadButton"] button:hover {
  transform: translateY(-2px); box-shadow: 0 8px 22px rgba(91, 46, 255, 0.35); filter: brightness(1.04);
  color: #0B0B14 !important; border: none;
}
.stButton button:active, [data-testid="stDownloadButton"] button:active { transform: translateY(0); }
.stButton button:focus-visible, [data-testid="stDownloadButton"] button:focus-visible {
  outline: 2px solid var(--blue); outline-offset: 2px;
}
[data-testid="stDownloadButton"] button { background: #BFDBFE; }
[data-testid="stDownloadButton"] button:hover { box-shadow: 0 8px 22px rgba(59, 130, 246, 0.35); }
.st-key-flag button { background: #C7F5E6; }
.st-key-flag button:hover { box-shadow: 0 8px 22px rgba(199, 245, 230, 0.25); }
.st-key-ask button { padding: 0.55rem 2rem; }
__SAMPLE_RULES__

@keyframes cic-fade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def sample_button_rules():
    return "\n".join(
        f".st-key-sample_{i} button {{ background: {PASTELS[i % len(PASTELS)]}; }}" for i in range(len(SAMPLE_QUESTIONS))
    )


def inject_css():
    st.markdown(CSS.replace("__SAMPLE_RULES__", sample_button_rules()), unsafe_allow_html=True)


def render_header():
    badges = "".join(f'<span class="cic-badge">{label}</span>' for label in BADGES)
    st.markdown(
        f"""<div class="cic-header">
<h1 class="cic-title">Commercial Insights Copilot</h1>
<p class="cic-sub">Ask a business question in English. A language model writes the SQL, safety checks validate it,
and it runs read-only on a synthetic pharma sales warehouse.</p>
{badges}
</div>""",
        unsafe_allow_html=True,
    )


def render_how_it_works():
    with st.expander("How it works"):
        arrow = '<span class="cic-arrow">&rarr;</span>'
        steps = arrow.join(f'<span class="cic-step">{step}</span>' for step in PIPELINE_STEPS)
        st.markdown(f'<div class="cic-flow">{steps}</div>', unsafe_allow_html=True)
        st.caption(
            "The most relevant tables are retrieved, the model writes one SELECT, the guardrails check it against an "
            "allowlist and cap the rows, and DuckDB runs it in read-only mode with a 10 second timeout."
        )


@st.cache_resource(show_spinner="Preparing the synthetic database...")
def ensure_database():
    if not Path(DB_PATH).exists():
        write_db(build_frames(), DB_PATH)
    return str(DB_PATH)


@st.cache_resource(show_spinner="Starting the Gemini client...")
def get_schema_llm():
    return build_schema_llm()


@st.cache_resource(show_spinner="Starting the language model and retriever...")
def get_services():
    return build_llm(), get_retriever()


def is_quota_error(text):
    lowered = (text or "").lower()
    return "resource_exhausted" in lowered or "exceeded your current quota" in lowered


def tables_used(question, variant, retriever):
    if variant != "retrieval_few_shot":
        return load_schema().table_names
    try:
        return retriever.tables_for(question)
    except Exception:
        return []


def render_status_message(answer):
    if answer.status == "generation_error":
        if is_quota_error(answer.error):
            st.warning(QUOTA_MESSAGE)
        else:
            st.error("The language model could not be reached just now. Please try again in a moment.")
        st.caption(f"Details: {answer.error}")
    elif answer.status == "blocked":
        st.error(f"The safety checks stopped this query, so nothing was run. Reason: {answer.verdict.reason}")
    elif answer.status == "execution_error":
        st.error("The query passed the safety checks but the database could not run it. Try rephrasing the question.")
        st.caption(f"Details: {answer.result.error}")


def render(answer, tables):
    render_status_message(answer)
    answer_tab, sql_tab, checked_tab = st.tabs(["Answer", "SQL", "How it was checked"])

    with answer_tab:
        if answer.status != "ok":
            st.caption("No result to show. See the message above, or the SQL tab for what the model wrote.")
        else:
            frame = answer.result.frame
            spec = chart_spec(frame)
            if spec:
                data = frame.set_index(spec["x"])[spec["y"]]
                st.line_chart(data) if spec["kind"] == "line" else st.bar_chart(data)
            st.dataframe(frame, use_container_width=True)
            left, right = st.columns([1, 1])
            with left:
                st.download_button(
                    "Download CSV", frame.to_csv(index=False).encode("utf-8"), file_name="answer.csv", mime="text/csv"
                )
            with right:
                if st.button("Flag this answer as wrong", key="flag"):
                    append_jsonl(FEEDBACK_PATH, {"question": answer.question, "variant": answer.variant, "sql": answer.raw_sql})
                    st.success("Thanks, the answer was flagged for review.")

    with sql_tab:
        st.code(answer.raw_sql or "(nothing generated)", language="sql")
        if answer.verdict and answer.verdict.ok and answer.verdict.sql.strip() != (answer.raw_sql or "").strip():
            st.caption("Query as run, after the guardrails applied the row limit:")
            st.code(answer.verdict.sql, language="sql")

    with checked_tab:
        st.markdown(f"**Prompting strategy:** {answer.variant}")
        st.markdown(f"**Tables shown to the model:** {', '.join(tables) if tables else 'not available'}")
        if answer.verdict is None:
            st.markdown("**Guardrail verdict:** not reached, because no SQL was generated")
        else:
            label = "allowed" if answer.verdict.ok else "blocked"
            st.markdown(f"**Guardrail verdict:** {label} ({answer.verdict.reason})")
        st.markdown(f"**Total latency:** {answer.latency_ms:.0f} ms")
        if answer.result is not None:
            rows = len(answer.result.frame) if answer.result.ok else 0
            st.markdown(f"**Query time:** {answer.result.latency_ms:.0f} ms")
            st.markdown(f"**Rows returned:** {rows}")


def render_schema_answer(answer, dialect):
    if answer.check.status == "blocked":
        st.error(f"The SQL sanity check stopped this query, so it is not shown. Reason: {answer.check.reason}")
    else:
        if answer.check.status == "unparsed":
            st.warning(
                f"sqlglot could not parse this as {dialect} SQL, so the safety checks are incomplete. "
                f"Review it carefully before you run it. Details: {answer.check.reason}"
            )
        st.code(answer.sql or "(nothing generated)", language="sql")
    if answer.explanation:
        st.markdown(answer.explanation)
    with st.expander("Exactly what was sent about your schema"):
        st.code(answer.schema_sent, language="text")


def run_schema_mode():
    st.warning(PRIVACY_WARNING)
    st.caption("This mode never connects to or runs anything on your database. Copy the SQL and run it yourself.")
    dialect = st.selectbox("SQL dialect", list(DIALECTS), key="schema_dialect")
    schema_paste = st.text_area(
        "Your schema: CREATE TABLE statements, or one table per line such as orders: order_id, customer_id, total",
        key="schema_paste",
        height=240,
        max_chars=MAX_SCHEMA_CHARS,
    )
    question = st.text_area("Your question", key="schema_question", height=100, max_chars=300)
    if st.button("Write SQL", type="primary", key="ask") and question.strip():
        live_calls = st.session_state.get("live_calls", 0)
        if live_calls >= MAX_LIVE_CALLS:
            st.warning("This session reached its live-call limit.")
            return
        st.session_state.pop("schema_last", None)
        try:
            llm = get_schema_llm()
        except LLMError as error:
            st.error(f"Schema mode uses Google's Gemini API. {error}")
            return
        try:
            with st.spinner("Writing SQL..."):
                st.session_state["schema_last"] = (write_sql(question, schema_paste, dialect, llm), dialect)
            st.session_state["live_calls"] = live_calls + 1
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.session_state["live_calls"] = live_calls + 1
            if is_quota_error(str(error)):
                st.warning("Free-tier limit reached. Please try again later.")
            else:
                st.error(f"The language model could not be reached just now ({type(error).__name__}). Please try again.")
    if st.session_state.get("schema_last"):
        render_schema_answer(*st.session_state["schema_last"])


inject_css()
render_header()
if DEMO_MODE:
    st.info(f"Demo mode: live model calls are limited to {MAX_LIVE_CALLS} per session. Answers to repeated questions are cached.")

with st.sidebar:
    mode = st.radio("Mode", MODES, key="mode")

if mode == "Schema-only assistant":
    run_schema_mode()
    st.stop()

db_path = ensure_database()
try:
    llm, retriever = get_services()
except LLMError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Sample questions")
    for index, sample in enumerate(SAMPLE_QUESTIONS):
        if st.button(sample, key=f"sample_{index}"):
            st.session_state["question"] = sample
    st.divider()
    variant = st.selectbox("Prompting strategy", VARIANTS, index=VARIANTS.index("retrieval_few_shot"))
    with st.expander("Tables available"):
        schema = load_schema()
        for name in schema.table_names:
            st.markdown(f"**{name}** - {schema.tables[name]['description']}")

render_how_it_works()
question = st.text_area("Your question", key="question", height=120, max_chars=300)
if st.button("Ask", type="primary", key="ask") and question.strip():
    live_calls = st.session_state.get("live_calls", 0)
    if live_calls >= MAX_LIVE_CALLS:
        st.warning("This session reached its live-call limit. Try one of the sample questions, which may be cached.")
    else:
        before = getattr(llm, "misses", 0)
        with st.spinner("Thinking..."):
            st.session_state["last"] = answer_question(question.strip(), variant, llm, retriever, db_path=db_path)
            st.session_state["last_tables"] = tables_used(question.strip(), variant, retriever)
        if getattr(llm, "misses", 0) > before:
            st.session_state["live_calls"] = live_calls + 1

if st.session_state.get("last"):
    render(st.session_state["last"], st.session_state.get("last_tables", []))
