import re

from src.examples import FEW_SHOT_EXAMPLES
from src.schema import load_schema

VARIANTS = ("zero_shot_full_schema", "few_shot_full_schema", "retrieval_few_shot")

INSTRUCTIONS_TEMPLATE = """You are an expert analytics engineer who writes DuckDB SQL for {domain}.

Rules:
- Return exactly one SELECT statement (a WITH ... SELECT is allowed) inside a single ```sql fenced block and nothing else.
- Use only the tables and columns listed in the schema. Never invent a column or a table.
- Never modify data and never use files, extensions or system functions.
- Join with explicit JOIN ... ON using the join keys given.
- Round monetary and ratio results to 2 decimals with ROUND.
- Give every computed column a clear alias.
- When the question asks for a list, ranking or top N, add a deterministic ORDER BY with a tie-breaker."""
PHARMA_DOMAIN = "a pharma commercial analytics warehouse"
INSTRUCTIONS = INSTRUCTIONS_TEMPLATE.format(domain=PHARMA_DOMAIN)

_FENCED = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_BARE = re.compile(r"\b(with|select)\b.*", re.S | re.I)


def extract_sql(text):
    if not text:
        return ""
    fenced = _FENCED.findall(text)
    if fenced:
        return fenced[0].strip().rstrip(";").strip()
    bare = _BARE.search(text)
    return bare.group(0).strip().rstrip(";").strip() if bare else text.strip()


def render_examples(examples=None):
    examples = FEW_SHOT_EXAMPLES if examples is None else examples
    blocks = [f"Question: {e['question']}\n```sql\n{e['sql']}\n```" for e in examples]
    return "\n\n".join(blocks)


def build_prompt(question, variant, retriever=None, schema=None, examples=None, domain=None):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    schema = schema or load_schema()
    if variant == "retrieval_few_shot":
        if retriever is None:
            raise ValueError("retrieval_few_shot needs a retriever")
        schema_text = retriever.context_for(question)
    else:
        schema_text = schema.render(schema.table_names)
    instructions = INSTRUCTIONS if domain is None else INSTRUCTIONS_TEMPLATE.format(domain=domain)
    parts = [instructions, "SCHEMA:\n" + schema_text, "NOTES:\n" + schema.render_notes()]
    if variant != "zero_shot_full_schema":
        parts.append("EXAMPLES:\n" + render_examples(examples))
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def generate_sql(question, variant, llm, retriever=None, schema=None, examples=None, domain=None):
    prompt = build_prompt(question, variant, retriever, schema, examples, domain)
    return extract_sql(llm.complete(prompt))
