"""The datasets the app can query. Each has its own database, schema docs, examples and table allowlist."""

from dataclasses import dataclass
from pathlib import Path

from src.config import DB_PATH, EVAL_DIR, GOLD_PATH, ROOT, SCHEMA_DOCS_PATH
from src.examples import FEW_SHOT_EXAMPLES
from src.guardrails import ALLOWED_TABLES

RETAIL_SOURCE = (
    "Source: UCI Machine Learning Repository, Online Retail dataset, CC BY 4.0. "
    "Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33."
)

RETAIL_EXAMPLES = [
    {
        "question": "How many non-cancelled invoices came from Germany?",
        "sql": "SELECT COUNT(*) AS invoice_count\nFROM invoices\nWHERE country = 'Germany' AND NOT is_cancelled",
    },
    {
        "question": "What are the 3 products with the highest net revenue in France?",
        "sql": (
            "SELECT p.stock_code, p.description, ROUND(SUM(l.line_total), 2) AS net_revenue\n"
            "FROM invoice_lines l\n"
            "JOIN invoices i ON l.invoice_no = i.invoice_no\n"
            "JOIN dim_product p ON l.stock_code = p.stock_code\n"
            "WHERE i.country = 'France'\n"
            "GROUP BY p.stock_code, p.description\n"
            "ORDER BY net_revenue DESC, p.stock_code\n"
            "LIMIT 3"
        ),
    },
    {
        "question": "For Germany and France, show the top 2 customers by number of non-cancelled invoices in each country.",
        "sql": (
            "WITH counts AS (\n"
            "  SELECT country, customer_id, COUNT(*) AS invoice_count\n"
            "  FROM invoices\n"
            "  WHERE NOT is_cancelled AND customer_id IS NOT NULL AND country IN ('Germany', 'France')\n"
            "  GROUP BY country, customer_id\n"
            "), ranked AS (\n"
            "  SELECT country, customer_id, invoice_count,\n"
            "         ROW_NUMBER() OVER (PARTITION BY country ORDER BY invoice_count DESC, customer_id) AS rn\n"
            "  FROM counts\n"
            ")\n"
            "SELECT country, customer_id, invoice_count\n"
            "FROM ranked\n"
            "WHERE rn <= 2\n"
            "ORDER BY country, rn"
        ),
    },
]


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    domain: str
    db_path: Path
    schema_docs_path: Path
    gold_path: Path
    examples: tuple
    allowed_tables: frozenset
    sample_questions: tuple
    source: str | None
    report_suffix: str

    def schema(self):
        from src.schema import load_schema

        return load_schema(str(self.schema_docs_path))


PHARMA = Dataset(
    key="pharma",
    label="Pharma synthetic",
    domain="a pharma commercial analytics warehouse",
    db_path=DB_PATH,
    schema_docs_path=SCHEMA_DOCS_PATH,
    gold_path=GOLD_PATH,
    examples=tuple(FEW_SHOT_EXAMPLES),
    allowed_tables=ALLOWED_TABLES,
    sample_questions=(
        "What was the total revenue for each product in 2025?",
        "What is the total revenue for each region?",
        "Which 5 territories had the highest total revenue?",
        "Show total revenue by month together with the change from the previous month.",
        "Which territories missed their total target revenue in Q4 2025? Show actual revenue and target.",
        "What are the top 3 territories by total revenue within each region?",
        "What percentage of total revenue does each region contribute?",
        "How many calls were made to HCPs in each tier?",
    ),
    source=None,
    report_suffix="",
)

RETAIL = Dataset(
    key="retail",
    label="Online Retail real",
    domain="a UK online gift retailer's sales database (real transactions, Dec 2010 to Dec 2011)",
    db_path=Path(ROOT / "data" / "retail.duckdb"),
    schema_docs_path=Path(ROOT / "src" / "schema_docs_retail.yaml"),
    gold_path=Path(EVAL_DIR / "gold_retail.yaml"),
    examples=tuple(RETAIL_EXAMPLES),
    allowed_tables=frozenset({"dim_customer", "dim_product", "invoices", "invoice_lines"}),
    sample_questions=(
        "What was the net revenue in each month of 2011?",
        "Which 10 products earned the most net revenue?",
        "How many customers are there in each country?",
        "What share of invoices were cancelled in each country?",
        "What is the average net revenue per non-cancelled invoice by country?",
        "Which day of the week has the most non-cancelled invoices?",
    ),
    source=RETAIL_SOURCE,
    report_suffix="_retail",
)

DATASETS = {d.key: d for d in (PHARMA, RETAIL)}
