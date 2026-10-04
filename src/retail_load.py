"""Build data/retail.duckdb from the UCI Online Retail dataset (id 352, CC BY 4.0).

Usage: python -m src.retail_load [--raw PATH] [--db PATH]

Downloads the raw file into data/raw/ if it is missing, cleans it with pandas and writes a small
star schema. Every cleaning decision is documented in docs/RETAIL_DATA_NOTES.md.
"""

import argparse
import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
RAW_PATH = RAW_DIR / "Online Retail.xlsx"
RETAIL_DB_PATH = DATA_DIR / "retail.duckdb"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
CITATION = (
    "Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33."
)
TABLES = ("dim_customer", "dim_product", "invoices", "invoice_lines")


@dataclass
class LoadReport:
    raw_rows: int = 0
    steps: list = field(default_factory=list)  # (decision, count, unit, action)
    table_rows: dict = field(default_factory=dict)

    def add(self, decision, count, action, unit="rows"):
        self.steps.append((decision, int(count), unit, action))

    def lines(self):
        out = [f"raw rows: {self.raw_rows:,}"]
        out += [f"{decision}: {count:,} {unit} {action}" for decision, count, unit, action in self.steps]
        out += [f"{name}: {rows:,} rows" for name, rows in self.table_rows.items()]
        return out


def download_raw(path=RAW_PATH, url=DOWNLOAD_URL):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open("Online Retail.xlsx") as source:
        path.write_bytes(source.read())
    return path


def read_raw(path=RAW_PATH):
    return pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str})


def _mode_per_group(frame, key, value):
    """Most frequent value per key; ties go to the alphabetically first value. Missing values are ignored."""
    counts = frame.dropna(subset=[value]).groupby([key, value]).size().reset_index(name="n")
    counts = counts.sort_values([key, "n", value], ascending=[True, False, True])
    return counts.drop_duplicates(key).set_index(key)[value]


def build_tables(raw):
    report = LoadReport(raw_rows=len(raw))
    df = raw.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

    duplicated = df.duplicated()
    report.add("exact duplicate rows", duplicated.sum(), "removed (kept the first copy)")
    df = df[~duplicated]

    adjustments = df["InvoiceNo"].str.startswith("A")
    report.add("bad-debt adjustment invoices (InvoiceNo starts with A)", adjustments.sum(), "removed")
    df = df[~adjustments]

    bad_price = df["UnitPrice"] <= 0
    report.add(
        "zero or negative unit price (stock write-offs, damages, free items)",
        bad_price.sum(),
        f"removed, including {int((bad_price & (df['Quantity'] < 0)).sum()):,} with a negative quantity",
    )
    df = df[~bad_price]

    cancelled = df["InvoiceNo"].str.startswith("C")
    negative = df["Quantity"] < 0
    if (negative & ~cancelled).any():
        raise ValueError("negative quantities outside cancellations remain after cleaning")
    if (cancelled & ~negative).any():
        raise ValueError("cancellation lines with a non-negative quantity were found")
    report.add("cancellation lines (InvoiceNo starts with C, negative quantity)", cancelled.sum(), "kept, is_cancelled = true")
    report.add("lines with a missing CustomerID", df["CustomerID"].isna().sum(), "kept with customer_id NULL")

    lower = df["StockCode"] != df["StockCode"].str.upper()
    report.add("stock codes with lower-case letters", lower.sum(), "upper-cased so 85123a and 85123A are one product")
    df = df.assign(
        StockCode=df["StockCode"].str.strip().str.upper(),
        Description=df["Description"].str.strip().replace("", pd.NA),
        CustomerID=df["CustomerID"].astype("Int64"),
        is_cancelled=cancelled,
    )

    multi_time = df.groupby("InvoiceNo")["InvoiceDate"].nunique()
    report.add("invoices whose lines have more than one timestamp", (multi_time > 1).sum(), "use the earliest timestamp", unit="invoices")
    invoices = (
        df.groupby("InvoiceNo", sort=True)
        .agg(invoice_date=("InvoiceDate", "min"), customer_id=("CustomerID", "first"), country=("Country", "first"),
             is_cancelled=("is_cancelled", "first"))
        .reset_index()
        .rename(columns={"InvoiceNo": "invoice_no"})
    )

    known = df.dropna(subset=["CustomerID"])
    multi_country = known.groupby("CustomerID")["Country"].nunique()
    report.add("customers seen in more than one country", (multi_country > 1).sum(), "use their most frequent country", unit="customers")
    dim_customer = _mode_per_group(known, "CustomerID", "Country").rename("country").reset_index()
    dim_customer = dim_customer.rename(columns={"CustomerID": "customer_id"}).sort_values("customer_id")

    descriptions = _mode_per_group(df, "StockCode", "Description")
    dim_product = pd.DataFrame({"stock_code": sorted(df["StockCode"].unique())})
    dim_product["description"] = dim_product["stock_code"].map(descriptions)
    report.add("products with no description on any kept line", dim_product["description"].isna().sum(), "kept with description NULL", unit="products")

    lines = pd.DataFrame(
        {
            "invoice_no": df["InvoiceNo"],
            "stock_code": df["StockCode"],
            "quantity": df["Quantity"].astype("int64"),
            "unit_price": df["UnitPrice"].astype("float64"),
        }
    )
    lines["line_total"] = (lines["quantity"] * lines["unit_price"]).round(2)

    tables = {
        "dim_customer": dim_customer.reset_index(drop=True),
        "dim_product": dim_product,
        "invoices": invoices,
        "invoice_lines": lines.reset_index(drop=True),
    }
    report.add("invoices with a missing CustomerID", invoices["customer_id"].isna().sum(), "kept with customer_id NULL", unit="invoices")
    report.table_rows = {name: len(frame) for name, frame in tables.items()}
    return tables, report


DDL = """
CREATE TABLE dim_customer (customer_id INTEGER PRIMARY KEY, country VARCHAR NOT NULL);
CREATE TABLE dim_product (stock_code VARCHAR PRIMARY KEY, description VARCHAR);
CREATE TABLE invoices (
    invoice_no VARCHAR PRIMARY KEY,
    invoice_date TIMESTAMP NOT NULL,
    customer_id INTEGER,
    country VARCHAR NOT NULL,
    is_cancelled BOOLEAN NOT NULL
);
CREATE TABLE invoice_lines (
    invoice_no VARCHAR NOT NULL,
    stock_code VARCHAR NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DOUBLE NOT NULL,
    line_total DOUBLE NOT NULL
);
"""


def write_db(tables, path=RETAIL_DB_PATH):
    import duckdb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        con.execute(DDL)
        for name in TABLES:
            frame = tables[name]
            con.register("frame", frame)
            con.execute(f"INSERT INTO {name} SELECT {', '.join(frame.columns)} FROM frame")
            con.unregister("frame")
    finally:
        con.close()
    tmp.replace(path)
    return path


def build(raw_path=RAW_PATH, db_path=RETAIL_DB_PATH, download=True):
    raw_path = Path(raw_path)
    if not raw_path.exists():
        if not download:
            raise FileNotFoundError(f"{raw_path} is missing. Run python -m src.retail_load to download it.")
        download_raw(raw_path)
    tables, report = build_tables(read_raw(raw_path))
    write_db(tables, db_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Build data/retail.duckdb from the UCI Online Retail dataset.")
    parser.add_argument("--raw", default=str(RAW_PATH))
    parser.add_argument("--db", default=str(RETAIL_DB_PATH))
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading a missing raw file")
    args = parser.parse_args()
    report = build(args.raw, args.db, download=not args.no_download)
    print("\n".join(report.lines()))
    print(f"wrote {args.db}")
    print(f"Source: {CITATION} Licence: CC BY 4.0.")


if __name__ == "__main__":
    main()
