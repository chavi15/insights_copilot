"""Build data/pharmacy.duckdb from the Kaggle dataset milanzdravkovic/pharma-sales-data (CC BY-NC 4.0).

Usage: python -m src.pharmacy_load [--raw PATH] [--db PATH]

Download the dataset from Kaggle yourself and copy salesdaily.csv to data/raw/salesdaily.csv.
Uses salesdaily.csv only: daily quantities sold by one pharmacy for 8 ATC drug categories,
2014-01-02 to 2019-10-08. The wide ATC columns are melted into a long fact table.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

DAILY_FILE = "salesdaily.csv"
RAW_PATH = DATA_DIR / "raw" / DAILY_FILE
PHARMACY_DB_PATH = DATA_DIR / "pharmacy.duckdb"
SOURCE = (
    "Source: Kaggle, Pharma sales data by Milan Zdravković (milanzdravkovic/pharma-sales-data), "
    "CC BY-NC 4.0. https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data"
)
TABLES = ("dim_date", "dim_atc", "fact_sales_daily")

# (atc_code, description, therapeutic_group). Groups are the WHO ATC second-level names.
ATC = [
    ("M01AB", "Acetic acid derivatives and related substances (NSAID)", "Anti-inflammatory and antirheumatic products"),
    ("M01AE", "Propionic acid derivatives (NSAID)", "Anti-inflammatory and antirheumatic products"),
    ("N02BA", "Salicylic acid and derivatives (analgesics)", "Analgesics"),
    ("N02BE", "Pyrazolones and anilides (analgesics), listed as N02BE/B", "Analgesics"),
    ("N05B", "Anxiolytics", "Psycholeptics"),
    ("N05C", "Hypnotics and sedatives", "Psycholeptics"),
    ("R03", "Drugs for obstructive airway diseases", "Drugs for obstructive airway diseases"),
    ("R06", "Antihistamines for systemic use", "Antihistamines for systemic use"),
]
ATC_CODES = [code for code, _, _ in ATC]
EXPECTED_COLUMNS = ["datum", *ATC_CODES, "Year", "Month", "Hour", "Weekday Name"]


@dataclass
class LoadReport:
    raw_rows: int = 0
    notes: list = field(default_factory=list)
    table_rows: dict = field(default_factory=dict)

    def lines(self):
        out = [f"raw rows ({DAILY_FILE}): {self.raw_rows:,}"]
        out += self.notes
        out += [f"{name}: {rows:,} rows" for name, rows in self.table_rows.items()]
        return out


def read_raw(path):
    raw = pd.read_csv(path)
    if list(raw.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns in {path}: {list(raw.columns)}; expected {EXPECTED_COLUMNS}")
    return raw


def build_tables(raw):
    report = LoadReport(raw_rows=len(raw))
    dates = pd.to_datetime(raw["datum"], format="%m/%d/%Y")
    if dates.duplicated().any():
        raise ValueError("duplicate dates in the daily file")
    if not (dates.dt.day_name() == raw["Weekday Name"]).all():
        raise ValueError("Weekday Name does not match the parsed date; check the date format")
    if (raw[ATC_CODES] < 0).any().any() or raw[ATC_CODES].isna().any().any():
        raise ValueError("negative or missing quantities in the daily file")

    zero_days = int((raw[ATC_CODES].sum(axis=1) == 0).sum())
    report.notes += [
        f"date range: {dates.min().date()} to {dates.max().date()}, "
        f"{len(pd.date_range(dates.min(), dates.max()).difference(dates))} missing days",
        f"days with zero sales in every category: {zero_days} (kept, mostly public holidays)",
        "dropped columns: Year, Month and Weekday Name (rebuilt in dim_date from the date) and Hour (not meaningful per day)",
    ]

    dim_date = pd.DataFrame({"date": dates.dt.date})
    dim_date["year"] = dates.dt.year
    dim_date["month"] = dates.dt.month
    dim_date["quarter"] = dates.dt.quarter
    dim_date["weekday"] = dates.dt.day_name()
    dim_date["is_weekend"] = dates.dt.dayofweek >= 5
    dim_date = dim_date.sort_values("date").reset_index(drop=True)

    dim_atc = pd.DataFrame(ATC, columns=["atc_code", "description", "therapeutic_group"])

    wide = raw[ATC_CODES].assign(date=dates.dt.date)
    fact = wide.melt(id_vars="date", value_vars=ATC_CODES, var_name="atc_code", value_name="quantity")
    fact = fact.sort_values(["date", "atc_code"]).reset_index(drop=True)
    report.notes.append(f"melted {len(raw):,} days x {len(ATC_CODES)} ATC columns into {len(fact):,} rows")

    tables = {"dim_date": dim_date, "dim_atc": dim_atc, "fact_sales_daily": fact}
    report.table_rows = {name: len(frame) for name, frame in tables.items()}
    return tables, report


DDL = """
CREATE TABLE dim_date (
    date DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    weekday VARCHAR NOT NULL,
    is_weekend BOOLEAN NOT NULL
);
CREATE TABLE dim_atc (atc_code VARCHAR PRIMARY KEY, description VARCHAR NOT NULL, therapeutic_group VARCHAR NOT NULL);
CREATE TABLE fact_sales_daily (
    date DATE NOT NULL REFERENCES dim_date (date),
    atc_code VARCHAR NOT NULL REFERENCES dim_atc (atc_code),
    quantity DECIMAL(18, 9) NOT NULL,  -- exact: raw values have up to 9 decimals, so sums never drift
    PRIMARY KEY (date, atc_code)
);
"""


def write_db(tables, path=PHARMACY_DB_PATH):
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


def build(raw_path=RAW_PATH, db_path=PHARMACY_DB_PATH):
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download milanzdravkovic/pharma-sales-data from Kaggle and copy {DAILY_FILE} there."
        )
    tables, report = build_tables(read_raw(path))
    write_db(tables, db_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Build data/pharmacy.duckdb from the Kaggle pharma sales daily file.")
    parser.add_argument("--raw", default=str(RAW_PATH))
    parser.add_argument("--db", default=str(PHARMACY_DB_PATH))
    args = parser.parse_args()
    report = build(args.raw, args.db)
    print("\n".join(report.lines()))
    print(f"wrote {args.db}")
    print(SOURCE)


if __name__ == "__main__":
    main()
