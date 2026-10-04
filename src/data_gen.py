import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS = [
    (1, "Cardiva", "Cardiology", "2023-01-01"),
    (2, "Neurolux", "Neurology", "2023-01-01"),
    (3, "Respira", "Respiratory", "2023-01-01"),
    (4, "Gastrex", "Gastroenterology", "2023-01-01"),
    (5, "Oncora", "Oncology", "2023-06-01"),
    (6, "Dermis", "Dermatology", "2024-04-01"),
    (7, "Immunix", "Immunology", "2024-10-01"),
    (8, "Diabetol", "Endocrinology", "2025-03-01"),
]
PRICE = {1: 42.0, 2: 65.0, 3: 38.0, 4: 30.0, 5: 180.0, 6: 55.0, 7: 120.0, 8: 48.0}
REGIONS = ["North", "South", "East", "West"]
SPECIALTY_PRODUCT = {
    "Cardiologist": 1,
    "Neurologist": 2,
    "Pulmonologist": 3,
    "Gastroenterologist": 4,
    "Oncologist": 5,
    "Dermatologist": 6,
    "Rheumatologist": 7,
    "Endocrinologist": 8,
    "General Physician": 0,
}
SPECIALTIES = list(SPECIALTY_PRODUCT)
SPECIALTY_WEIGHTS = [0.12, 0.10, 0.10, 0.09, 0.08, 0.09, 0.07, 0.09, 0.26]
CHANNELS = ["In-person", "Phone", "Email", "Video"]
CHANNEL_WEIGHTS = [0.45, 0.2, 0.2, 0.15]
TIERS = ["A", "B", "C"]
TIER_WEIGHTS = [0.15, 0.35, 0.50]
TIER_VOLUME = {"A": 2.2, "B": 1.2, "C": 0.6}
TIER_CALL_RATE = {"A": 1.4, "B": 0.8, "C": 0.35}
FIRST_NAMES = ["Aarav", "Meera", "Rohan", "Anika", "Kabir", "Isha", "Vihaan", "Diya", "Arjun", "Nisha", "Rahul", "Priya", "Sameer", "Tara", "Karan", "Neha", "Dev", "Sana", "Imran", "Leela"]
LAST_NAMES = ["Sharma", "Iyer", "Reddy", "Khan", "Gupta", "Nair", "Singh", "Das", "Mehta", "Joshi", "Patel", "Menon", "Bose", "Kapoor", "Rao", "Verma", "Shetty", "Malhotra", "Sethi", "Pillai"]
START_MONTH = "2024-01-01"
N_MONTHS = 24
SQLITE_INDEXES = [
    ("fact_sales", "hcp_id"), ("fact_sales", "date_id"), ("fact_sales", "product_id"),
    ("fact_calls", "hcp_id"), ("fact_calls", "date_id"), ("dim_hcp", "territory_id"),
    ("targets", "territory_id"), ("targets", "date_id"), ("targets", "product_id"),
]
TABLE_ORDER = ["dim_product", "dim_territory", "dim_hcp", "dim_date", "fact_sales", "fact_calls", "targets"]


def _names(rng, n, prefix=""):
    first = rng.choice(FIRST_NAMES, n)
    last = rng.choice(LAST_NAMES, n)
    return [f"{prefix}{f} {l}" for f, l in zip(first, last)]


def build_frames(seed=42, n_hcp=2000, n_territories=40):
    rng = np.random.default_rng(seed)

    months = pd.date_range(START_MONTH, periods=N_MONTHS, freq="MS")
    dim_date = pd.DataFrame(
        {
            "date_id": (months.year * 100 + months.month).astype(int),
            "month_start": [m.date() for m in months],
            "year": months.year.astype(int),
            "quarter": months.quarter.astype(int),
            "month": months.month.astype(int),
            "month_name": months.strftime("%b"),
        }
    )

    dim_product = pd.DataFrame(PRODUCTS, columns=["product_id", "product_name", "therapeutic_area", "launch_date"])
    dim_product["launch_date"] = pd.to_datetime(dim_product["launch_date"]).dt.date

    ids = np.arange(1, n_territories + 1)
    per_region = n_territories // len(REGIONS)
    region = [REGIONS[min((i - 1) // per_region, len(REGIONS) - 1)] for i in ids]
    dim_territory = pd.DataFrame(
        {
            "territory_id": ids,
            "territory_name": [f"{r} T{((i - 1) % per_region) + 1:02d}" for r, i in zip(region, ids)],
            "region": region,
            "rep_name": _names(rng, n_territories),
        }
    )

    hcp_ids = np.arange(1, n_hcp + 1)
    dim_hcp = pd.DataFrame(
        {
            "hcp_id": hcp_ids,
            "hcp_name": _names(rng, n_hcp, "Dr. "),
            "specialty": rng.choice(SPECIALTIES, n_hcp, p=SPECIALTY_WEIGHTS),
            "tier": rng.choice(TIERS, n_hcp, p=TIER_WEIGHTS),
            "territory_id": rng.integers(1, n_territories + 1, n_hcp),
        }
    )

    product_ids = [p[0] for p in PRODUCTS]
    pairs = []
    for hcp_id, specialty in zip(dim_hcp["hcp_id"], dim_hcp["specialty"]):
        primary = SPECIALTY_PRODUCT[specialty]
        pool = [p for p in product_ids if p != primary]
        if primary:
            chosen = [primary] + list(rng.choice(pool, 2, replace=False))
        else:
            chosen = list(rng.choice(product_ids, 3, replace=False))
        for p in chosen:
            pairs.append((hcp_id, int(p), 1.0 if p == primary else 0.0))
    hcp_product = pd.DataFrame(pairs, columns=["hcp_id", "product_id", "is_primary"])

    territory_mult = pd.Series(rng.uniform(0.7, 1.3, n_territories), index=ids)
    base = hcp_product.merge(dim_hcp[["hcp_id", "tier", "territory_id"]], on="hcp_id")
    grid = base.merge(dim_date[["date_id", "year", "month"]], how="cross")
    launch = dim_product.set_index("product_id")["launch_date"]
    launch_idx = launch.map(lambda d: d.year * 12 + d.month)
    months_since = grid["year"] * 12 + grid["month"] - grid["product_id"].map(launch_idx)
    ramp = np.where(months_since < 0, 0.0, np.minimum(1.0, (months_since + 1) / 6.0))
    seasonal = 1 + 0.12 * np.cos(2 * np.pi * (grid["month"] - 2) / 12) + np.where(grid["month"] >= 10, 0.06, 0.0)
    lam = (
        7.0
        * grid["tier"].map(TIER_VOLUME)
        * seasonal
        * ramp
        * (1 + 0.5 * grid["is_primary"])
        * grid["territory_id"].map(territory_mult)
    )
    units = rng.poisson(lam.to_numpy())
    discount = rng.uniform(0.92, 1.0, len(grid))
    sales = grid.assign(units=units, discount=discount)
    sales = sales[sales["units"] > 0].copy()
    sales["revenue"] = (sales["units"] * sales["product_id"].map(PRICE) * sales["discount"]).round(2)
    sales = sales.sort_values(["date_id", "hcp_id", "product_id"]).reset_index(drop=True)
    sales["sale_id"] = np.arange(1, len(sales) + 1)
    fact_sales = sales[["sale_id", "date_id", "hcp_id", "product_id", "units", "revenue"]].astype(
        {"sale_id": "int64", "date_id": "int64", "hcp_id": "int64", "product_id": "int64", "units": "int64"}
    )

    hcp_month = dim_hcp[["hcp_id", "tier"]].merge(dim_date[["date_id"]], how="cross")
    n_calls = rng.poisson(hcp_month["tier"].map(TIER_CALL_RATE).to_numpy())
    calls = hcp_month.loc[hcp_month.index.repeat(n_calls), ["date_id", "hcp_id"]].reset_index(drop=True)
    calls["channel"] = rng.choice(CHANNELS, len(calls), p=CHANNEL_WEIGHTS)
    calls["duration_minutes"] = rng.integers(5, 46, len(calls))
    calls = calls.sort_values(["date_id", "hcp_id"]).reset_index(drop=True)
    calls["call_id"] = np.arange(1, len(calls) + 1)
    fact_calls = calls[["call_id", "date_id", "hcp_id", "channel", "duration_minutes"]].astype(
        {"call_id": "int64", "date_id": "int64", "hcp_id": "int64", "duration_minutes": "int64"}
    )

    actual = (
        fact_sales.merge(dim_hcp[["hcp_id", "territory_id"]], on="hcp_id")
        .groupby(["territory_id", "product_id", "date_id"], as_index=False)["revenue"]
        .sum()
    )
    actual = actual.merge(dim_date[["date_id", "year", "quarter"]], on="date_id")
    keys = actual[["territory_id", "product_id", "year", "quarter"]].drop_duplicates().reset_index(drop=True)
    keys["factor"] = rng.uniform(0.82, 1.22, len(keys))
    actual = actual.merge(keys, on=["territory_id", "product_id", "year", "quarter"])
    actual["target_revenue"] = (actual["revenue"] * actual["factor"]).round(2)
    targets = actual[["territory_id", "product_id", "date_id", "target_revenue"]].sort_values(
        ["date_id", "territory_id", "product_id"]
    ).reset_index(drop=True)
    targets = targets.astype({"territory_id": "int64", "product_id": "int64", "date_id": "int64"})

    return {
        "dim_product": dim_product,
        "dim_territory": dim_territory,
        "dim_hcp": dim_hcp.astype({"hcp_id": "int64", "territory_id": "int64"}),
        "dim_date": dim_date,
        "fact_sales": fact_sales,
        "fact_calls": fact_calls,
        "targets": targets,
    }


def write_duckdb(frames, path):
    import duckdb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    try:
        for name in TABLE_ORDER:
            con.register("_frame", frames[name])
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _frame")
            con.unregister("_frame")
    finally:
        con.close()


def write_sqlite(frames, path):
    import sqlite3

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    try:
        for name in TABLE_ORDER:
            frame = frames[name].copy()
            for col in frame.columns:
                if frame[col].dtype == object and len(frame) and hasattr(frame[col].iloc[0], "isoformat"):
                    frame[col] = frame[col].map(lambda d: d.isoformat())
            frame.to_sql(name, con, index=False)
        for table, column in SQLITE_INDEXES:
            con.execute(f"CREATE INDEX idx_{table}_{column} ON {table}({column})")
        con.commit()
    finally:
        con.close()


def write_db(frames, path):
    if str(path).endswith(".duckdb"):
        write_duckdb(frames, path)
    else:
        write_sqlite(frames, path)


def main():
    from src.config import DB_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frames = build_frames(seed=args.seed)
    write_db(frames, args.db)
    for name in TABLE_ORDER:
        print(f"{name}: {len(frames[name])} rows")
    print(f"written to {args.db}")


if __name__ == "__main__":
    main()
