"""Independent pandas answers for eval/gold_pharmacy.yaml, computed from the raw wide salesdaily.csv.

This deliberately does not import src.pharmacy_load and never melts the data: each answer works on the
original one-column-per-ATC-code layout, so it also cross-checks the loader's melt into long format.
"""

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

CODES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
GROUPS = {
    "M01AB": "Anti-inflammatory and antirheumatic products",
    "M01AE": "Anti-inflammatory and antirheumatic products",
    "N02BA": "Analgesics",
    "N02BE": "Analgesics",
    "N05B": "Psycholeptics",
    "N05C": "Psycholeptics",
    "R03": "Drugs for obstructive airway diseases",
    "R06": "Antihistamines for systemic use",
}


def exact_round(value):
    """Round half away from zero, like DuckDB's ROUND on DECIMAL."""
    return float(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def load_wide(path):
    """Quantities are kept as exact Decimals parsed from the CSV text, so sums are exact."""
    wide = pd.read_csv(path, dtype={c: str for c in CODES})
    for code in CODES:
        wide[code] = wide[code].map(Decimal)
    when = pd.to_datetime(wide["datum"], format="%m/%d/%Y")
    wide["year"], wide["month"], wide["quarter"] = when.dt.year, when.dt.month, when.dt.quarter
    wide["weekday"] = when.dt.day_name()
    wide["weekend"] = when.dt.dayofweek >= 5
    return wide


def answers(wide):
    out = {}
    out["pe01"] = pd.DataFrame({"n": [len(wide)]})
    out["pe02"] = pd.DataFrame({"code": CODES, "q": [exact_round(wide[c].sum()) for c in CODES]})
    out["pe03"] = pd.DataFrame({"n": [int((wide["N05C"] == 0).sum())]})
    counts = pd.Series(GROUPS).value_counts()
    out["pe04"] = pd.DataFrame({"group": counts.index, "n": counts.values})

    rows = []
    for group in sorted(set(GROUPS.values())):
        cols = [c for c, g in GROUPS.items() if g == group]
        for year, part in wide.groupby("year"):
            rows.append((group, year, exact_round(sum(part[c].sum() for c in cols))))
    out["pm01"] = pd.DataFrame(rows, columns=["group", "year", "q"])

    by_weekday = wide.groupby("weekday")["N02BE"].agg(lambda s: round(float(s.sum()) / len(s), 2))
    out["pm02"] = by_weekday.reset_index()

    r06 = wide[wide["year"] == 2018].groupby("quarter")["R06"].agg(lambda s: exact_round(s.sum()))
    out["pm03"] = r06.reset_index()

    r03 = wide.groupby(["year", "month"])["R03"].agg(lambda s: exact_round(s.sum())).reset_index()
    r03 = r03.sort_values(["R03", "year", "month"], ascending=[False, True, True]).head(3)
    out["pm04"] = r03

    rows = []
    for code in CODES:
        by_month = wide.groupby("month")[code].agg(lambda s: float(s.sum()) / len(s))
        best = max(by_month.index, key=lambda m: (by_month[m], -m))
        rows.append((code, best, round(by_month[best], 2)))
    out["ph01"] = pd.DataFrame(rows, columns=["code", "month", "avg"])

    wide = wide.assign(day_total=[sum(row) for row in wide[CODES].itertuples(index=False)])
    by_weekend = wide.groupby("weekend")["day_total"].agg(lambda s: round(float(s.sum()) / len(s), 2))
    out["ph02"] = by_weekend.reset_index()
    return out
