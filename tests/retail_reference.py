"""Independent pandas answers for eval/gold_retail.yaml, computed straight from the raw UCI file.

This deliberately does not import src.retail_load: it re-applies the documented cleaning rules
(docs/RETAIL_DATA_NOTES.md) to the raw rows and answers each gold question with pandas.
"""

import pandas as pd


def clean_raw(raw):
    df = raw.drop_duplicates()
    invoice = df["InvoiceNo"].astype(str).str.strip()
    df = df[~invoice.str.startswith("A") & (df["UnitPrice"] > 0)].copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].str.strip()
    df["cancelled"] = df["InvoiceNo"].str.startswith("C")
    df["line_total"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["invoice_time"] = df.groupby("InvoiceNo")["InvoiceDate"].transform("min")
    return df


def _most_frequent(values):
    counts = values.dropna().value_counts()
    if counts.empty:
        return None
    best = counts.max()
    return sorted(counts[counts == best].index)[0]


def customer_country(df):
    return df.dropna(subset=["CustomerID"]).groupby("CustomerID")["Country"].agg(_most_frequent)


def answers(df):
    out = {}
    out["re01"] = pd.DataFrame({"n": [df["CustomerID"].dropna().nunique()]})
    out["re02"] = pd.DataFrame({"n": [df.loc[df["cancelled"], "InvoiceNo"].nunique()]})

    countries = customer_country(df).value_counts()
    countries = sorted(countries.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    out["re03"] = pd.DataFrame(countries, columns=["country", "n"])

    out["re04"] = pd.DataFrame({"revenue": [round(df["line_total"].sum(), 2)]})

    by_country = df.groupby("Country")["line_total"].sum().round(2)
    by_country = sorted(by_country.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    out["rm01"] = pd.DataFrame(by_country, columns=["country", "revenue"])

    descriptions = df.groupby("StockCode")["Description"].agg(_most_frequent)
    units = df.groupby("StockCode")["Quantity"].sum()
    top = sorted(units.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    out["rm02"] = pd.DataFrame([(code, descriptions[code], n) for code, n in top], columns=["code", "desc", "units"])

    in_2011 = df[df["invoice_time"].dt.year == 2011]
    monthly = in_2011.groupby(in_2011["invoice_time"].dt.month)["line_total"].sum().round(2)
    out["rm03"] = monthly.reset_index().set_axis(["month", "revenue"], axis=1)

    known = df.dropna(subset=["CustomerID"])
    spend = known.groupby("CustomerID")["line_total"].sum().round(2)
    country = customer_country(df)
    top = sorted(spend.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    out["rm04"] = pd.DataFrame([(int(c), country[c], v) for c, v in top], columns=["customer", "country", "revenue"])

    bought = known[~known["cancelled"]]
    first = bought.groupby("CustomerID")["invoice_time"].min()
    first = first[first.dt.year == 2011]
    out["rh01"] = first.dt.month.value_counts().sort_index().reset_index().set_axis(["month", "n"], axis=1)

    per_customer = bought.groupby("CustomerID")["InvoiceNo"].nunique()
    out["rh02"] = pd.DataFrame({"pct": [round(100.0 * (per_customer > 1).sum() / len(per_customer), 2)]})
    return out
