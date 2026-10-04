import pandas as pd

from src.data_gen import build_frames


def test_row_counts_and_keys(frames):
    assert len(frames["dim_product"]) == 8
    assert len(frames["dim_territory"]) == 40
    assert len(frames["dim_hcp"]) == 2000
    assert len(frames["dim_date"]) == 24
    assert len(frames["fact_sales"]) > 50_000
    assert len(frames["fact_calls"]) > 10_000
    assert frames["fact_sales"]["sale_id"].is_unique
    assert frames["fact_calls"]["call_id"].is_unique


def test_foreign_keys_resolve(frames):
    sales, calls, targets, hcp = frames["fact_sales"], frames["fact_calls"], frames["targets"], frames["dim_hcp"]
    assert sales["hcp_id"].isin(hcp["hcp_id"]).all()
    assert sales["product_id"].isin(frames["dim_product"]["product_id"]).all()
    assert sales["date_id"].isin(frames["dim_date"]["date_id"]).all()
    assert calls["hcp_id"].isin(hcp["hcp_id"]).all()
    assert hcp["territory_id"].isin(frames["dim_territory"]["territory_id"]).all()
    assert targets["territory_id"].isin(frames["dim_territory"]["territory_id"]).all()


def test_is_deterministic():
    a, b = build_frames(seed=7, n_hcp=200), build_frames(seed=7, n_hcp=200)
    pd.testing.assert_frame_equal(a["fact_sales"], b["fact_sales"])
    assert not a["fact_sales"].equals(build_frames(seed=8, n_hcp=200)["fact_sales"])


def test_no_sales_before_launch(frames):
    sales = frames["fact_sales"].merge(frames["dim_date"], on="date_id").merge(frames["dim_product"], on="product_id")
    sales["month_start"] = pd.to_datetime(sales["month_start"])
    sales["launch_date"] = pd.to_datetime(sales["launch_date"])
    assert (sales["month_start"] >= sales["launch_date"].dt.to_period("M").dt.to_timestamp()).all()


def test_launch_ramp_for_new_product(frames):
    sales = frames["fact_sales"][frames["fact_sales"]["product_id"] == 7]
    by_month = sales.groupby("date_id")["revenue"].sum()
    assert by_month.index.min() == 202410
    assert by_month.loc[202603 - 100 + 100 - 100 if False else 202412] > by_month.loc[202410]


def test_tier_a_sells_more_per_hcp(frames):
    per_hcp = frames["fact_sales"].groupby("hcp_id")["revenue"].sum().rename("revenue")
    joined = frames["dim_hcp"].merge(per_hcp, on="hcp_id", how="left").fillna({"revenue": 0})
    by_tier = joined.groupby("tier")["revenue"].mean()
    assert by_tier["A"] > by_tier["B"] > by_tier["C"]


def test_some_territories_miss_target_in_some_quarters(frames):
    actual = (
        frames["fact_sales"]
        .merge(frames["dim_hcp"][["hcp_id", "territory_id"]], on="hcp_id")
        .groupby(["territory_id", "product_id", "date_id"], as_index=False)["revenue"]
        .sum()
    )
    merged = frames["targets"].merge(actual, on=["territory_id", "product_id", "date_id"])
    assert len(merged) == len(frames["targets"])
    missed = (merged["revenue"] < merged["target_revenue"]).mean()
    assert 0.2 < missed < 0.8
