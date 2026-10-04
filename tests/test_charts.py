import pandas as pd

from src.charts import chart_spec


def test_line_chart_for_time_series():
    frame = pd.DataFrame({"date_id": [202401, 202402, 202403], "revenue": [1.0, 2.0, 3.0]})
    assert chart_spec(frame) == {"kind": "line", "x": "date_id", "y": ["revenue"]}


def test_line_chart_for_iso_month_strings():
    frame = pd.DataFrame({"m": ["2024-01-01", "2024-02-01"], "v": [1, 2]})
    assert chart_spec(frame)["kind"] == "line"


def test_bar_chart_for_category_and_number():
    frame = pd.DataFrame({"region": ["North", "South"], "revenue": [10.0, 20.0]})
    assert chart_spec(frame) == {"kind": "bar", "x": "region", "y": ["revenue"]}


def test_no_chart_for_scalar_or_text_only():
    assert chart_spec(pd.DataFrame({"n": [5]})) is None
    assert chart_spec(pd.DataFrame({"a": ["x"], "b": ["y"]})) is None
    assert chart_spec(pd.DataFrame()) is None
    assert chart_spec(None) is None
