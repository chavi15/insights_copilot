import pandas as pd


def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def chart_spec(frame):
    if frame is None or frame.empty or len(frame.columns) < 2:
        return None
    numeric = [c for c in frame.columns if _is_numeric(frame[c])]
    if not numeric:
        return None
    temporal = [
        c
        for c in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[c])
        or c in ("date_id", "month_start")
        or (pd.api.types.is_string_dtype(frame[c]) and frame[c].astype(str).str.fullmatch(r"\d{4}-\d{2}(-\d{2})?").all())
    ]
    if temporal and len(frame) > 1:
        x = temporal[0]
        y = [c for c in numeric if c != x][:3]
        if y:
            return {"kind": "line", "x": x, "y": y}
    categorical = [c for c in frame.columns if c not in numeric]
    if categorical and len(frame) <= 60:
        y = [c for c in numeric][:3]
        return {"kind": "bar", "x": categorical[0], "y": y}
    return None
