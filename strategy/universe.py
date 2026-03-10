from __future__ import annotations

import pandas as pd


def filter_universe(df: pd.DataFrame, universe_cfg: dict) -> pd.DataFrame:
    """
    Filters on spot-like dataframe with columns: symbol, name, close, amount
    """
    out = df.copy()

    min_price = float(universe_cfg.get("min_price", 5.0))
    max_price = float(universe_cfg.get("max_price", 0.0))
    min_avg_amount_20 = float(universe_cfg.get("min_avg_amount_20", 1e8))
    exclude_st = bool(universe_cfg.get("exclude_st", True))

    # basic numeric filters (spot)
    out = out[out["close"] >= min_price]
    if max_price > 0:
        out = out[out["close"] <= max_price]
    out = out[out["amount"] >= min_avg_amount_20]

    # name-based filters
    if exclude_st and "name" in out.columns:
        out = out[~out["name"].astype(str).str.contains("ST|\\*ST|退", regex=True)]

    return out.reset_index(drop=True)
