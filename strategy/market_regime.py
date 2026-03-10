from __future__ import annotations

import pandas as pd

from features.indicators import atr


def market_regime_ok(index_df: pd.DataFrame, ma_window: int = 200) -> bool:
    if index_df is None or index_df.empty:
        return False
    if "close" not in index_df.columns:
        return False
    close = index_df["close"].astype(float)
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    if ma.empty or close.empty:
        return False
    last_close = float(close.iloc[-1])
    last_ma = float(ma.iloc[-1])
    if not (last_close == last_close and last_ma == last_ma):
        return False
    return last_close >= last_ma


def compute_regime_flags(
    index_df: pd.DataFrame,
    *,
    ma_fast: int = 20,
    ma_slow: int = 60,
    require_streak: int = 1,
    atr_window: int = 14,
    atr_pct_max: float = 0.025,
    use_volatility_filter: bool = True,
    vol_ratio_thresh: float | None = None,
    vol_window: int = 20,
    vol_avg_window: int = 60,
) -> pd.Series:
    """Return boolean regime flags indexed by date string."""
    if index_df is None or index_df.empty or "close" not in index_df.columns:
        return pd.Series(dtype=bool)
    df = index_df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.slice(0, 10)
        df = df.sort_values("date")
        df = df.set_index("date")
    close = pd.to_numeric(df["close"], errors="coerce")
    ma20 = close.rolling(ma_fast, min_periods=ma_fast).mean()
    ma60 = close.rolling(ma_slow, min_periods=ma_slow).mean()

    above = (close > ma20) & (close > ma60)
    if require_streak and require_streak > 1:
        above = above.rolling(require_streak, min_periods=require_streak).min() == 1

    vol_ok = pd.Series(True, index=close.index)
    if use_volatility_filter:
        if vol_ratio_thresh is not None:
            vol = close.pct_change().rolling(vol_window, min_periods=vol_window).std()
            vol_avg = vol.rolling(vol_avg_window, min_periods=vol_avg_window).mean()
            vol_ratio = vol / vol_avg
            vol_ok = vol_ratio <= float(vol_ratio_thresh)
        else:
            atr_pct = None
            if all(c in df.columns for c in ("high", "low", "close")):
                atr_val = atr(df[["high", "low", "close"]].astype(float), atr_window)
                atr_pct = atr_val / close
            if atr_pct is None:
                vol = close.pct_change().rolling(atr_window, min_periods=atr_window).std()
                atr_pct = vol
            vol_ok = atr_pct <= float(atr_pct_max)

    flags = above & vol_ok
    return flags.fillna(False)


def market_regime_ok_strict(
    index_df: pd.DataFrame,
    *,
    date: str | None = None,
    ma_fast: int = 20,
    ma_slow: int = 60,
    require_streak: int = 1,
    atr_window: int = 14,
    atr_pct_max: float = 0.025,
    use_volatility_filter: bool = True,
    vol_ratio_thresh: float | None = None,
    vol_window: int = 20,
    vol_avg_window: int = 60,
) -> bool:
    flags = compute_regime_flags(
        index_df,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        require_streak=require_streak,
        atr_window=atr_window,
        atr_pct_max=atr_pct_max,
        use_volatility_filter=use_volatility_filter,
        vol_ratio_thresh=vol_ratio_thresh,
        vol_window=vol_window,
        vol_avg_window=vol_avg_window,
    )
    if flags.empty:
        return False
    if date:
        return bool(flags.get(str(date)[:10], False))
    return bool(flags.iloc[-1])


def detect_market_env(
    index_df: pd.DataFrame,
    *,
    ma_fast: int = 20,
    ma_slow: int = 60,
    vol_window: int = 20,
    up_thresh: float = 0.05,
    down_thresh: float = -0.05,
    vol_thresh: float = 0.02,
) -> str:
    if index_df is None or index_df.empty or "close" not in index_df.columns:
        return "震荡市场"
    close = index_df["close"].astype(float)
    if close.empty:
        return "震荡市场"
    ma20 = close.rolling(ma_fast, min_periods=ma_fast).mean()
    ma60 = close.rolling(ma_slow, min_periods=ma_slow).mean()
    last_close = float(close.iloc[-1])
    last_ma20 = float(ma20.iloc[-1]) if not ma20.empty else float("nan")
    last_ma60 = float(ma60.iloc[-1]) if not ma60.empty else float("nan")
    if not (last_close == last_close and last_ma20 == last_ma20 and last_ma60 == last_ma60):
        return "震荡市场"

    price_pos = last_close / last_ma20 - 1 if last_ma20 > 0 else 0.0
    if price_pos > up_thresh and last_ma20 > last_ma60:
        return "上涨趋势"
    if price_pos < down_thresh and last_ma20 < last_ma60:
        return "下跌趋势"

    vol = close.pct_change().rolling(vol_window, min_periods=vol_window).std()
    last_vol = float(vol.iloc[-1]) if not vol.empty else float("nan")
    if last_vol == last_vol and last_vol > vol_thresh:
        return "震荡市场"
    return "横盘整理"


def compute_volatility_ratio(
    index_df: pd.DataFrame,
    *,
    window: int = 20,
    history: int = 20,
) -> tuple[float, float, float] | None:
    if index_df is None or index_df.empty or "close" not in index_df.columns:
        return None
    close = index_df["close"].astype(float)
    if close.empty:
        return None
    ret = close.pct_change()
    vol = ret.rolling(window, min_periods=window).std()
    if vol.empty:
        return None
    current_vol = float(vol.iloc[-1])
    if history <= 0:
        return None
    hist_vol = vol.tail(history)
    avg_vol = float(hist_vol.mean()) if not hist_vol.empty else float("nan")
    if not (current_vol == current_vol and avg_vol == avg_vol) or avg_vol <= 0:
        return None
    ratio = float(current_vol / avg_vol)
    return current_vol, avg_vol, ratio
