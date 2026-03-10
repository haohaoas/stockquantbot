from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _latest_minute_file(sym_dir: Path, interval: str) -> Path | None:
    files = sorted(sym_dir.glob(f"*_{interval}.csv"))
    if not files:
        return None
    return files[-1]


def _load_minute(symbol: str, minute_dir: str, interval: str) -> tuple[pd.DataFrame, str]:
    sym = str(symbol).zfill(6)
    sym_dir = Path(minute_dir) / sym
    if not sym_dir.exists():
        return pd.DataFrame(), ""
    latest = _latest_minute_file(sym_dir, interval)
    if not latest:
        return pd.DataFrame(), ""
    try:
        df = pd.read_csv(latest)
    except Exception:
        return pd.DataFrame(), latest.stem
    if df.empty:
        return df, latest.stem
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df, latest.stem


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or v != v:
            return None
        return float(v)
    except Exception:
        return None


def compute_intraday_hint(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty or "close" not in df.columns:
        return {"entry_hint": "分钟数据缺失", "entry_level": np.nan, "entry_tag": "NO_DATA"}

    df = df.copy()
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if df.empty:
        return {"entry_hint": "分钟数据缺失", "entry_level": np.nan, "entry_tag": "NO_DATA"}

    last = df.iloc[-1]
    close = float(last["close"])
    high = float(last.get("high", close))
    vol = _safe_float(last.get("volume"))

    ma_short = df["close"].rolling(3, min_periods=3).mean().iloc[-1]
    ma_long = df["close"].rolling(6, min_periods=6).mean().iloc[-1]

    recent_high = df["high"].rolling(12, min_periods=6).max().shift(1).iloc[-1]
    vol_mean = df["volume"].rolling(12, min_periods=6).mean().iloc[-1] if "volume" in df.columns else np.nan
    vol_spike = vol is not None and vol_mean == vol_mean and vol_mean > 0 and vol >= 1.5 * vol_mean

    pullback_ok = False
    if ma_short == ma_short and ma_long == ma_long:
        pullback_ok = abs(close / ma_short - 1) <= 0.003 and ma_short >= ma_long

    breakout_ok = False
    if recent_high == recent_high:
        breakout_ok = close >= recent_high

    if breakout_ok and vol_spike:
        return {
            "entry_hint": "盘中放量突破近1小时新高，回踩不破可观察",
            "entry_level": recent_high,
            "entry_tag": "BREAKOUT",
        }
    if pullback_ok:
        return {
            "entry_hint": "回踩短均线企稳，量能稳定可观察",
            "entry_level": ma_short,
            "entry_tag": "PULLBACK",
        }
    if ma_short == ma_short and close >= ma_short * 1.015:
        return {
            "entry_hint": "价格偏离短均线较多，追高风险",
            "entry_level": ma_short,
            "entry_tag": "EXTENDED",
        }
    return {
        "entry_hint": "等待站稳或放量确认",
        "entry_level": ma_short if ma_short == ma_short else np.nan,
        "entry_tag": "WAIT",
    }


def attach_intraday_hints(
    df: pd.DataFrame,
    minute_dir: str,
    interval: str = "m5",
    only_today: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    hints = []
    levels = []
    tags = []
    dates = []
    today = dt.date.today().isoformat()

    for _, row in out.iterrows():
        sym = str(row.get("symbol", "")).zfill(6)
        data, stem = _load_minute(sym, minute_dir, interval)
        date_tag = ""
        if stem:
            date_tag = stem.split("_", 1)[0]
        if only_today and date_tag and date_tag != today:
            hints.append("分钟数据非今日")
            levels.append(np.nan)
            tags.append("STALE")
            dates.append(date_tag)
            continue
        info = compute_intraday_hint(data)
        hints.append(info.get("entry_hint", ""))
        levels.append(info.get("entry_level", np.nan))
        tags.append(info.get("entry_tag", ""))
        dates.append(date_tag)

    out["entry_hint"] = hints
    out["entry_level"] = levels
    out["entry_tag"] = tags
    out["entry_date"] = dates
    return out
