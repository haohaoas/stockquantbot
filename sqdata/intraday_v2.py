from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_hhmm(text: str, default_minutes: int) -> int:
    s = str(text or "").strip()
    if not s:
        return default_minutes
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return default_minutes


def _latest_minute_file(sym: str, minute_dir: str, interval: str) -> Path | None:
    p = Path(minute_dir) / str(sym).zfill(6)
    if not p.exists():
        return None
    files = sorted(p.glob(f"*_{interval}.csv"))
    if not files:
        return None
    return files[-1]


def _load_minute(sym: str, minute_dir: str, interval: str, only_today: bool) -> pd.DataFrame:
    f = _latest_minute_file(sym, minute_dir, interval)
    if f is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(f)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "datetime" not in df.columns:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    if df.empty:
        return df
    if only_today:
        today = dt.date.today().isoformat()
        df = df[df["datetime"].dt.date.astype(str) == today].copy()
    for c in ("open", "close", "high", "low", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close", "high", "low"]).reset_index(drop=True)
    return df


def _atr14(df: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    tr = pd.concat(
        [
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _daily_llv5(symbol: str, daily_dir: str = "./data/manual_hist") -> float:
    p = Path(daily_dir) / f"{str(symbol).zfill(6)}.csv"
    if not p.exists():
        return float("nan")
    try:
        d = pd.read_csv(p)
    except Exception:
        return float("nan")
    if d.empty or "low" not in d.columns:
        return float("nan")
    low = pd.to_numeric(d["low"], errors="coerce")
    if low.dropna().empty:
        return float("nan")
    return float(low.tail(5).min())


def _compute_intraday_signal(
    symbol: str,
    minute_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, Any]:
    if minute_df is None or minute_df.empty or len(minute_df) < 30:
        return {"entry_tag": "NO_DATA", "entry_hint": "分钟数据缺失", "intraday_rule_ok": 0}

    df = minute_df.copy().reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df.get("volume"), errors="coerce")

    ma3 = close.rolling(3, min_periods=3).mean()
    ma6 = close.rolling(6, min_periods=6).mean()
    ma12 = close.rolling(12, min_periods=12).mean()
    atr14 = _atr14(df)
    atr_pct = atr14 / close.replace(0, np.nan)
    hhv12_prev = high.rolling(12, min_periods=12).max().shift(1)
    roll_max20 = close.rolling(20, min_periods=20).max()
    dd_now = (roll_max20 - close) / roll_max20.replace(0, np.nan)
    mdd20 = dd_now.rolling(20, min_periods=20).max()

    vol_ma5_prev = volume.rolling(5, min_periods=5).mean().shift(1)
    vol_ratio = volume / vol_ma5_prev

    vr_q40 = vol_ratio.rolling(20, min_periods=20).quantile(0.40).shift(1)
    vr_q80 = vol_ratio.rolling(20, min_periods=20).quantile(0.80).shift(1)
    vr_q30 = vol_ratio.rolling(20, min_periods=20).quantile(0.30).shift(1)

    breakout_mult = float(cfg.get("breakout_mult", 1.0015))
    pullback_ma_dev = float(cfg.get("pullback_ma_dev", 0.005))
    extended_z_max = float(cfg.get("extended_z_max", 1.8))
    breakout_vr_min = float(cfg.get("breakout_vol_ratio_min", 1.0))
    breakout_vr_max = float(cfg.get("breakout_vol_ratio_max", 1.8))
    pullback_vr_max = float(cfg.get("pullback_vol_ratio_max", 1.1))
    adaptive_volume = bool(cfg.get("adaptive_volume", False))
    atr_pct_max = float(cfg.get("atr_pct_max", 0.08))
    mdd20_max = float(cfg.get("mdd20_max", 0.12))
    if mdd20_max > 1:
        mdd20_max = mdd20_max / 100.0
    start_min = _parse_hhmm(str(cfg.get("entry_start", "09:00")), 9 * 60)
    end_min = _parse_hhmm(str(cfg.get("entry_end", "14:00")), 14 * 60)

    ext_z = (close - ma12) / atr14.replace(0, np.nan)

    best: dict[str, Any] = {
        "entry_tag": "WAIT",
        "entry_hint": "等待分钟线确认",
        "intraday_rule_ok": 0,
    }

    def _in_window(ts: pd.Timestamp) -> bool:
        t = int(ts.hour) * 60 + int(ts.minute)
        return start_min <= t <= end_min

    for trig in range(20, len(df) - 1):
        conf = trig + 1
        ts = df["datetime"].iloc[conf]
        if not _in_window(ts):
            continue
        level = hhv12_prev.iloc[trig]
        if not np.isfinite(level) or level <= 0:
            continue

        lower = breakout_vr_min
        upper = breakout_vr_max
        if adaptive_volume and np.isfinite(vr_q40.iloc[trig]) and np.isfinite(vr_q80.iloc[trig]):
            lower = float(vr_q40.iloc[trig])
            upper = float(vr_q80.iloc[trig])

        cond_breakout = (
            close.iloc[trig] > level * breakout_mult
            and ma3.iloc[trig] > ma6.iloc[trig] > ma12.iloc[trig]
            and np.isfinite(vol_ratio.iloc[trig])
            and vol_ratio.iloc[trig] >= lower
            and vol_ratio.iloc[trig] <= upper
            and np.isfinite(ext_z.iloc[trig])
            and ext_z.iloc[trig] <= extended_z_max
            and np.isfinite(atr_pct.iloc[trig])
            and atr_pct.iloc[trig] <= atr_pct_max
            and np.isfinite(mdd20.iloc[trig])
            and mdd20.iloc[trig] <= mdd20_max
            and close.iloc[conf] >= level * breakout_mult
            and low.iloc[conf] >= level
        )
        if cond_breakout:
            entry = float(close.iloc[conf])
            atr_v = float(atr14.iloc[conf]) if np.isfinite(atr14.iloc[conf]) else float("nan")
            swing_low = float(low.iloc[max(0, conf - 9) : conf + 1].min())
            daily_stop = _daily_llv5(symbol)
            minute_stop = swing_low - 0.3 * atr_v if np.isfinite(atr_v) else swing_low
            init_stop = max(daily_stop, minute_stop) if np.isfinite(daily_stop) else minute_stop
            r = entry - init_stop if np.isfinite(init_stop) else float("nan")
            target = entry + 1.2 * r if np.isfinite(r) and r > 0 else float("nan")
            best = {
                "entry_tag": "BREAKOUT",
                "entry_hint": "分钟突破确认",
                "intraday_rule_ok": 1,
                "entry_level": float(level * breakout_mult),
                "entry_m5": entry,
                "stop_m5": init_stop,
                "target_m5": target,
                "entry_time_m5": ts.isoformat(),
            }

    for i in range(20, len(df)):
        ts = df["datetime"].iloc[i]
        if not _in_window(ts):
            continue
        if i < 2:
            continue
        low_ok = low.iloc[i - 1] >= low.iloc[i - 2] and low.iloc[i] >= low.iloc[i - 1]
        close_up = close.iloc[i] > close.iloc[i - 1]
        ma12_i = ma12.iloc[i]
        if not np.isfinite(ma12_i) or ma12_i <= 0:
            continue
        near_ma12 = abs(close.iloc[i] / ma12_i - 1.0) <= pullback_ma_dev
        above_ma12 = close.iloc[i] >= ma12_i
        if adaptive_volume and np.isfinite(vr_q30.iloc[i]):
            pullback_vr = float(vr_q30.iloc[i])
        else:
            pullback_vr = pullback_vr_max
        vr_ok = np.isfinite(vol_ratio.iloc[i]) and vol_ratio.iloc[i] <= pullback_vr
        trend_ok = np.isfinite(ma3.iloc[i]) and np.isfinite(ma6.iloc[i]) and (ma3.iloc[i] > ma6.iloc[i])
        vol_ok = (
            np.isfinite(atr_pct.iloc[i])
            and atr_pct.iloc[i] <= atr_pct_max
            and np.isfinite(mdd20.iloc[i])
            and mdd20.iloc[i] <= mdd20_max
        )
        cond_pullback = near_ma12 and low_ok and close_up and above_ma12 and vr_ok and trend_ok and vol_ok
        if cond_pullback:
            entry = float(close.iloc[i])
            atr_v = float(atr14.iloc[i]) if np.isfinite(atr14.iloc[i]) else float("nan")
            swing_low = float(low.iloc[max(0, i - 9) : i + 1].min())
            daily_stop = _daily_llv5(symbol)
            minute_stop = swing_low - 0.3 * atr_v if np.isfinite(atr_v) else swing_low
            init_stop = max(daily_stop, minute_stop) if np.isfinite(daily_stop) else minute_stop
            r = entry - init_stop if np.isfinite(init_stop) else float("nan")
            target = entry + 1.2 * r if np.isfinite(r) and r > 0 else float("nan")
            best = {
                "entry_tag": "PULLBACK",
                "entry_hint": "分钟回踩确认",
                "intraday_rule_ok": 1,
                "entry_level": float(ma12_i),
                "entry_m5": entry,
                "stop_m5": init_stop,
                "target_m5": target,
                "entry_time_m5": ts.isoformat(),
            }

    if best.get("entry_tag") == "WAIT":
        z_last = float(ext_z.iloc[-1]) if np.isfinite(ext_z.iloc[-1]) else float("nan")
        if np.isfinite(z_last) and z_last > extended_z_max:
            best["entry_tag"] = "EXTENDED"
            best["entry_hint"] = "分钟偏离过大，禁止追高"
    return best


def apply_intraday_v2(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return out

    minute_dir = str(cfg.get("minute_dir", "./data/manual_minute"))
    interval = str(cfg.get("interval", "m5"))
    only_today = bool(cfg.get("only_today", False))
    apply_to_watch = bool(cfg.get("apply_to_watch", True))
    no_data_action = str(cfg.get("no_data_action", "WATCH")).upper()

    tags: list[str] = []
    hints: list[str] = []
    levels: list[float] = []
    entrys: list[float] = []
    stops: list[float] = []
    targets: list[float] = []
    times: list[str] = []
    oks: list[int] = []
    actions: list[str] = []
    reasons: list[str] = []

    for _, r in out.iterrows():
        sym = str(r.get("symbol", "")).zfill(6)
        action = str(r.get("action", ""))
        reason = str(r.get("reason", "") or "")
        amount = float(pd.to_numeric(r.get("amount", float("nan")), errors="coerce"))
        if action not in ("BUY", "WATCH") or (action == "WATCH" and not apply_to_watch):
            tags.append("")
            hints.append("")
            levels.append(float("nan"))
            entrys.append(float("nan"))
            stops.append(float("nan"))
            targets.append(float("nan"))
            times.append("")
            oks.append(0)
            actions.append(action)
            reasons.append(reason)
            continue

        m = _load_minute(sym, minute_dir, interval, only_today=only_today)
        info = _compute_intraday_signal(sym, m, cfg)
        tag = str(info.get("entry_tag", ""))
        ok = int(info.get("intraday_rule_ok", 0))
        hint = str(info.get("entry_hint", ""))
        lvl = info.get("entry_level", float("nan"))
        ent = info.get("entry_m5", float("nan"))
        stp = info.get("stop_m5", float("nan"))
        tgt = info.get("target_m5", float("nan"))
        tms = str(info.get("entry_time_m5", "") or "")

        new_action = action
        if action == "BUY":
            if ok != 1:
                if tag == "NO_DATA" and amount == amount and amount <= 0:
                    new_action = "AVOID"
                    hint = "疑似停牌/无成交，跳过"
                else:
                    new_action = no_data_action if tag == "NO_DATA" else "WATCH"
            if ok == 1 and tag in ("BREAKOUT", "PULLBACK"):
                reason = f"{reason}; 分钟{tag}确认" if reason else f"分钟{tag}确认"
            elif hint:
                reason = f"{reason}; {hint}" if reason else hint

        tags.append(tag)
        hints.append(hint)
        levels.append(lvl if isinstance(lvl, (float, int)) else float("nan"))
        entrys.append(ent if isinstance(ent, (float, int)) else float("nan"))
        stops.append(stp if isinstance(stp, (float, int)) else float("nan"))
        targets.append(tgt if isinstance(tgt, (float, int)) else float("nan"))
        times.append(tms)
        oks.append(ok)
        actions.append(new_action)
        reasons.append(reason)

    out["entry_tag"] = tags
    out["entry_hint"] = hints
    out["entry_level"] = levels
    out["entry_m5"] = entrys
    out["stop_m5"] = stops
    out["target_m5"] = targets
    out["entry_time_m5"] = times
    out["intraday_rule_ok"] = oks
    out["action"] = actions
    out["reason"] = reasons
    return out
