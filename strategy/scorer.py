from __future__ import annotations

import pandas as pd


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.empty:
        return s
    min_v = s.min()
    max_v = s.max()
    if min_v != min_v or max_v != max_v or max_v == min_v:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - min_v) / (max_v - min_v)


def _compute_factor_score(out: pd.DataFrame, weights: dict) -> pd.Series:
    cfg = weights.get("factor_weights", {}) if isinstance(weights, dict) else {}
    w_accel_5 = float(cfg.get("price_accel_5", 0.2))
    w_accel_10 = float(cfg.get("price_accel_10", 0.2))
    w_vs_high = float(cfg.get("price_vs_high_20", 0.2))
    w_vs_ma5 = float(cfg.get("price_vs_ma_5", 0.2))
    w_bull = float(cfg.get("bull_volume", 0.1))
    w_shrink = float(cfg.get("volume_shrink_20d", 0.1))
    w_bear = float(cfg.get("bear_volume_penalty", 0.1))

    def _col_or_zeros(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(0.0, index=out.index)

    accel_5 = _minmax(_col_or_zeros("price_accel_5"))
    accel_10 = _minmax(_col_or_zeros("price_accel_10"))
    vs_high = _minmax(_col_or_zeros("price_vs_high_20"))
    vs_ma5 = _minmax(_col_or_zeros("price_vs_ma_5"))

    bull = _col_or_zeros("bull_volume").fillna(0.0)
    shrink = _col_or_zeros("volume_shrink_20d").fillna(0.0)
    bear = _col_or_zeros("bear_volume").fillna(0.0)

    score = (
        accel_5 * w_accel_5
        + accel_10 * w_accel_10
        + vs_high * w_vs_high
        + vs_ma5 * w_vs_ma5
        + bull * w_bull
        + shrink * w_shrink
        - bear * w_bear
    )
    return score.clip(0, 1)


def score_and_rank(df: pd.DataFrame, weights: dict, top_n: int = 20, sector_cfg: dict | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    score_mode = str(weights.get("mode", "rule") if isinstance(weights, dict) else "rule").strip().lower()

    w_breakout = float(weights.get("breakout", 40))
    w_rs = float(weights.get("rs", 30))
    w_trend = float(weights.get("trend", 20))
    w_volume = float(weights.get("volume", 10))
    w_rsi = float(weights.get("rsi", 5))
    w_macd = float(weights.get("macd", 5))
    risk_penalty_max = float(weights.get("risk_penalty_max", 20))

    out = df.copy()

    # Normalize components
    out["rs_score"] = out["rs_proxy"].clip(0, 1)  # already 0..1
    out["trend_score"] = out["trend"].clip(0, 1)
    out["volume_score"] = (out["vol_ratio"] / 2.5).clip(0, 1)  # cap at 2.5x
    if "rsi" in out.columns:
        out["rsi_score"] = (1 - (out["rsi"].fillna(50) - 60).abs() / 40).clip(0, 1)
    else:
        out["rsi_score"] = 0.0
    if "macd_hist" in out.columns:
        out["macd_score"] = (out["macd_hist"].fillna(0) > 0).astype(float)
    else:
        out["macd_score"] = 0.0
    # Risk penalty based on atr_pct + mdd20
    # Both typically ~0..0.15; scale to 0..1 then to penalty
    risk_raw = (out["atr_pct"].fillna(0) / 0.08).clip(0, 1) * 0.6 + (out["mdd20"].fillna(0) / 0.12).clip(0, 1) * 0.4
    out["risk_penalty"] = (risk_raw * risk_penalty_max).clip(0, risk_penalty_max)

    breakout_col = "breakout"
    if "breakout_rt" in out.columns and out["breakout_rt"].notna().any():
        breakout_col = "breakout_rt"

    out["score"] = (
        w_breakout * out[breakout_col].astype(float)
        + w_rs * out["rs_score"]
        + w_trend * out["trend_score"]
        + w_volume * out["volume_score"]
        + w_rsi * out["rsi_score"]
        + w_macd * out["macd_score"]
        - out["risk_penalty"]
    )
    out["rule_score"] = out["score"]

    # Composite factor score (0..1), optional mode switch
    out["factor_score"] = _compute_factor_score(out, weights)
    if score_mode == "factor":
        out["score"] = (out["factor_score"] * 100).clip(0, 100)
    elif score_mode in ("hybrid", "mix", "blend"):
        hybrid_rule = float(weights.get("hybrid_rule", 0.7))
        hybrid_factor = float(weights.get("hybrid_factor", 0.3))
        out["score"] = (
            out["rule_score"] * hybrid_rule + out["factor_score"] * 100.0 * hybrid_factor
        )

    # Sector resonance boost (optional)
    cfg = sector_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    if enabled and "sector" in out.columns:
        min_score = float(cfg.get("min_score", 0.0))
        require_breakout = bool(cfg.get("require_breakout", True))
        require_vol_ok = bool(cfg.get("require_vol_ok", True))
        min_count = int(cfg.get("min_count", 3))
        boost_per_stock = float(cfg.get("boost_per_stock", 1.5))
        max_boost = float(cfg.get("max_boost", 6.0))

        mask = out["score"] >= min_score
        if require_breakout and "breakout" in out.columns:
            mask = mask & (out["breakout"].astype(float) >= 1.0)
        if require_vol_ok and "vol_ok" in out.columns:
            mask = mask & (out["vol_ok"].astype(float) >= 1.0)

        counts = out.loc[mask].groupby("sector").size().to_dict()
        out["sector_strength"] = out["sector"].map(counts).fillna(0).astype(int)
        out["sector_boost"] = (out["sector_strength"] * boost_per_stock).clip(0, max_boost)
        out.loc[out["sector_strength"] < min_count, "sector_boost"] = 0.0
        out["score"] = out["score"] + out["sector_boost"]
    else:
        out["sector_strength"] = 0
        out["sector_boost"] = 0.0

    out = out.sort_values(["score", breakout_col, "vol_ratio", "amount"], ascending=[False, False, False, False]).reset_index(drop=True)

    # Add simple “trigger” and “stop” suggestions
    out["buy_trigger"] = out.apply(
        lambda r: f"站上{r['hhv']:.2f}且放量(≥{r['vol_ratio']:.2f}x)" if r.get(breakout_col, 0) else f"突破{r['hhv']:.2f}并放量确认",
        axis=1,
    )
    out["stop_ref"] = out.apply(lambda r: f"跌破MA20({r['ma20']:.2f})" if r.get("ma20", 0) else "跌破关键支撑", axis=1)

    return out.head(int(top_n))
