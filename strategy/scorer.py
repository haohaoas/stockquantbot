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


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _col(out: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in out.columns:
        return pd.to_numeric(out[name], errors="coerce")
    return pd.Series(default, index=out.index, dtype=float)


def _compute_factor_score(out: pd.DataFrame, weights: dict) -> pd.Series:
    cfg = weights.get("factor_weights", {}) if isinstance(weights, dict) else {}
    w_accel_5 = float(cfg.get("price_accel_5", 0.2))
    w_accel_10 = float(cfg.get("price_accel_10", 0.2))
    w_vs_high = float(cfg.get("price_vs_high_20", 0.2))
    w_vs_ma5 = float(cfg.get("price_vs_ma_5", 0.2))
    w_bull = float(cfg.get("bull_volume", 0.1))
    w_shrink = float(cfg.get("volume_shrink_20d", 0.1))
    w_bear = float(cfg.get("bear_volume_penalty", 0.1))

    accel_5 = _minmax(_col(out, "price_accel_5"))
    accel_10 = _minmax(_col(out, "price_accel_10"))
    vs_high = _minmax(_col(out, "price_vs_high_20"))
    vs_ma5 = _minmax(_col(out, "price_vs_ma_5"))

    bull = _col(out, "bull_volume").fillna(0.0)
    shrink = _col(out, "volume_shrink_20d").fillna(0.0)
    bear = _col(out, "bear_volume").fillna(0.0)

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


def _compute_path_scores(out: pd.DataFrame, weights: dict, decision_cfg: dict | None = None) -> pd.DataFrame:
    cfg = (decision_cfg or {}).get("signal_paths") or {}
    breakout_cfg = cfg.get("breakout") or {}
    pullback_cfg = cfg.get("pullback") or {}
    rebound_cfg = cfg.get("rebound") or {}

    breakout_enabled = bool(breakout_cfg.get("enabled", True))
    breakout_vol_threshold = float(breakout_cfg.get("vol_threshold", 1.2))
    breakout_pct_min = float(breakout_cfg.get("min_pct", 1.0))

    pullback_enabled = bool(pullback_cfg.get("enabled", True))
    pullback_max = float(pullback_cfg.get("max_ma20_dist", 0.05))
    pullback_shrink_threshold = float(pullback_cfg.get("shrink_threshold", 0.8))
    pullback_shadow_min = float(pullback_cfg.get("lower_wick_ratio_min", 0.3))
    pullback_min_days_since_high = float(pullback_cfg.get("min_days_since_high", 3.0))

    rebound_enabled = bool(rebound_cfg.get("enabled", True))
    rebound_bias_threshold = float(rebound_cfg.get("bias_threshold", -0.08))
    rebound_vol_max = float(rebound_cfg.get("vol_ratio_5_max", 1.2))
    rebound_rsi_max = float(rebound_cfg.get("rsi_max", 30.0))
    rebound_shadow_min = float(rebound_cfg.get("lower_wick_ratio_min", 0.35))

    close = _col(out, "close")
    ma20 = _col(out, "ma20")
    ma60 = _col(out, "ma60")
    ma20_slope = _col(out, "ma20_slope")
    breakout = _col(out, "breakout")
    breakout_rt = _col(out, "breakout_rt", default=float("nan"))
    breakout_flag = breakout.where(breakout_rt.isna(), breakout_rt).fillna(0.0)
    breakout_level = _col(out, "breakout_level")
    breakout_strength = _col(out, "breakout_strength")
    breakout_strength_rt = _col(out, "breakout_strength_rt", default=float("nan"))
    breakout_strength = breakout_strength.where(breakout_strength_rt.isna(), breakout_strength_rt).fillna(0.0)
    pct_chg = _col(out, "pct_chg")
    body = _col(out, "body")
    upper_wick = _col(out, "upper_wick").abs()
    lower_wick = _col(out, "lower_wick").abs()
    vol_ratio = _col(out, "vol_ratio")
    vol_ratio_5 = _col(out, "vol_ratio_5").fillna(vol_ratio)
    volume_change_20d = _col(out, "volume_change_20d", default=float("nan"))
    price_ma20_dist = _col(out, "price_ma20_dist", default=float("nan"))
    rsi_val = _col(out, "rsi", default=float("nan"))
    days_since_hhv = _col(out, "days_since_hhv", default=float("nan"))
    price_accel_5 = _col(out, "price_accel_5", default=float("nan"))
    price_accel_10 = _col(out, "price_accel_10", default=float("nan"))

    candle_range = (body.abs() + upper_wick + lower_wick).replace(0, pd.NA)
    lower_wick_ratio = (lower_wick / candle_range).fillna(0.0)
    price_up = (body > 0) | (pct_chg >= breakout_pct_min)

    breakout_trend_ok = (close > ma20) & (ma20 > ma60) & (ma20_slope > 0)
    breakout_vol_ok = vol_ratio_5 >= breakout_vol_threshold
    breakout_hit = breakout_enabled & (breakout_flag >= 1.0) & breakout_vol_ok & price_up & breakout_trend_ok
    breakout_raw = ((close / breakout_level.replace(0, pd.NA)) - 1.0).fillna(breakout_strength).clip(lower=0.0)
    breakout_volume_boost = vol_ratio_5.clip(lower=0.0, upper=3.0) / 2.0
    breakout_path_strength = (breakout_raw * breakout_volume_boost).where(breakout_hit, 0.0).clip(0.0, 1.0)

    pullback_trend_ok = (ma20 > ma60) & (ma20_slope > 0) & (close > ma60)
    pullback_dist_ok = price_ma20_dist.abs() <= pullback_max
    pullback_vol_ok = (vol_ratio_5 <= pullback_shrink_threshold) | (volume_change_20d < -0.1)
    pullback_stop_ok = (lower_wick_ratio >= pullback_shadow_min) | (pct_chg >= 0)
    pullback_days_ok = days_since_hhv.fillna(pullback_min_days_since_high) >= pullback_min_days_since_high
    pullback_hit = pullback_enabled & pullback_trend_ok & pullback_dist_ok & pullback_vol_ok & pullback_stop_ok & pullback_days_ok
    pullback_proximity = (1.0 - (price_ma20_dist.abs() / max(pullback_max, 1e-6))).clip(0.0, 1.0)
    pullback_volume_relief = (1.0 - (vol_ratio_5 / max(pullback_shrink_threshold, 1e-6))).clip(0.0, 1.0)
    pullback_path_strength = (pullback_proximity * (0.6 + 0.4 * pullback_volume_relief)).where(pullback_hit, 0.0).clip(0.0, 1.0)

    rebound_dist_ok = price_ma20_dist <= rebound_bias_threshold
    rebound_vol_ok = vol_ratio_5 <= rebound_vol_max
    rebound_rsi_ok = rsi_val <= rebound_rsi_max
    rebound_stop_ok = (lower_wick_ratio >= rebound_shadow_min) | ((body > 0) & (pct_chg > 0))
    rebound_hit = rebound_enabled & rebound_dist_ok & rebound_vol_ok & rebound_rsi_ok & rebound_stop_ok
    rebound_depth = (price_ma20_dist.abs() / max(abs(rebound_bias_threshold), 1e-6)).clip(0.0, 1.0)
    rebound_relief = (1.0 - (vol_ratio_5 / 2.0)).clip(0.0, 1.0)
    rebound_path_strength = (rebound_depth * rebound_relief).where(rebound_hit, 0.0).clip(0.0, 1.0)

    momentum_raw = (
        (price_accel_5 > 0)
        & (price_accel_10 == price_accel_10)
        & (price_accel_5 > price_accel_10)
        & (close > breakout_level)
    )
    momentum_path_strength = (
        (price_accel_5.clip(lower=0.0) * (close / breakout_level.replace(0, pd.NA)).clip(lower=0.0)).fillna(0.0)
    ).where(momentum_raw, 0.0).clip(0.0, 1.0)

    path_weights = weights.get("path_weights", {}) if isinstance(weights, dict) else {}
    w_break = float(path_weights.get("breakout", 0.45))
    w_pull = float(path_weights.get("pullback", 0.35))
    w_rebound = float(path_weights.get("rebound", 0.20))
    w_momentum = float(path_weights.get("momentum", 0.0))
    weight_sum = max(w_break + w_pull + w_rebound + w_momentum, 1e-6)

    out["path_breakout_hit"] = breakout_hit.astype(int)
    out["path_pullback_hit"] = pullback_hit.astype(int)
    out["path_rebound_hit"] = rebound_hit.astype(int)
    out["path_breakout_strength"] = breakout_path_strength
    out["path_pullback_strength"] = pullback_path_strength
    out["path_rebound_strength"] = rebound_path_strength
    out["path_momentum_strength"] = momentum_path_strength
    out["path_score"] = (
        breakout_path_strength * w_break
        + pullback_path_strength * w_pull
        + rebound_path_strength * w_rebound
        + momentum_path_strength * w_momentum
    ) / weight_sum

    best_names = []
    for _, row in out[["path_breakout_strength", "path_pullback_strength", "path_rebound_strength"]].iterrows():
        vals = {
            "breakout": float(row.get("path_breakout_strength", 0.0) or 0.0),
            "pullback": float(row.get("path_pullback_strength", 0.0) or 0.0),
            "rebound": float(row.get("path_rebound_strength", 0.0) or 0.0),
        }
        best_name = max(vals, key=vals.get)
        best_names.append(best_name if vals[best_name] > 0 else "none")
    out["best_path"] = best_names
    return out


def score_and_rank(
    df: pd.DataFrame,
    weights: dict,
    top_n: int = 20,
    sector_cfg: dict | None = None,
    decision_cfg: dict | None = None,
) -> pd.DataFrame:
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

    out["rs_score"] = _clip01(_col(out, "rs_proxy"))
    out["trend_score"] = _clip01(_col(out, "trend"))
    out["volume_score"] = _clip01(_col(out, "vol_ratio") / 2.5)
    if "rsi" in out.columns:
        out["rsi_score"] = (1 - (_col(out, "rsi").fillna(50) - 60).abs() / 40).clip(0, 1)
    else:
        out["rsi_score"] = 0.0
    if "macd_hist" in out.columns:
        out["macd_score"] = (_col(out, "macd_hist").fillna(0) > 0).astype(float)
    else:
        out["macd_score"] = 0.0

    risk_raw = (_clip01(_col(out, "atr_pct") / 0.08) * 0.6) + (_clip01(_col(out, "mdd20") / 0.12) * 0.4)
    out["risk_penalty"] = (risk_raw * risk_penalty_max).clip(0, risk_penalty_max)

    breakout_col = "breakout"
    if "breakout_rt" in out.columns and out["breakout_rt"].notna().any():
        breakout_col = "breakout_rt"

    out["rule_score_legacy"] = (
        w_breakout * _col(out, breakout_col).fillna(0.0)
        + w_rs * out["rs_score"]
        + w_trend * out["trend_score"]
        + w_volume * out["volume_score"]
        + w_rsi * out["rsi_score"]
        + w_macd * out["macd_score"]
        - out["risk_penalty"]
    )

    out["factor_score"] = _compute_factor_score(out, weights)
    out = _compute_path_scores(out, weights, decision_cfg=decision_cfg)

    path_blend = float(weights.get("path_blend", 0.65))
    path_score_100 = out["path_score"] * 100.0

    if score_mode == "factor":
        out["score"] = (out["factor_score"] * 100).clip(0, 100)
    elif score_mode in ("hybrid", "mix", "blend"):
        hybrid_rule = float(weights.get("hybrid_rule", 0.7))
        hybrid_factor = float(weights.get("hybrid_factor", 0.3))
        base_score = out["rule_score_legacy"] * hybrid_rule + out["factor_score"] * 100.0 * hybrid_factor
        out["score"] = base_score * (1.0 - path_blend) + path_score_100 * path_blend
    else:
        out["score"] = out["rule_score_legacy"] * (1.0 - path_blend) + path_score_100 * path_blend

    out["rule_score"] = out["score"]

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
        if require_breakout and "path_breakout_hit" in out.columns:
            mask = mask & (_col(out, "path_breakout_hit") >= 1.0)
        elif require_breakout and "breakout" in out.columns:
            mask = mask & (_col(out, "breakout") >= 1.0)
        if require_vol_ok and "vol_ok" in out.columns:
            mask = mask & (_col(out, "vol_ok") >= 1.0)

        counts = out.loc[mask].groupby("sector").size().to_dict()
        out["sector_strength"] = out["sector"].map(counts).fillna(0).astype(int)
        out["sector_boost"] = (out["sector_strength"] * boost_per_stock).clip(0, max_boost)
        out.loc[out["sector_strength"] < min_count, "sector_boost"] = 0.0
        out["score"] = out["score"] + out["sector_boost"]
    else:
        out["sector_strength"] = 0
        out["sector_boost"] = 0.0

    out = out.sort_values(
        ["score", "path_score", "path_breakout_strength", "path_pullback_strength", "vol_ratio", "amount"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    def _build_trigger(r: pd.Series) -> str:
        path = str(r.get("best_path", "") or "")
        if path == "pullback":
            return f"靠近MA20({float(r.get('ma20', 0.0)):.2f})缩量止跌"
        if path == "rebound":
            return "超跌后关注止跌反包/下影线修复"
        return f"站上{float(r.get('hhv', 0.0)):.2f}并放量确认"

    def _build_stop(r: pd.Series) -> str:
        path = str(r.get("best_path", "") or "")
        if path == "rebound":
            return "跌破当日止跌低点"
        if path == "pullback":
            return f"跌破MA20({float(r.get('ma20', 0.0)):.2f})"
        return f"跌破MA20({float(r.get('ma20', 0.0)):.2f})"

    out["buy_trigger"] = out.apply(_build_trigger, axis=1)
    out["stop_ref"] = out.apply(_build_stop, axis=1)

    return out.head(int(top_n))
