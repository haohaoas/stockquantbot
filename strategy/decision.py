from __future__ import annotations

import math
import pandas as pd


def apply_env_overrides(decision_cfg: dict, market_env: str) -> dict:
    cfg = decision_cfg or {}
    env = str(market_env or "").strip()
    if not env:
        return cfg
    env_overrides = cfg.get("env_overrides") or {}
    if not isinstance(env_overrides, dict):
        return cfg
    if env not in env_overrides:
        return cfg
    merged = dict(cfg)
    merged.update(env_overrides.get(env) or {})
    merged["_env_used"] = env
    return merged


def apply_short_term_decision(df: pd.DataFrame, decision_cfg: dict) -> pd.DataFrame:
    if df.empty:
        return df

    cfg = decision_cfg or {}
    min_score = float(cfg.get("min_score", 20))
    enable_breakout = bool(cfg.get("enable_breakout", True))
    require_breakout = bool(cfg.get("require_breakout", True))
    breakout_min_score = float(cfg.get("breakout_min_score", min_score))
    require_vol_ok = bool(cfg.get("require_vol_ok", True))
    require_shrink_vol = bool(cfg.get("require_shrink_vol", False))
    shrink_vol_ratio_max = float(cfg.get("shrink_vol_ratio_max", 1.0))
    shrink_require_20d = bool(cfg.get("shrink_require_20d", False))
    require_trend_confirmation = bool(cfg.get("require_trend_confirmation", True))
    min_trend = float(cfg.get("min_trend", 0.8))
    max_atr_pct = float(cfg.get("max_atr_pct", 0.08))
    max_mdd20 = float(cfg.get("max_mdd20", 0.12))
    watch_near_hhv_pct = float(cfg.get("watch_near_hhv_pct", 2.0))
    breakout_confirm_days = int(cfg.get("breakout_confirm_days", 0))
    breakout_min_volume_multiple = float(cfg.get("breakout_min_volume_multiple", cfg.get("vol_ratio_min", 0.0)))
    require_close_above_breakout = bool(cfg.get("require_close_above_breakout", False))
    breakout_within_days = int(cfg.get("breakout_within_days", 0))
    breakout_threshold = float(cfg.get("breakout_threshold", 0.0))
    require_volume_persistence = bool(cfg.get("require_volume_persistence", False))
    allow_weak_breakout = bool(cfg.get("allow_weak_breakout", False))
    require_macd_confirmation = bool(cfg.get("require_macd_confirmation", False))
    require_bollinger_breakout = bool(cfg.get("require_bollinger_breakout", False))
    stop_atr_mult = float(cfg.get("stop_atr_mult", 2.0))
    target_rr = float(cfg.get("target_rr", 2.0))
    enable_pullback = bool(cfg.get("enable_pullback", True))
    pullback_ma = int(cfg.get("pullback_ma", 20))
    pullback_max_pct = float(cfg.get("pullback_max_pct", 2.0))
    pullback_min_score = float(cfg.get("pullback_min_score", max(min_score * 0.6, 10)))
    pullback_require_vol_ok = bool(cfg.get("pullback_require_vol_ok", False))
    pullback_vol_ratio_min = float(cfg.get("pullback_vol_ratio_min", 0.0))
    pullback_require_reclaim_ma = bool(cfg.get("pullback_require_reclaim_ma", True))
    pullback_reclaim_ma_period = int(cfg.get("pullback_reclaim_ma_period", 0))
    pullback_require_rsi = bool(cfg.get("pullback_require_rsi", True))
    pullback_rsi_min = float(cfg.get("pullback_rsi_min", 40))
    pullback_max_rsi = float(cfg.get("pullback_max_rsi", 0.0))
    pullback_max_price_from_ma = float(cfg.get("pullback_max_price_from_ma", 0.0))
    pullback_min_days_since_high = float(cfg.get("pullback_min_days_since_high", 0.0))
    pullback_require_price_above_ma20 = bool(cfg.get("pullback_require_price_above_ma20", False))
    pullback_require_pct = bool(cfg.get("pullback_require_pct", True))
    pullback_min_pct = float(cfg.get("pullback_min_pct", -2.0))
    pullback_require_macd = bool(cfg.get("pullback_require_macd", False))
    pullback_override_ratio = float(cfg.get("pullback_override_ratio", 1.5))
    post_filter_rsi_min = float(cfg.get("post_filter_rsi_min", 0.0))
    post_filter_rsi_max = float(cfg.get("post_filter_rsi_max", 100.0))
    post_filter_macd_positive = bool(cfg.get("post_filter_macd_positive", False))
    weibi_enabled = bool(cfg.get("weibi_enabled", True))
    weibi_key_signal_only = bool(cfg.get("weibi_key_signal_only", True))
    weibi_pos_min = float(cfg.get("weibi_pos_min", 5.0))
    weibi_neg_max = float(cfg.get("weibi_neg_max", -5.0))
    weibi_strong_pos = float(cfg.get("weibi_strong_pos", 12.0))
    weibi_strong_neg = float(cfg.get("weibi_strong_neg", -12.0))
    weibi_breakout_neg_reduce_scale = float(cfg.get("weibi_breakout_neg_reduce_scale", 0.6))
    weibi_pullback_pos_boost_scale = float(cfg.get("weibi_pullback_pos_boost_scale", 1.2))
    weibi_pullback_neg_reduce_scale = float(cfg.get("weibi_pullback_neg_reduce_scale", 0.85))
    intraday_v2_cfg = cfg.get("intraday_v2") or {}
    daily_gate_enabled = bool(intraday_v2_cfg.get("daily_gate_enabled", True))
    daily_close_ma20_ratio = float(intraday_v2_cfg.get("daily_close_ma20_ratio", 0.98))

    out = df.copy()

    actions = []
    entries = []
    stops = []
    targets = []
    risks = []
    position_scales = []
    reasons = []
    paths = []

    for _, r in out.iterrows():
        score = float(r.get("score", 0))
        last_close = float(r.get("close", 0))
        hhv_val = float(r.get("hhv", 0))
        breakout = int(r.get("breakout", 0))
        breakout_level = float(r.get("breakout_level", 0))
        breakout_strength = float(r.get("breakout_strength", 0))
        breakout_confirm = float(r.get("breakout_confirm", 0))
        breakout_days_since = float(r.get("breakout_days_since", float("nan")))
        breakout_rt = float(r.get("breakout_rt", float("nan")))
        if breakout_rt == breakout_rt:
            breakout = int(breakout_rt)
        breakout_strength_rt = float(r.get("breakout_strength_rt", float("nan")))
        if breakout_strength_rt == breakout_strength_rt:
            breakout_strength = breakout_strength_rt
        breakout_confirm_rt = float(r.get("breakout_confirm_rt", float("nan")))
        if breakout_confirm_rt == breakout_confirm_rt:
            breakout_confirm = breakout_confirm_rt
        breakout_days_since_rt = float(r.get("breakout_days_since_rt", float("nan")))
        if breakout_days_since_rt == breakout_days_since_rt:
            breakout_days_since = breakout_days_since_rt
        vol_ok = int(r.get("vol_ok", 0))
        volume_shrink_20d = float(r.get("volume_shrink_20d", float("nan")))
        vol_ratio_min_3 = float(r.get("vol_ratio_min_3", float("nan")))
        vol_ratio_min_5 = float(r.get("vol_ratio_min_5", float("nan")))
        trend = float(r.get("trend", 0))
        atr = float(r.get("atr", 0))
        atr_pct = float(r.get("atr_pct", 0))
        mdd20 = float(r.get("mdd20", 0))
        ma20 = float(r.get("ma20", 0))
        ma60 = float(r.get("ma60", 0))
        ma10 = float(r.get("ma10", 0))
        ma30 = float(r.get("ma30", 0))
        rsi_val = float(r.get("rsi", float("nan")))
        ma20_slope = float(r.get("ma20_slope", float("nan")))
        macd_hist = float(r.get("macd_hist", float("nan")))
        bb_upper = float(r.get("bb_upper", float("nan")))
        pct_chg = float(r.get("pct_chg", 0))
        vol_ratio = float(r.get("vol_ratio", 0))
        weibi = float(pd.to_numeric(r.get("weibi", float("nan")), errors="coerce"))
        days_since_hhv = float(r.get("days_since_hhv", float("nan")))
        sector_boost = float(r.get("sector_boost", 0.0))
        pullback_vol_ok = False

        cond_score_breakout = score >= breakout_min_score
        cond_trend = (trend >= min_trend) if require_trend_confirmation else True
        cond_risk = (atr_pct <= max_atr_pct) and (mdd20 <= max_mdd20)
        cond_daily_gate = True
        if daily_gate_enabled:
            cond_daily_gate = (
                (last_close > ma60 if ma60 > 0 else False)
                and (ma20 > ma60 if ma20 > 0 and ma60 > 0 else False)
                and (ma20_slope > 0 if ma20_slope == ma20_slope else False)
                and (mdd20 <= 0.12 if mdd20 == mdd20 else False)
                and (last_close >= ma20 * daily_close_ma20_ratio if ma20 > 0 else False)
            )

        cond_breakout = (breakout == 1) if require_breakout else True
        if cond_breakout and require_close_above_breakout and breakout_level > 0:
            cond_breakout = last_close >= breakout_level
        if cond_breakout and breakout_confirm_days > 0:
            cond_breakout = breakout_confirm >= breakout_confirm_days
        if cond_breakout and breakout_within_days > 0:
            if breakout_days_since != breakout_days_since:
                cond_breakout = False
            else:
                cond_breakout = breakout_days_since <= breakout_within_days
        if cond_breakout and breakout_threshold > 0 and not allow_weak_breakout:
            cond_breakout = breakout_strength >= breakout_threshold
        if cond_breakout and require_macd_confirmation:
            cond_breakout = (macd_hist == macd_hist) and macd_hist > 0
        if cond_breakout and require_bollinger_breakout:
            cond_breakout = (bb_upper == bb_upper) and last_close >= bb_upper

        if require_vol_ok:
            if breakout_min_volume_multiple > 0:
                cond_vol = vol_ratio >= breakout_min_volume_multiple
            else:
                cond_vol = vol_ok == 1
            if cond_vol and require_volume_persistence:
                window = 3 if breakout_confirm_days <= 3 else 5
                if window == 3 and vol_ratio_min_3 == vol_ratio_min_3:
                    cond_vol = vol_ratio_min_3 >= breakout_min_volume_multiple
                elif window == 5 and vol_ratio_min_5 == vol_ratio_min_5:
                    cond_vol = vol_ratio_min_5 >= breakout_min_volume_multiple
        else:
            cond_vol = True

        if require_shrink_vol:
            shrink_ratio_ok = (vol_ratio == vol_ratio) and (vol_ratio <= shrink_vol_ratio_max)
            shrink_20d_ok = (volume_shrink_20d == volume_shrink_20d) and (volume_shrink_20d >= 1.0)
            cond_vol = (shrink_ratio_ok and shrink_20d_ok) if shrink_require_20d else (shrink_ratio_ok or shrink_20d_ok)
        else:
            shrink_ratio_ok = False
            shrink_20d_ok = False

        # Volume/price morphology for weibi policy.
        shrink_ratio_tag = (vol_ratio == vol_ratio) and (vol_ratio <= shrink_vol_ratio_max)
        shrink_20d_tag = (volume_shrink_20d == volume_shrink_20d) and (volume_shrink_20d >= 1.0)
        is_shrink = (shrink_ratio_tag and shrink_20d_tag) if shrink_require_20d else (shrink_ratio_tag or shrink_20d_tag)
        is_expand = (vol_ratio == vol_ratio) and (vol_ratio > 1.05) and not is_shrink
        price_up = pct_chg >= 0

        near_breakout = False
        if hhv_val > 0:
            near_breakout = last_close >= hhv_val * (1 - watch_near_hhv_pct / 100.0)

        action = "AVOID"
        signal_path = ""
        pullback = False
        position_scale = 1.0

        def _ma_by_period(period: int) -> float:
                if period == 5:
                    return float(r.get("ma5", ma20))
                if period == 10:
                    return ma10
                if period == 20:
                    return ma20
                if period == 30:
                    return ma30
                if period == 60:
                    return ma60
                return ma20

        ma_ref_period = pullback_reclaim_ma_period if pullback_reclaim_ma_period > 0 else pullback_ma
        ma_ref = _ma_by_period(ma_ref_period)
        dist = float("nan")
        max_dist_pct = pullback_max_pct
        if enable_pullback and ma_ref > 0:
            dist = abs(last_close / ma_ref - 1) * 100
            if pullback_max_price_from_ma > 0:
                if pullback_max_price_from_ma <= 1:
                    max_dist_pct = pullback_max_price_from_ma * 100
                else:
                    max_dist_pct = pullback_max_price_from_ma
            if require_shrink_vol:
                pullback_vol_ok = (shrink_ratio_ok and shrink_20d_ok) if shrink_require_20d else (shrink_ratio_ok or shrink_20d_ok)
            elif pullback_require_vol_ok:
                pullback_vol_ok = (
                    (vol_ratio >= pullback_vol_ratio_min)
                    if pullback_vol_ratio_min > 0
                    else (vol_ok == 1)
                )
            else:
                pullback_vol_ok = True
            pullback = (
                dist <= max_dist_pct
                and trend >= min_trend
                and cond_risk
                and cond_daily_gate
                and score >= pullback_min_score
                and pullback_vol_ok
            )
            if pullback and pullback_require_reclaim_ma:
                pullback = last_close >= ma_ref
            if pullback and pullback_require_rsi:
                pullback = (rsi_val == rsi_val) and rsi_val >= pullback_rsi_min
                if pullback and pullback_max_rsi > 0:
                    pullback = rsi_val <= pullback_max_rsi
            if pullback and pullback_require_price_above_ma20:
                pullback = last_close >= ma20
            if pullback and pullback_min_days_since_high > 0:
                if days_since_hhv != days_since_hhv:
                    pullback = False
                else:
                    pullback = days_since_hhv >= pullback_min_days_since_high
            if pullback and pullback_require_pct:
                pullback = pct_chg >= pullback_min_pct
            if pullback and pullback_require_macd:
                pullback = (macd_hist == macd_hist) and macd_hist > 0

        main_ok = enable_breakout and cond_score_breakout and cond_breakout and cond_vol and cond_trend and cond_risk and cond_daily_gate
        if main_ok and pullback:
            # Strength comparison to decide path
            base_strength = max(breakout_threshold, 0.02)
            breakout_norm = min(1.0, max(0.0, breakout_strength / base_strength))
            vol_base = max(breakout_min_volume_multiple, 1.2)
            vol_norm = min(1.0, max(0.0, vol_ratio / vol_base))
            main_strength = breakout_norm * vol_norm

            depth_norm = 0.0
            if dist == dist and max_dist_pct > 0:
                depth_norm = max(0.0, min(1.0, 1.0 - dist / max_dist_pct))
            support_strength = 0.5
            if ma20_slope == ma20_slope:
                support_strength = max(0.2, min(1.0, 0.5 + 10.0 * ma20_slope))
            pullback_strength = depth_norm * support_strength

            if pullback_strength > main_strength * pullback_override_ratio:
                action = "BUY"
                signal_path = "pullback"
            else:
                action = "BUY"
                signal_path = "main"
        elif main_ok:
            action = "BUY"
            signal_path = "main"
        elif pullback:
            action = "BUY"
            signal_path = "pullback"
        elif near_breakout and trend >= 0.4:
            action = "WATCH"
            signal_path = "watch"

        weibi_hint = ""
        if weibi_enabled and (weibi == weibi):
            key_signal = signal_path in ("main", "pullback")
            if (not weibi_key_signal_only) or key_signal:
                # BREAKOUT路径：委比负只降仓，不放弃信号。
                if action == "BUY" and signal_path == "main":
                    if weibi <= weibi_neg_max:
                        position_scale *= weibi_breakout_neg_reduce_scale
                        weibi_hint = "突破委比为负，建议降仓参与"
                    elif is_expand and weibi >= weibi_strong_pos:
                        weibi_hint = "突破委比偏强，买盘主动"

                # PULLBACK路径：委比正增强信心；委比负仅轻度降仓。
                if action == "BUY" and signal_path == "pullback":
                    if weibi >= weibi_pos_min:
                        position_scale *= weibi_pullback_pos_boost_scale
                        weibi_hint = "回踩委比为正，低吸信心增强"
                    elif weibi <= weibi_neg_max:
                        position_scale *= weibi_pullback_neg_reduce_scale
                        weibi_hint = "回踩委比偏弱，建议轻仓试错"

        # Entry/stop/target
        entry = last_close if last_close > 0 else float("nan")
        stop = float("nan")
        target = float("nan")
        risk_pct = float("nan")

        if entry > 0:
            stop_atr = entry - stop_atr_mult * atr if atr > 0 else float("nan")
            candidates = [x for x in [ma20, stop_atr] if x and not math.isnan(x) and x > 0]
            if candidates:
                stop = min(candidates)
            if stop > 0 and stop < entry:
                risk = entry - stop
                risk_pct = risk / entry * 100
                target = entry + target_rr * risk

        reason_parts: list[str] = []
        if action == "BUY":
            if signal_path == "pullback":
                reason_parts.append("趋势回踩MA20")
                if require_shrink_vol and pullback_vol_ok:
                    reason_parts.append("缩量")
                elif pullback_require_vol_ok and pullback_vol_ok:
                    reason_parts.append("放量")
                if cond_risk:
                    reason_parts.append("风险可控")
                if cond_score_breakout:
                    reason_parts.append("评分达标")
            else:
                if cond_breakout:
                    reason_parts.append("突破新高")
                if require_shrink_vol and cond_vol:
                    reason_parts.append("缩量")
                elif require_vol_ok and cond_vol:
                    reason_parts.append("放量")
                if cond_trend:
                    reason_parts.append("趋势强")
                if cond_risk:
                    reason_parts.append("风险可控")
                if cond_score_breakout:
                    reason_parts.append("评分达标")
            if sector_boost > 0:
                reason_parts.append("板块共振")
        elif action == "WATCH":
            reason_parts.append("接近新高")
            if trend >= 0.4:
                reason_parts.append("趋势尚可")
            if sector_boost > 0:
                reason_parts.append("板块共振")
        else:
            if enable_breakout and not cond_score_breakout:
                reason_parts.append("评分不足")
            if enable_breakout and not cond_breakout:
                reason_parts.append("未突破")
            if require_shrink_vol and not cond_vol:
                reason_parts.append("未缩量")
            elif not cond_vol:
                reason_parts.append("放量不足")
            if not cond_trend:
                reason_parts.append("趋势偏弱")
            if not cond_risk:
                reason_parts.append("波动/回撤偏高")
            if not cond_daily_gate:
                reason_parts.append("日线准入未通过")
            if require_close_above_breakout and breakout_level > 0 and last_close < breakout_level:
                reason_parts.append("收盘未站上突破位")
            if breakout_confirm_days > 0 and breakout_confirm < breakout_confirm_days:
                reason_parts.append("突破确认不足")
            if breakout_within_days > 0 and (breakout_days_since != breakout_days_since or breakout_days_since > breakout_within_days):
                reason_parts.append("突破不够新")
            if breakout_threshold > 0 and not allow_weak_breakout and breakout_strength < breakout_threshold:
                reason_parts.append("突破力度不足")
            if require_macd_confirmation and not ((macd_hist == macd_hist) and macd_hist > 0):
                reason_parts.append("MACD未确认")
            if require_bollinger_breakout and not ((bb_upper == bb_upper) and last_close >= bb_upper):
                reason_parts.append("未破布林上轨")

        # Post filters (RSI/MACD)
        if action == "BUY":
            if post_filter_macd_positive and not ((macd_hist == macd_hist) and macd_hist > 0):
                action = "AVOID"
                reason_parts.append("MACD过滤")
            if rsi_val == rsi_val:
                if rsi_val < post_filter_rsi_min:
                    action = "AVOID"
                    reason_parts.append("RSI偏低")
                if rsi_val > post_filter_rsi_max:
                    action = "AVOID"
                    reason_parts.append("RSI偏高")
            if action != "BUY":
                signal_path = "filtered"

        position_scale = max(0.3, min(1.5, float(position_scale)))
        if weibi_hint and weibi_hint not in reason_parts:
            reason_parts.append(weibi_hint)


        actions.append(action)
        entries.append(entry)
        stops.append(stop)
        targets.append(target)
        risks.append(risk_pct)
        position_scales.append(position_scale)
        reasons.append("；".join(reason_parts) if reason_parts else "")
        paths.append(signal_path)

    out["action"] = actions
    out["entry"] = entries
    out["stop"] = stops
    out["target"] = targets
    out["risk_pct"] = risks
    out["position_scale"] = position_scales
    out["reason"] = reasons
    out["signal_path"] = paths

    return out
