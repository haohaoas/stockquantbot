from __future__ import annotations

import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from sqdata.akshare_fetcher import fetch_hist, fetch_a_share_daily_panel
from sqdata.fetcher import get_realtime
from features.indicators import sma, hhv, atr, max_drawdown, rsi, macd


def build_factors(
    spot_df: pd.DataFrame,
    signals_cfg: dict,
    trade_date: str,
    no_cache: bool = False,
    stats: dict | None = None,
) -> pd.DataFrame:
    """
    For each candidate in spot_df, fetch recent history and compute short-term factors.
    Returns a dataframe with factor columns.
    """
    lookback_hhv = int(signals_cfg.get("lookback_hhv", 20))
    ma_fast = int(signals_cfg.get("ma_fast", 5))
    ma_mid = int(signals_cfg.get("ma_mid", 20))
    ma_slow = int(signals_cfg.get("ma_slow", 60))
    vol_ratio_min = float(signals_cfg.get("vol_ratio_min", 1.5))
    hist_days = int(signals_cfg.get("hist_days", 120))
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    use_cache = bool(signals_cfg.get("hist_cache", True))
    cache_ttl_sec = int(signals_cfg.get("hist_cache_ttl_sec", 6 * 3600))
    cache_dir = str(signals_cfg.get("hist_cache_dir", "./cache/hist"))
    allow_akshare_fallback = bool(signals_cfg.get("allow_akshare_fallback", True))
    use_parallel = bool(signals_cfg.get("parallel", True))
    max_workers = int(signals_cfg.get("max_workers", 8))
    rs_window = int(signals_cfg.get("rs_window", 20))
    rsi_window = int(signals_cfg.get("rsi_window", 14))
    macd_fast = int(signals_cfg.get("macd_fast", 12))
    macd_slow = int(signals_cfg.get("macd_slow", 26))
    macd_signal = int(signals_cfg.get("macd_signal", 9))
    realtime_fix = bool(signals_cfg.get("realtime_fix", True))
    realtime_fix_limit = int(signals_cfg.get("realtime_fix_limit", 200))
    realtime_fix_batch = bool(signals_cfg.get("realtime_fix_batch", True))
    realtime_fix_batch_limit = int(signals_cfg.get("realtime_fix_batch_limit", 0))
    realtime_fix_single_fallback = bool(signals_cfg.get("realtime_fix_single_fallback", True))
    realtime_allow_direct_fallback = bool(signals_cfg.get("realtime_allow_direct_fallback", True))
    realtime_provider = str(signals_cfg.get("realtime_provider", "auto")).lower()
    intraday_breakout = bool(signals_cfg.get("intraday_breakout", False))

    rows = []
    lock = threading.Lock()

    def _inc(key: str, symbol: str | None = None) -> None:
        if stats is None:
            return
        stats[key] = int(stats.get(key, 0)) + 1
        if symbol:
            list_key = f"{key}_symbols"
            lst = stats.get(list_key) or []
            if len(lst) < 10 and symbol not in lst:
                lst.append(symbol)
            stats[list_key] = lst
    records = spot_df.to_dict("records")
    fix_budget = {"remaining": realtime_fix_limit}
    fix_quotes: dict[str, tuple[float, float]] = {}

    if realtime_fix and realtime_fix_batch and not spot_df.empty:
        close_ser = pd.to_numeric(spot_df.get("close"), errors="coerce")
        sym_ser = pd.to_numeric(spot_df.get("symbol"), errors="coerce")
        bad_close_mask = close_ser.isna() | (close_ser <= 0) | (close_ser >= 10000) | ((close_ser - sym_ser).abs() < 1e-6)
        fix_symbols = (
            spot_df.loc[bad_close_mask, "symbol"].astype(str).str.zfill(6).drop_duplicates().tolist()
            if bad_close_mask.any()
            else []
        )
        if realtime_fix_batch_limit > 0:
            fix_symbols = fix_symbols[:realtime_fix_batch_limit]
        if stats is not None:
            stats["spot_fix_batch_candidates"] = int(len(fix_symbols))
        if fix_symbols:
            t_batch = time.perf_counter()
            try:
                fix_cfg = dict(signals_cfg)
                fix_cfg["symbols"] = fix_symbols
                fix_df = fetch_a_share_daily_panel(
                    trade_date=trade_date,
                    signals_cfg=fix_cfg,
                    no_cache=False,
                    use_proxy=use_proxy,
                    proxy=proxy,
                    allow_eastmoney_fallback=False,
                )
                if fix_df is not None and not fix_df.empty:
                    fix_df = fix_df.copy()
                    fix_df["symbol"] = fix_df["symbol"].astype(str).str.zfill(6)
                    fix_df["close"] = pd.to_numeric(fix_df.get("close"), errors="coerce")
                    fix_df["pct_chg"] = pd.to_numeric(fix_df.get("pct_chg"), errors="coerce")
                    fix_df = fix_df[(fix_df["close"] > 0) & (fix_df["close"] < 10000)]
                    fix_quotes = {
                        str(row["symbol"]): (float(row["close"]), float(row["pct_chg"]))
                        for _, row in fix_df[["symbol", "close", "pct_chg"]].dropna(subset=["symbol", "close"]).iterrows()
                    }
            except Exception:
                if stats is not None:
                    stats["spot_fix_batch_error"] = int(stats.get("spot_fix_batch_error", 0)) + 1
            finally:
                if stats is not None:
                    elapsed = (time.perf_counter() - t_batch) * 1000.0
                    stats["time_realtime_fix_batch_ms"] = round(float(stats.get("time_realtime_fix_batch_ms", 0.0)) + elapsed, 2)
                    stats["spot_fix_batch_hits"] = int(len(fix_quotes))

    def _compute_row(r: dict) -> dict | None:
        symbol = str(r["symbol"])
        name = str(r.get("name", ""))
        row_t0 = time.perf_counter()
        hist_ms = 0.0
        rt_fix_ms = 0.0
        done_timing = False

        def _commit_timing() -> None:
            nonlocal done_timing
            if done_timing:
                return
            done_timing = True
            if stats is None:
                return
            row_ms = (time.perf_counter() - row_t0) * 1000.0
            calc_ms = max(row_ms - hist_ms - rt_fix_ms, 0.0)
            with lock:
                stats["time_hist_io_ms"] = round(float(stats.get("time_hist_io_ms", 0.0)) + hist_ms, 2)
                stats["time_realtime_fix_ms"] = round(float(stats.get("time_realtime_fix_ms", 0.0)) + rt_fix_ms, 2)
                stats["time_factor_calc_ms"] = round(float(stats.get("time_factor_calc_ms", 0.0)) + calc_ms, 2)

        t_hist = time.perf_counter()
        try:
            hist = fetch_hist(
                symbol,
                end_date=trade_date,
                hist_days=hist_days,
                no_cache=no_cache,
                use_proxy=use_proxy,
                proxy=proxy,
                cache_dir=cache_dir,
                cache_ttl_sec=cache_ttl_sec,
                use_cache=use_cache,
                allow_akshare_fallback=allow_akshare_fallback,
            )
            hist_ms += (time.perf_counter() - t_hist) * 1000.0
        except Exception as e:
            hist_ms += (time.perf_counter() - t_hist) * 1000.0
            with lock:
                _inc("hist_error", symbol)
            if stats is not None:
                err_key = "hist_error_msgs"
                with lock:
                    msgs = stats.get(err_key) or []
                    if len(msgs) < 5:
                        msgs.append(f"{symbol}: {e}")
                    stats[err_key] = msgs
            _commit_timing()
            return None

        if hist.empty:
            with lock:
                _inc("hist_empty", symbol)
            _commit_timing()
            return None
        if len(hist) < max(ma_slow, lookback_hhv) + 2:
            with lock:
                _inc("hist_short", symbol)
            _commit_timing()
            return None

        close = hist["close"]
        volume = hist["volume"]

        ma5_series = sma(close, ma_fast)
        ma20_series = sma(close, ma_mid)
        ma60_series = sma(close, ma_slow)
        ma10_series = sma(close, 10)
        ma30_series = sma(close, 30)

        ma5 = ma5_series.iloc[-1]
        ma20 = ma20_series.iloc[-1]
        ma60 = ma60_series.iloc[-1]
        ma10 = ma10_series.iloc[-1]
        ma30 = ma30_series.iloc[-1]
        ma60_prev = float(ma60_series.iloc[-2]) if len(ma60_series) >= 2 else float("nan")

        # Breakout: close >= HHV(lookback)
        hhv20 = hhv(close, lookback_hhv).iloc[-1]
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
        hhv_prev = hhv(close, lookback_hhv).shift(1).iloc[-1]
        breakout_level = float(hhv_prev) if hhv_prev == hhv_prev else float(hhv20)
        breakout = 1 if last_close >= breakout_level else 0
        breakout_strength = (last_close / breakout_level - 1.0) if breakout_level > 0 else 0.0
        breakout_confirm = 0.0
        breakout_days_since = float("nan")
        if len(close) >= 2:
            window = close.tail(lookback_hhv)
            if not window.empty and breakout_level > 0:
                above = (window >= breakout_level).to_numpy()
                if above.any():
                    # Days since last close above breakout level (0 if today is above).
                    last_true = int(np.where(above)[0][-1])
                    breakout_days_since = float(len(above) - 1 - last_true)
                    # Consecutive confirmations from the end.
                    cnt = 0
                    for v in above[::-1]:
                        if v:
                            cnt += 1
                        else:
                            break
                    breakout_confirm = float(cnt)
        days_since_hhv = float("nan")
        if len(close) >= lookback_hhv:
            window = close.tail(lookback_hhv)
            if not window.empty:
                max_idx = int(window.values.argmax())
                days_since_hhv = float(len(window) - 1 - max_idx)

        # Volume ratio: today vs avg(20)
        avg_vol_20 = volume.rolling(20, min_periods=20).mean().iloc[-1]
        vol_std_20 = volume.rolling(20, min_periods=20).std().iloc[-1]
        vol_ratio = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 and avg_vol_20 > 0 else 0.0
        vol_ok = 1 if vol_ratio >= vol_ratio_min else 0
        vol_z20 = float((volume.iloc[-1] - avg_vol_20) / vol_std_20) if vol_std_20 and vol_std_20 > 0 else 0.0
        vol_ratio_series = volume / volume.rolling(20, min_periods=20).mean()
        vol_ratio_min_3 = float("nan")
        vol_ratio_min_5 = float("nan")
        if len(vol_ratio_series) >= 3:
            vol_ratio_min_3 = float(vol_ratio_series.tail(3).min())
        if len(vol_ratio_series) >= 5:
            vol_ratio_min_5 = float(vol_ratio_series.tail(5).min())

        # Trend: MA structure
        trend = 1.0 if (ma20 > ma60 and last_close > ma20) else (0.4 if last_close > ma20 else 0.0)

        # Risk: ATR% and max drawdown (20d)
        atr14 = atr(hist, 14).iloc[-1]
        atr_pct = float(atr14 / last_close) if last_close > 0 else 0.0
        mdd20 = abs(max_drawdown(close.tail(20)))

        # RSI / MACD
        rsi_val = float(rsi(close, rsi_window).iloc[-1]) if len(close) >= rsi_window else float("nan")
        rsi_prev = float(rsi(close, rsi_window).iloc[-2]) if len(close) >= (rsi_window + 1) else float("nan")
        rsi6 = float(rsi(close, 6).iloc[-1]) if len(close) >= 6 else float("nan")
        rsi24 = float(rsi(close, 24).iloc[-1]) if len(close) >= 24 else float("nan")
        macd_line, macd_signal_line, macd_hist = macd(close, macd_fast, macd_slow, macd_signal)
        macd_line_v = float(macd_line.iloc[-1]) if len(macd_line) > 0 else float("nan")
        macd_signal_v = float(macd_signal_line.iloc[-1]) if len(macd_signal_line) > 0 else float("nan")
        macd_hist_v = float(macd_hist.iloc[-1]) if len(macd_hist) > 0 else float("nan")

        # Bollinger bands (20)
        bb_mid = close.rolling(20, min_periods=20).mean().iloc[-1]
        bb_std = close.rolling(20, min_periods=20).std().iloc[-1]
        bb_upper = float(bb_mid + 2 * bb_std) if bb_mid == bb_mid and bb_std == bb_std else float("nan")
        bb_lower = float(bb_mid - 2 * bb_std) if bb_mid == bb_mid and bb_std == bb_std else float("nan")

        # Extra features for ML
        def _safe_ratio(num: float, den: float) -> float:
            if den and den > 0:
                return float(num / den)
            return float("nan")

        ma5_slope = float("nan")
        if len(ma5_series) >= 6:
            prev = float(ma5_series.iloc[-6])
            if prev > 0:
                ma5_slope = float(ma5 / prev - 1)

        ma10_slope = float("nan")
        if len(ma10_series) >= 6:
            prev = float(ma10_series.iloc[-6])
            if prev > 0:
                ma10_slope = float(ma10 / prev - 1)

        ma20_slope = float("nan")
        if len(ma20_series) >= 6:
            prev = float(ma20_series.iloc[-6])
            if prev > 0:
                ma20_slope = float(ma20 / prev - 1)

        ma60_slope = float("nan")
        if len(ma60_series) >= 6:
            prev = float(ma60_series.iloc[-6])
            if prev > 0:
                ma60_slope = float(ma60 / prev - 1)

        price_ma20_dist = _safe_ratio(last_close, float(ma20)) - 1 if ma20 and ma20 > 0 else float("nan")
        price_ma60_dist = _safe_ratio(last_close, float(ma60)) - 1 if ma60 and ma60 > 0 else float("nan")
        ma_gap_5_20 = _safe_ratio(float(ma5), float(ma20)) - 1 if ma20 and ma20 > 0 else float("nan")
        ma_gap_20_60 = _safe_ratio(float(ma20), float(ma60)) - 1 if ma60 and ma60 > 0 else float("nan")

        ret_1 = float("nan")
        if len(close) >= 2:
            base = float(close.iloc[-2])
            if base > 0:
                ret_1 = float(last_close / base - 1)

        ret_3 = float("nan")
        if len(close) >= 4:
            base = float(close.iloc[-4])
            if base > 0:
                ret_3 = float(last_close / base - 1)

        ret_5 = float("nan")
        if len(close) >= 6:
            base = float(close.iloc[-6])
            if base > 0:
                ret_5 = float(last_close / base - 1)

        ret_10 = float("nan")
        if len(close) >= 11:
            base = float(close.iloc[-11])
            if base > 0:
                ret_10 = float(last_close / base - 1)

        ret_20 = float("nan")
        if len(close) >= 21:
            base = float(close.iloc[-21])
            if base > 0:
                ret_20 = float(last_close / base - 1)

        price_accel_5 = float("nan")
        if ret_5 == ret_5 and ret_10 == ret_10:
            price_accel_5 = float(ret_5 - ret_10)

        price_accel_10 = float("nan")
        if ret_10 == ret_10 and ret_20 == ret_20:
            price_accel_10 = float(ret_10 - ret_20)

        price_vs_high_20 = float("nan")
        if hhv20 and hhv20 > 0:
            price_vs_high_20 = float(last_close / hhv20)

        price_vs_ma_5 = float("nan")
        if ma5 and ma5 > 0:
            price_vs_ma_5 = float(last_close / ma5 - 1)

        volume_change_20d = float("nan")
        if len(volume) >= 21:
            base = float(volume.iloc[-21])
            last_vol = float(volume.iloc[-1])
            if base > 0 and last_vol > 0:
                volume_change_20d = float(np.log(last_vol / base))

        volume_shrink_20d = float("nan")
        if volume_change_20d == volume_change_20d:
            volume_shrink_20d = 1.0 if volume_change_20d < np.log(0.9) else 0.0

        vol_ratio_5 = float("nan")
        if len(volume) >= 5:
            avg_vol_5 = volume.rolling(5, min_periods=5).mean().iloc[-1]
            if avg_vol_5 and avg_vol_5 > 0:
                vol_ratio_5 = float(volume.iloc[-1] / avg_vol_5)

        hv20 = float("nan")
        if len(close) >= 20:
            hv20 = float(close.pct_change().rolling(20, min_periods=20).std().iloc[-1])

        range_5 = float("nan")
        if "high" in hist.columns and "low" in hist.columns and len(hist) >= 5:
            high = hist["high"].astype(float)
            low = hist["low"].astype(float)
            range_5 = float(((high - low) / close).rolling(5, min_periods=5).mean().iloc[-1])

        price_pos_20 = float("nan")
        if "high" in hist.columns and "low" in hist.columns and len(hist) >= 20:
            high20 = float(hist["high"].astype(float).rolling(20, min_periods=20).max().iloc[-1])
            low20 = float(hist["low"].astype(float).rolling(20, min_periods=20).min().iloc[-1])
            if high20 > low20:
                price_pos_20 = float((last_close - low20) / (high20 - low20))

        body = float("nan")
        upper_wick = float("nan")
        lower_wick = float("nan")
        lower_wick_ratio = float("nan")
        bullish_engulfing = 0.0
        open_last = float("nan")
        if "open" in hist.columns and "high" in hist.columns and "low" in hist.columns and len(hist) > 0:
            open_last = float(hist["open"].iloc[-1])
            high_last = float(hist["high"].iloc[-1])
            low_last = float(hist["low"].iloc[-1])
            if open_last > 0:
                body = (last_close - open_last) / open_last
                upper_wick = (high_last - max(open_last, last_close)) / open_last
                lower_wick = (min(open_last, last_close) - low_last) / open_last
                candle_span = max(high_last - low_last, 0.0)
                if candle_span > 0:
                    lower_wick_ratio = float((min(open_last, last_close) - low_last) / candle_span)
            if len(hist) >= 2:
                prev_open = float(hist["open"].iloc[-2])
                prev_close_c = float(hist["close"].iloc[-2])
                prev_body_down = prev_close_c < prev_open
                curr_body_up = last_close > open_last
                if prev_body_down and curr_body_up and open_last <= prev_close_c and last_close >= prev_open:
                    bullish_engulfing = 1.0

        rsi_rebound = 1.0 if (rsi_val == rsi_val and rsi_prev == rsi_prev and rsi_val > rsi_prev) else 0.0
        long_lower_wick = 1.0 if (lower_wick_ratio == lower_wick_ratio and lower_wick_ratio >= 0.3) else 0.0

        bull_volume = float("nan")
        bear_volume = float("nan")
        if vol_ratio == vol_ratio and open_last == open_last:
            bull_volume = 1.0 if (vol_ratio > 1.2 and last_close > open_last) else 0.0
            bear_volume = 1.0 if (vol_ratio > 1.5 and last_close < open_last) else 0.0

        # Use spot values when sensible; otherwise fall back to history.
        spot_close = float(r.get("close", float("nan")))
        spot_pct = float(r.get("pct_chg", float("nan")))
        spot_weibi = float(pd.to_numeric(r.get("weibi", float("nan")), errors="coerce"))
        price_source = "spot"
        pct_source = "spot"
        try:
            sym_num = float(symbol)
        except Exception:
            sym_num = None

        close_ok = spot_close > 0 and spot_close < 10000
        if sym_num is not None and abs(spot_close - sym_num) < 1e-6:
            close_ok = False

        def _price_looks_ok(px: float, ref: float, sym: str) -> bool:
            if not px or px <= 0:
                return False
            try:
                sym_num = float(sym)
                if abs(px - sym_num) < 1e-6:
                    return False
            except Exception:
                pass
            if ref > 0:
                ratio = px / ref
                if ratio < 0.5 or ratio > 1.5:
                    return False
            if px > 10000:
                return False
            return True

        sym_key = str(symbol).zfill(6)
        if not close_ok and realtime_fix:
            q_batch = fix_quotes.get(sym_key)
            if q_batch is not None:
                q_px, q_pct = q_batch
                if _price_looks_ok(q_px, last_close, symbol):
                    spot_close = float(q_px)
                    spot_pct = float(q_pct) if q_pct == q_pct else spot_pct
                    price_source = "spot_fix_batch"
                    pct_source = "spot_fix_batch"
                    with lock:
                        _inc("spot_fix_batch_used", symbol)

            if price_source == "spot" and realtime_fix_single_fallback:
                do_fix = False
                with lock:
                    if fix_budget["remaining"] > 0:
                        fix_budget["remaining"] -= 1
                        do_fix = True
                        _inc("spot_fix_single_try", symbol)
                if do_fix:
                    t_fix = time.perf_counter()
                    try:
                        q = get_realtime(
                            symbol,
                            provider=realtime_provider if realtime_provider else "auto",
                            use_proxy=use_proxy,
                            proxy=proxy,
                            allow_direct_fallback=realtime_allow_direct_fallback,
                        )
                        if _price_looks_ok(q.price, last_close, symbol):
                            spot_close = q.price
                            spot_pct = q.pct
                            price_source = "spot_fix"
                            pct_source = "spot_fix"
                            with lock:
                                _inc("spot_fix_single_hit", symbol)
                    except Exception:
                        with lock:
                            _inc("spot_fix_single_error", symbol)
                    finally:
                        rt_fix_ms += (time.perf_counter() - t_fix) * 1000.0

        if not close_ok and price_source == "spot":
            spot_close = last_close
            price_source = "hist"
            with lock:
                _inc("spot_close_fixed", symbol)

        pct_ok = abs(spot_pct) <= 40
        if not pct_ok or not (spot_pct == spot_pct):
            if price_source.startswith("spot") and spot_close > 0 and last_close > 0:
                # Use realtime price vs last close for intraday pct
                spot_pct = (spot_close / last_close - 1) * 100
                pct_source = "spot_calc"
            elif prev_close > 0:
                spot_pct = (last_close / prev_close - 1) * 100
                pct_source = "hist"
            else:
                spot_pct = 0.0
                pct_source = "hist"
            with lock:
                _inc("spot_pct_fixed", symbol)

        breakout_rt = float("nan")
        breakout_strength_rt = float("nan")
        breakout_days_since_rt = float("nan")
        breakout_confirm_rt = float("nan")
        if intraday_breakout and price_source.startswith("spot") and spot_close > 0 and breakout_level > 0:
            breakout_rt = 1.0 if spot_close >= breakout_level else 0.0
            breakout_strength_rt = float(spot_close / breakout_level - 1.0)
            if breakout_rt >= 1.0:
                breakout_days_since_rt = 0.0
                breakout_confirm_rt = max(breakout_confirm, 1.0)

        rs_raw = None
        if len(close) > rs_window:
            base = float(close.iloc[-1 - rs_window])
            if base > 0:
                rs_raw = (last_close / base - 1) * 100
        if rs_raw is None and prev_close > 0:
            rs_raw = (last_close / prev_close - 1) * 100
        if rs_raw is None:
            rs_raw = 0.0

        row = {
            "symbol": symbol,
            "name": name,
            "close": float(spot_close),
            "amount": float(r.get("amount", 0.0)),
            "mkt_cap": float(pd.to_numeric(r.get("mkt_cap", float("nan")), errors="coerce")),
            "float_mkt_cap": float(pd.to_numeric(r.get("float_mkt_cap", float("nan")), errors="coerce")),
            "pct_chg": float(spot_pct),
            "weibi": float(spot_weibi) if spot_weibi == spot_weibi else float("nan"),
            "breakout": breakout,
            "hhv_n": lookback_hhv,
            "hhv": float(hhv20),
            "breakout_level": float(breakout_level),
            "breakout_strength": float(breakout_strength),
            "breakout_confirm": float(breakout_confirm),
            "breakout_days_since": float(breakout_days_since),
            "days_since_hhv": days_since_hhv,
            "vol_ratio": vol_ratio,
            "vol_z20": vol_z20,
            "vol_ok": vol_ok,
            "vol_ratio_min_3": vol_ratio_min_3,
            "vol_ratio_min_5": vol_ratio_min_5,
            "ma5": float(ma5),
            "ma20": float(ma20),
            "ma60": float(ma60),
            "ma10": float(ma10),
            "ma30": float(ma30),
            "ma60_prev": ma60_prev,
            "ma5_slope": ma5_slope,
            "ma10_slope": ma10_slope,
            "ma20_slope": ma20_slope,
            "ma60_slope": ma60_slope,
            "ma_gap_5_20": ma_gap_5_20,
            "ma_gap_20_60": ma_gap_20_60,
            "price_ma20_dist": price_ma20_dist,
            "price_ma60_dist": price_ma60_dist,
            "price_pos_20": price_pos_20,
            "body": body,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "lower_wick_ratio": lower_wick_ratio,
            "long_lower_wick": long_lower_wick,
            "bullish_engulfing": bullish_engulfing,
            "ret_1": ret_1,
            "ret_3": ret_3,
            "ret_5": ret_5,
            "ret_10": ret_10,
            "ret_20": ret_20,
            "price_accel_5": price_accel_5,
            "price_accel_10": price_accel_10,
            "price_vs_high_20": price_vs_high_20,
            "price_vs_ma_5": price_vs_ma_5,
            "bull_volume": bull_volume,
            "bear_volume": bear_volume,
            "volume_change_20d": volume_change_20d,
            "volume_shrink_20d": volume_shrink_20d,
            "vol_ratio_5": vol_ratio_5,
            "hv20": hv20,
            "range_5": range_5,
            "trend": float(trend),
            "atr": float(atr14),
            "atr_pct": atr_pct,
            "mdd20": float(mdd20),
            "rsi": rsi_val,
            "rsi_prev": rsi_prev,
            "rsi_rebound": rsi_rebound,
            "rsi_6": rsi6,
            "rsi_24": rsi24,
            "macd_line": macd_line_v,
            "macd_signal": macd_signal_v,
            "macd_hist": macd_hist_v,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "rs_raw": float(rs_raw),
            "price_source": price_source,
            "pct_source": pct_source,
        }
        if row["float_mkt_cap"] > 0 and row["amount"] > 0:
            row["turnover_pct_est"] = float(row["amount"] / row["float_mkt_cap"] * 100.0)
        else:
            row["turnover_pct_est"] = float("nan")
        if intraday_breakout:
            row.update(
                {
                    "breakout_rt": float(breakout_rt),
                    "breakout_strength_rt": float(breakout_strength_rt),
                    "breakout_confirm_rt": float(breakout_confirm_rt),
                    "breakout_days_since_rt": float(breakout_days_since_rt),
                }
            )
        with lock:
            _inc("ok", symbol)
        _commit_timing()
        return row

    if use_parallel and max_workers > 1 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for row in ex.map(_compute_row, records):
                if row:
                    rows.append(row)
    else:
        for r in records:
            row = _compute_row(r)
            if row:
                rows.append(row)

    fac = pd.DataFrame(rows)
    if fac.empty:
        return fac

    rs_source = str(signals_cfg.get("rs_source", "ret")).lower()
    if rs_source in ("ret", "return", "ret_n", "ret20"):
        rs_vals = fac.get("rs_raw", fac["pct_chg"])
    else:
        rs_vals = fac["pct_chg"]

    fac["rs_proxy"] = rs_vals.fillna(0).rank(pct=True)  # 0~1

    return fac.sort_values(["breakout", "vol_ratio", "rs_proxy"], ascending=[False, False, False]).reset_index(drop=True)
