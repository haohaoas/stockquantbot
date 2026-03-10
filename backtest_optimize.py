from __future__ import annotations

import argparse
import calendar
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from features.indicators import sma, hhv, atr, max_drawdown, rsi, macd
from sqdata.akshare_fetcher import fetch_hist
from sqdata.universe import load_universe_symbols, filter_symbols_by_market, filter_symbols_by_board
from strategy.scorer import score_and_rank
from sqdata.sector_map import apply_sector_map
from strategy.decision import apply_short_term_decision


def load_config(path: str) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from e

    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def save_config(path: str, cfg: dict) -> None:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from e
    p = Path(path)
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_universe_info(path: str) -> tuple[list[str], Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return [], {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return load_universe_symbols(path), {}

    symbol_col = None
    name_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("symbol", "code", "ts_code", "ticker"):
            symbol_col = c
        if cl in ("name", "sec_name"):
            name_col = c
    if symbol_col is None:
        return load_universe_symbols(path), {}

    df[symbol_col] = df[symbol_col].astype(str).str.zfill(6)
    symbols = df[symbol_col].tolist()
    name_map: Dict[str, str] = {}
    if name_col is not None:
        for _, r in df.iterrows():
            s = str(r.get(symbol_col, "")).zfill(6)
            name_map[s] = str(r.get(name_col, ""))
    return sorted(set(symbols)), name_map


def load_trade_calendar(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    for c in df.columns:
        if "date" in c.lower():
            return [str(x) for x in df[c].dropna().astype(str).tolist()]
    return []


def pick_trade_days(trade_days: list[str], end_date: dt.date, years: int) -> list[str]:
    if not trade_days:
        return []
    start_date = end_date - dt.timedelta(days=365 * years)
    out = [d for d in trade_days if start_date.isoformat() <= d <= end_date.isoformat()]
    return out


def add_months(d: dt.date, months: int) -> dt.date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    day = min(d.day, last_day)
    return dt.date(y, m, day)


def build_rolling_windows(
    trade_days: list[str],
    train_months: int = 8,
    val_months: int = 4,
    step_months: int = 1,
) -> list[tuple[list[str], list[str]]]:
    if not trade_days:
        return []
    days = [dt.date.fromisoformat(d) for d in trade_days]
    start = days[0]
    end = days[-1]
    windows: list[tuple[list[str], list[str]]] = []

    cursor = start
    while True:
        train_start = cursor
        train_end = add_months(train_start, train_months) - dt.timedelta(days=1)
        val_start = train_end + dt.timedelta(days=1)
        val_end = add_months(val_start, val_months) - dt.timedelta(days=1)
        if val_end > end:
            break

        train_days = [d for d in trade_days if train_start.isoformat() <= d <= train_end.isoformat()]
        val_days = [d for d in trade_days if val_start.isoformat() <= d <= val_end.isoformat()]
        if train_days and val_days:
            windows.append((train_days, val_days))

        cursor = add_months(cursor, step_months)

    return windows


def compute_features_for_symbol(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    lookback_hhv = int(params.get("lookback_hhv", 20))
    ma_fast = int(params.get("ma_fast", 5))
    ma_mid = int(params.get("ma_mid", 20))
    ma_slow = int(params.get("ma_slow", 60))
    rs_window = int(params.get("rs_window", 20))
    rsi_window = int(params.get("rsi_window", 14))
    macd_fast = int(params.get("macd_fast", 12))
    macd_slow = int(params.get("macd_slow", 26))
    macd_signal = int(params.get("macd_signal", 9))

    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)
    if "amount" in out.columns:
        amount = pd.to_numeric(out["amount"], errors="coerce")
    else:
        amount = pd.Series([float("nan")] * len(out))
    if amount.isna().all() or float(amount.fillna(0).abs().sum()) <= 0.0:
        # Tencent volume is in hands; approximate amount by volume * close * 100.
        amount = volume * close * 100.0
    out["amount"] = amount

    out["ma5"] = sma(close, ma_fast)
    out["ma20"] = sma(close, ma_mid)
    out["ma60"] = sma(close, ma_slow)
    out["ma10"] = sma(close, 10)
    out["ma30"] = sma(close, 30)
    out["hhv"] = hhv(close, lookback_hhv)

    avg_vol_20 = volume.rolling(20, min_periods=20).mean()
    out["vol_ratio"] = volume / avg_vol_20

    atr14 = atr(out, 14)
    out["atr"] = atr14
    out["atr_pct"] = atr14 / close

    out["mdd20"] = close.rolling(20, min_periods=20).apply(
        lambda x: abs(max_drawdown(pd.Series(x))), raw=False
    )

    out["rsi"] = rsi(close, rsi_window)
    macd_line, macd_signal_line, macd_hist = macd(close, macd_fast, macd_slow, macd_signal)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal_line
    out["macd_hist"] = macd_hist

    out["ma5_slope"] = out["ma5"] / out["ma5"].shift(5) - 1
    out["ma20_slope"] = out["ma20"] / out["ma20"].shift(5) - 1
    out["ma10_slope"] = out["ma10"] / out["ma10"].shift(5) - 1
    out["ma60_slope"] = out["ma60"] / out["ma60"].shift(5) - 1
    out["price_ma20_dist"] = close / out["ma20"] - 1
    out["price_ma60_dist"] = close / out["ma60"] - 1
    out["ma_gap_5_20"] = out["ma5"] / out["ma20"] - 1
    out["ma_gap_20_60"] = out["ma20"] / out["ma60"] - 1
    out["ret_1"] = close / close.shift(1) - 1
    out["ret_3"] = close / close.shift(3) - 1
    out["ret_5"] = close / close.shift(5) - 1
    out["ret_10"] = close / close.shift(10) - 1
    out["ret_20"] = close / close.shift(20) - 1
    out["price_accel_5"] = out["ret_5"] - out["ret_10"]
    out["price_accel_10"] = out["ret_10"] - out["ret_20"]
    out["price_vs_high_20"] = close / out["hhv"]
    out["price_vs_ma_5"] = close / out["ma5"] - 1
    out["vol_ratio_5"] = volume / volume.rolling(5, min_periods=5).mean()
    volume_change_20d = np.log(volume / volume.shift(20))
    out["volume_change_20d"] = volume_change_20d
    out["volume_shrink_20d"] = (volume_change_20d < np.log(0.9)).astype(float)
    vol_mean_20 = volume.rolling(20, min_periods=20).mean()
    vol_std_20 = volume.rolling(20, min_periods=20).std()
    out["vol_z20"] = (volume - vol_mean_20) / vol_std_20
    vol_ratio_series = volume / volume.rolling(20, min_periods=20).mean()
    out["vol_ratio_min_3"] = vol_ratio_series.rolling(3, min_periods=3).min()
    out["vol_ratio_min_5"] = vol_ratio_series.rolling(5, min_periods=5).min()
    out["hv20"] = close.pct_change().rolling(20, min_periods=20).std()
    out["range_5"] = ((out["high"] - out["low"]) / close).rolling(5, min_periods=5).mean()
    high20 = out["high"].rolling(20, min_periods=20).max()
    low20 = out["low"].rolling(20, min_periods=20).min()
    out["price_pos_20"] = (close - low20) / (high20 - low20)
    body = (out["close"] - out["open"]) / out["open"]
    upper = (out["high"] - out[["open", "close"]].max(axis=1)) / out["open"]
    lower = (out[["open", "close"]].min(axis=1) - out["low"]) / out["open"]
    out["body"] = body
    out["upper_wick"] = upper
    out["lower_wick"] = lower
    out["bull_volume"] = ((out["vol_ratio"] > 1.2) & (out["close"] > out["open"])).astype(float)
    out["bear_volume"] = ((out["vol_ratio"] > 1.5) & (out["close"] < out["open"])).astype(float)
    out["rsi_6"] = rsi(close, 6)
    out["rsi_24"] = rsi(close, 24)

    out["rs_raw"] = (close / close.shift(rs_window) - 1) * 100
    out["pct_chg"] = (close / close.shift(1) - 1) * 100
    out["avg_amount_20"] = out["amount"].rolling(20, min_periods=20).mean()

    out["trend"] = np.where(
        (out["ma20"] > out["ma60"]) & (close > out["ma20"]),
        1.0,
        np.where(close > out["ma20"], 0.4, 0.0),
    )
    # Breakout details (align with live factor logic)
    hhv_prev = out["hhv"].shift(1)
    out["breakout_level"] = hhv_prev.where(hhv_prev.notna(), out["hhv"])
    out["breakout"] = (close >= out["breakout_level"]).astype(int)
    out["breakout_strength"] = (close / out["breakout_level"] - 1).where(out["breakout_level"] > 0, 0.0)
    # Days since last close above breakout level (within lookback window)
    def _days_since_last_true(window: pd.Series) -> float:
        if window.empty:
            return float("nan")
        if not window.any():
            return float("nan")
        last_true = int(window.to_numpy().nonzero()[0][-1])
        return float(len(window) - 1 - last_true)

    def _consecutive_true_from_end(window: pd.Series) -> float:
        if window.empty:
            return 0.0
        cnt = 0
        for v in window.iloc[::-1]:
            if bool(v):
                cnt += 1
            else:
                break
        return float(cnt)

    above = close >= out["breakout_level"]
    out["breakout_days_since"] = above.rolling(lookback_hhv, min_periods=1).apply(_days_since_last_true, raw=False)
    out["breakout_confirm"] = above.rolling(lookback_hhv, min_periods=1).apply(_consecutive_true_from_end, raw=False)

    # Days since highest close in lookback
    def _days_since_max(window: pd.Series) -> float:
        if window.empty:
            return float("nan")
        idx = int(window.to_numpy().argmax())
        return float(len(window) - 1 - idx)

    out["days_since_hhv"] = close.rolling(lookback_hhv, min_periods=1).apply(_days_since_max, raw=False)

    # Bollinger bands (20)
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std

    return out


@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    ret_pct: float
    reason: str


def _get_close_on(df: pd.DataFrame, date: str) -> float:
    if date in df.index:
        return float(df.loc[date].get("close", np.nan))
    idx = df.index.searchsorted(date) - 1
    if idx >= 0:
        return float(df.iloc[idx].get("close", np.nan))
    return float("nan")


def simulate_trade(
    symbol: str,
    signal_date: str,
    entry: float,
    stop: float,
    target: float,
    hist_map: dict,
    max_exit_date: str | None = None,
) -> Trade | None:
    if entry <= 0 or stop <= 0 or target <= 0 or stop >= entry:
        return None
    df = hist_map.get(symbol)
    if df is None or df.empty:
        return None
    if signal_date not in df.index:
        return None

    pos = df.index.get_loc(signal_date)
    if isinstance(pos, slice) or isinstance(pos, np.ndarray):
        return None
    if pos + 1 >= len(df):
        return None

    for i in range(pos + 1, len(df)):
        day = df.index[i]
        if max_exit_date and day > max_exit_date:
            break
        row = df.iloc[i]
        low = float(row.get("low", 0) or 0)
        high = float(row.get("high", 0) or 0)
        if low <= stop and high >= target:
            exit_price = stop
            exit_date = day
            ret = (exit_price - entry) / entry * 100
            return Trade(symbol, signal_date, exit_date, entry, exit_price, ret, "stop_first")
        if low <= stop:
            exit_price = stop
            exit_date = day
            ret = (exit_price - entry) / entry * 100
            return Trade(symbol, signal_date, exit_date, entry, exit_price, ret, "stop")
        if high >= target:
            exit_price = target
            exit_date = day
            ret = (exit_price - entry) / entry * 100
            return Trade(symbol, signal_date, exit_date, entry, exit_price, ret, "target")

    if max_exit_date:
        exit_price = _get_close_on(df, max_exit_date)
        if not (exit_price == exit_price):
            return None
        exit_date = max_exit_date
        ret = (exit_price - entry) / entry * 100
        return Trade(symbol, signal_date, exit_date, entry, exit_price, ret, "window_end")

    exit_price = float(df.iloc[-1].get("close", entry) or entry)
    exit_date = df.index[-1]
    ret = (exit_price - entry) / entry * 100
    return Trade(symbol, signal_date, exit_date, entry, exit_price, ret, "end")


def grid_params(cfg: dict) -> list[dict]:
    decision_cfg = cfg.get("decision", {}) or {}
    base_min_score = float(decision_cfg.get("min_score", 20))
    base_stop = float(decision_cfg.get("stop_atr_mult", 2.0))
    base_rr = float(decision_cfg.get("target_rr", 2.0))

    min_score_grid = sorted({max(0.0, base_min_score - 10), base_min_score, base_min_score + 10})
    stop_grid = sorted({max(0.5, base_stop - 0.5), base_stop, base_stop + 0.5})
    rr_grid = sorted({max(0.5, base_rr - 0.5), base_rr, base_rr + 0.5})

    grid = []
    for min_score in min_score_grid:
        for stop_atr_mult in stop_grid:
            for target_rr in rr_grid:
                for require_breakout in (True, False):
                    for require_vol_ok in (True, False):
                        g = dict(decision_cfg)
                        g["min_score"] = min_score
                        g["stop_atr_mult"] = stop_atr_mult
                        g["target_rr"] = target_rr
                        g["require_breakout"] = require_breakout
                        g["require_vol_ok"] = require_vol_ok
                        grid.append(g)
    return grid


def compute_trades_for_days(
    days: list[str],
    ranked_by_date: dict[str, pd.DataFrame],
    hist_map: dict[str, pd.DataFrame],
    decision_cfg: dict,
    exclude_limit_up: bool,
    limit_up_pct: float,
) -> list[Trade]:
    trades: list[Trade] = []
    last_exit: Dict[str, str] = {}

    if not days:
        return trades

    window_end = days[-1]

    for day in days:
        ranked = ranked_by_date.get(day)
        if ranked is None or ranked.empty:
            continue
        decided = apply_short_term_decision(ranked, decision_cfg)
        if exclude_limit_up and "pct_chg" in decided.columns:
            decided = decided[pd.to_numeric(decided["pct_chg"], errors="coerce") < limit_up_pct]

        buys = decided[decided["action"] == "BUY"]
        if buys.empty:
            continue

        for _, r in buys.iterrows():
            sym = str(r.get("symbol", ""))
            if not sym:
                continue
            last_exit_date = last_exit.get(sym)
            if last_exit_date and day <= last_exit_date:
                continue

            entry = float(r.get("entry", 0) or 0)
            stop = float(r.get("stop", 0) or 0)
            target = float(r.get("target", 0) or 0)
            tr = simulate_trade(sym, day, entry, stop, target, hist_map, max_exit_date=window_end)
            if tr is None:
                continue
            trades.append(tr)
            last_exit[sym] = tr.exit_date

    return trades


def compute_equity_curve(trades: list[Trade], days: list[str], hist_map: dict[str, pd.DataFrame]) -> pd.Series:
    if not days:
        return pd.Series(dtype=float)

    equity = 1.0
    equity_series = []

    for day in days:
        open_trades = [t for t in trades if t.entry_date <= day <= t.exit_date]
        if open_trades:
            values = []
            for t in open_trades:
                df = hist_map.get(t.symbol)
                if df is None or df.empty:
                    continue
                if day == t.exit_date:
                    px = t.exit_price
                else:
                    px = _get_close_on(df, day)
                if not (px == px) or px <= 0:
                    continue
                values.append(px / t.entry_price)
            if values:
                equity = float(np.mean(values))
        equity_series.append(equity)

    return pd.Series(equity_series, index=days)


def compute_metrics(trades: list[Trade], days: list[str], hist_map: dict[str, pd.DataFrame]) -> dict:
    if not days:
        return {"sharpe": 0.0, "avg_ret": 0.0, "win_rate": 0.0, "trades": 0, "max_dd": 0.0}

    equity = compute_equity_curve(trades, days, hist_map)
    if equity.empty:
        return {"sharpe": 0.0, "avg_ret": 0.0, "win_rate": 0.0, "trades": 0, "max_dd": 0.0}

    daily_ret = equity.pct_change().fillna(0.0)
    mean = float(daily_ret.mean())
    std = float(daily_ret.std(ddof=1)) if len(daily_ret) > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(252)) if std > 1e-9 else 0.0

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(abs(dd.min())) if len(dd) else 0.0

    ret_pct = float(equity.iloc[-1] - 1)

    rets = np.array([t.ret_pct for t in trades], dtype=float) if trades else np.array([])
    win_rate = float((rets > 0).mean()) if len(rets) else 0.0

    return {
        "sharpe": sharpe,
        "avg_ret": ret_pct,
        "win_rate": win_rate,
        "trades": len(trades),
        "max_dd": max_dd,
    }


def pareto_front(df: pd.DataFrame, maximize: list[str], minimize: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    flags = []
    values = df.to_dict("records")
    for i, row in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            better_or_equal = all(other[c] >= row[c] for c in maximize) and all(other[c] <= row[c] for c in minimize)
            strictly_better = any(other[c] > row[c] for c in maximize) or any(other[c] < row[c] for c in minimize)
            if better_or_equal and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-objective backtest optimizer for StockQuantBot")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--train-months", type=int, default=8)
    parser.add_argument("--val-months", type=int, default=4)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--out", default="output/backtest_optimize.csv")
    parser.add_argument("--max-dd", type=float, default=0.15)
    parser.add_argument("--min-win-rate", type=float, default=0.45)
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--apply-best", action="store_true", help="Apply best params to config decision section")
    args = parser.parse_args()

    cfg = load_config(args.config)
    universe_file = str(cfg.get("universe_file", "./data/universe.csv"))
    trade_calendar_file = str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv"))
    universe_cfg = cfg.get("universe", {}) or {}
    sector_cfg = cfg.get("sector_boost", {}) or {}
    signals_cfg = cfg.get("signals", {}) or {}
    allow_eastmoney_fallback = bool(cfg.get("allow_eastmoney_fallback", True))
    weights = cfg.get("score_weights", {}) or {}
    output_cfg = cfg.get("output", {}) or {}

    market_scope = cfg.get("market_scope", ["sh", "sz"])
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = bool(cfg.get("exclude_chi_next", False))
    mainboard_only = bool(cfg.get("mainboard_only", True))

    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")

    preselect_n = int(signals_cfg.get("preselect_n", 300))
    vol_ratio_min = float(signals_cfg.get("vol_ratio_min", 1.5))
    rs_source = str(signals_cfg.get("rs_source", "ret")).lower()
    top_n = int(output_cfg.get("top_n", 20))


    exclude_limit_up = bool(cfg.get("exclude_limit_up", True))
    limit_up_pct = float(cfg.get("limit_up_pct", 9.8))
    index_cfg = cfg.get("index", {}) or {}
    index_symbol = str(index_cfg.get("symbol", "000300"))
    index_ma_window = int(index_cfg.get("ma_window", 200))
    regime_filter = bool(index_cfg.get("regime_filter", True))

    symbols, name_map = load_universe_info(universe_file)
    symbols = filter_symbols_by_market(symbols, market_scope)
    symbols = filter_symbols_by_board(
        symbols,
        exclude_star=exclude_star,
        exclude_chi_next=exclude_chi_next,
        mainboard_only=mainboard_only,
    )
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    trade_days = load_trade_calendar(trade_calendar_file)
    today = dt.date.today()
    trade_days = [d for d in trade_days if d <= today.isoformat()]
    if not trade_days:
        raise RuntimeError("trade_calendar is empty.")
    end_date = dt.date.fromisoformat(trade_days[-1])
    trade_days = pick_trade_days(trade_days, end_date, args.years)

    if not trade_days:
        raise RuntimeError("no trade days selected for backtest range")

    start_date = trade_days[0]
    end_date_str = trade_days[-1]

    windows = build_rolling_windows(trade_days, args.train_months, args.val_months, args.step_months)
    if not windows:
        raise RuntimeError("rolling windows empty; please check trade calendar or params")

    # Fetch history for each symbol and compute features.
    panel_rows = []
    hist_map: Dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols):
        try:
            hist = fetch_hist(
                sym,
                start=start_date,
                end=end_date_str,
                use_proxy=use_proxy,
                proxy=proxy,
                use_cache=bool(signals_cfg.get("hist_cache", True)),
                cache_dir=str(signals_cfg.get("hist_cache_dir", "./cache/hist")),
                cache_ttl_sec=int(signals_cfg.get("hist_cache_ttl_sec", 6 * 3600)),
                allow_akshare_fallback=allow_eastmoney_fallback,
            )
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        hist = hist.copy()
        hist["symbol"] = sym
        hist["name"] = name_map.get(sym, "")
        hist = compute_features_for_symbol(hist, signals_cfg)
        panel_rows.append(hist)

        h = hist[["date", "open", "high", "low", "close"]].dropna(subset=["date"]).copy()
        h = h.set_index("date")
        hist_map[sym] = h

        if (i + 1) % 200 == 0:
            print(f"[info] loaded {i+1}/{len(symbols)} symbols")

    if not panel_rows:
        raise RuntimeError("no historical data loaded")

    panel = pd.concat(panel_rows, ignore_index=True)

    # Build index regime map
    regime_map: dict[str, bool] = {}
    if regime_filter:
        try:
            idx_hist = fetch_hist(
                index_symbol,
                start=start_date,
                end=end_date_str,
                use_proxy=use_proxy,
                proxy=proxy,
                use_cache=bool(signals_cfg.get("hist_cache", True)),
                cache_dir=str(signals_cfg.get("hist_cache_dir", "./cache/hist")),
                cache_ttl_sec=int(signals_cfg.get("hist_cache_ttl_sec", 6 * 3600)),
                allow_akshare_fallback=allow_eastmoney_fallback,
            )
        except Exception:
            idx_hist = pd.DataFrame()
        if idx_hist is None or idx_hist.empty:
            regime_filter = False
        else:
            idx_hist = idx_hist.copy()
            idx_hist["close"] = pd.to_numeric(idx_hist["close"], errors="coerce")
            idx_hist["ma"] = idx_hist["close"].rolling(index_ma_window, min_periods=index_ma_window).mean()
            for _, r in idx_hist.iterrows():
                date = str(r.get("date", ""))
                close = float(r.get("close", 0) or 0)
                ma = float(r.get("ma", 0) or 0)
                regime_map[date] = bool(close >= ma and ma > 0)

    ranked_by_date: Dict[str, pd.DataFrame] = {}

    min_price = float(universe_cfg.get("min_price", 5.0))
    max_price = float(universe_cfg.get("max_price", 0.0))
    min_avg_amount_20 = float(universe_cfg.get("min_avg_amount_20", 1e8))
    exclude_st = bool(universe_cfg.get("exclude_st", True))
    try:
        use_amount_filter = float(panel.get("avg_amount_20", pd.Series([0.0])).fillna(0).max()) > 0
    except Exception:
        use_amount_filter = False

    for day in trade_days:
        if regime_filter and not regime_map.get(day, False):
            continue
        daily = panel[panel["date"] == day].copy()
        if daily.empty:
            continue

        daily = daily[daily["close"] >= min_price]
        if max_price > 0:
            daily = daily[daily["close"] <= max_price]
        if min_avg_amount_20 > 0 and use_amount_filter and "avg_amount_20" in daily.columns:
            daily = daily[daily["avg_amount_20"] >= min_avg_amount_20]
        if exclude_st and "name" in daily.columns:
            daily = daily[~daily["name"].astype(str).str.contains("ST|\\*ST|退", regex=True)]

        if daily.empty:
            continue

        if "amount" in daily.columns:
            daily = daily.sort_values("amount", ascending=False).head(preselect_n)
        else:
            daily = daily.head(preselect_n)

        daily["vol_ok"] = (daily["vol_ratio"] >= vol_ratio_min).astype(int)

        if rs_source in ("ret", "return", "ret_n", "ret20"):
            rs_vals = daily["rs_raw"]
        else:
            rs_vals = daily["pct_chg"]
        daily["rs_proxy"] = rs_vals.fillna(0).rank(pct=True)

        if sector_cfg.get("enabled"):
            map_file = str(sector_cfg.get("map_file", "./data/sector_map.csv"))
            daily = apply_sector_map(daily, map_file)
        ranked = score_and_rank(daily, weights, top_n=top_n, sector_cfg=sector_cfg)
        ranked_by_date[day] = ranked

    grid = grid_params(cfg)
    print(f"[info] grid size: {len(grid)} | windows: {len(windows)}")

    results = []
    for idx, param in enumerate(grid):
        val_sharpe_sum = 0.0
        val_count = 0
        val_max_dd = 0.0
        val_trades = 0
        val_wins = 0
        val_avg_ret_sum = 0.0

        train_sharpe_sum = 0.0
        train_count = 0

        for train_days, val_days in windows:
            train_trades = compute_trades_for_days(
                train_days, ranked_by_date, hist_map, param, exclude_limit_up, limit_up_pct
            )
            train_metrics = compute_metrics(train_trades, train_days, hist_map)
            train_sharpe_sum += train_metrics["sharpe"]
            train_count += 1

            val_trades_list = compute_trades_for_days(
                val_days, ranked_by_date, hist_map, param, exclude_limit_up, limit_up_pct
            )
            val_metrics = compute_metrics(val_trades_list, val_days, hist_map)
            val_sharpe_sum += val_metrics["sharpe"]
            val_avg_ret_sum += val_metrics["avg_ret"]
            val_count += 1
            val_max_dd = max(val_max_dd, val_metrics["max_dd"])
            val_trades += int(val_metrics["trades"])
            val_wins += int(sum(1 for t in val_trades_list if t.ret_pct > 0))

        val_sharpe = val_sharpe_sum / max(val_count, 1)
        val_avg_ret = val_avg_ret_sum / max(val_count, 1)
        win_rate = val_wins / val_trades if val_trades > 0 else 0.0

        constraints_ok = (
            val_max_dd < args.max_dd
            and win_rate > args.min_win_rate
            and val_trades > args.min_trades
        )

        row = {
            "min_score": param.get("min_score"),
            "stop_atr_mult": param.get("stop_atr_mult"),
            "target_rr": param.get("target_rr"),
            "require_breakout": param.get("require_breakout"),
            "require_vol_ok": param.get("require_vol_ok"),
            "val_sharpe": val_sharpe,
            "val_avg_ret": val_avg_ret,
            "val_max_dd": val_max_dd,
            "val_win_rate": win_rate,
            "val_trades": val_trades,
            "train_sharpe": train_sharpe_sum / max(train_count, 1),
            "constraints_ok": constraints_ok,
        }
        results.append(row)

        if (idx + 1) % 10 == 0:
            print(f"[info] tested {idx+1}/{len(grid)}")

    out_df = pd.DataFrame(results)
    if out_df.empty:
        raise RuntimeError("no results")

    ok_df = out_df[out_df["constraints_ok"]].copy()
    if ok_df.empty:
        pareto_base = out_df.copy()
    else:
        pareto_base = ok_df

    pareto_flags = pareto_front(
        pareto_base,
        maximize=["val_sharpe", "val_win_rate", "val_trades"],
        minimize=["val_max_dd"],
    )
    pareto_base["pareto"] = pareto_flags

    out_df = out_df.sort_values(["val_sharpe", "val_avg_ret"], ascending=[False, False])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    summary_path = out_path.with_suffix(".md")
    lines = []
    lines.append(f"# 多目标参数优化结果（最近{args.years}年，滚动训练/验证）\n")
    lines.append(f"范围: {start_date} ~ {end_date_str}\n")
    lines.append("训练/验证: 8个月训练 + 4个月验证，步长=1个月\n")
    lines.append("\n约束条件：\n")
    lines.append(f"- 最大回撤 < {args.max_dd:.2f}\n")
    lines.append(f"- 胜率 > {args.min_win_rate:.2f}\n")
    lines.append(f"- 交易次数 > {args.min_trades}\n")
    lines.append("\n假设：\n")
    lines.append("- 信号在当日收盘生成，入场价=当日收盘价。\n")
    lines.append("- 止损/止盈使用后续交易日的日内高低价近似；同日同时触发时先止损（保守）。\n")
    lines.append("- 同一标的未平仓时忽略新的 BUY 信号。\n")
    lines.append("- 组合净值按等权持仓近似：持仓平均值；无持仓时延续上日净值。\n")
    if not use_amount_filter and min_avg_amount_20 > 0:
        lines.append("- 成交额过滤已自动跳过（历史 amount 为 0）。\n")

    top = out_df.head(10)
    lines.append("\nTop 10（按验证集夏普）：\n")
    try:
        lines.append(top.to_markdown(index=False))
    except Exception:
        lines.append(top.to_string(index=False))

    pareto_df = pareto_base[pareto_base.get("pareto", False)].copy()
    lines.append("\n\nPareto 前沿解：\n")
    if pareto_df.empty:
        lines.append("无（约束条件下没有可行解）。\n")
    else:
        try:
            lines.append(pareto_df.to_markdown(index=False))
        except Exception:
            lines.append(pareto_df.to_string(index=False))

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[ok] saved: {out_path}")
    print(f"[ok] saved: {summary_path}")

    if args.apply_best:
        candidates = out_df[out_df["constraints_ok"]].copy()
        if candidates.empty:
            candidates = out_df
            print("[warn] no params satisfy constraints; using best overall by val_sharpe")
        best = candidates.sort_values(["val_sharpe", "val_avg_ret"], ascending=[False, False]).head(1)
        if best.empty:
            print("[warn] no best params found")
            return
        best_row = best.iloc[0].to_dict()
        decision_cfg = cfg.get("decision", {}) or {}
        decision_cfg["min_score"] = float(best_row.get("min_score", decision_cfg.get("min_score", 20)))
        decision_cfg["stop_atr_mult"] = float(best_row.get("stop_atr_mult", decision_cfg.get("stop_atr_mult", 2.0)))
        decision_cfg["target_rr"] = float(best_row.get("target_rr", decision_cfg.get("target_rr", 2.0)))
        decision_cfg["require_breakout"] = bool(best_row.get("require_breakout", decision_cfg.get("require_breakout", True)))
        decision_cfg["require_vol_ok"] = bool(best_row.get("require_vol_ok", decision_cfg.get("require_vol_ok", True)))
        cfg["decision"] = decision_cfg
        save_config(args.config, cfg)
        print(f"[ok] applied best params to {args.config}")


if __name__ == "__main__":
    main()
