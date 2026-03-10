from __future__ import annotations

import argparse
import calendar
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from backtest_optimize import (
    load_config,
    load_trade_calendar,
    pick_trade_days,
    compute_features_for_symbol,
)
from features.feature_factory import FeatureFactory
from ml.ref_model import load_ref_model
from sqdata.sector_map import apply_sector_map
from sqdata.universe import filter_symbols_by_market, filter_symbols_by_board
from strategy.scorer import score_and_rank
from strategy.decision import apply_short_term_decision


def _add_months(d: dt.date, months: int) -> dt.date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    day = min(d.day, last_day)
    return dt.date(y, m, day)


def _subtract_months(d: dt.date, months: int) -> dt.date:
    return _add_months(d, -months)


def _list_symbols(manual_dir: Path) -> list[str]:
    out = []
    for p in manual_dir.glob("*.csv"):
        name = p.stem.strip()
        if name.isdigit() and len(name) == 6:
            out.append(name)
    return sorted(set(out))


def _load_hist(manual_dir: Path, symbol: str) -> pd.DataFrame:
    p = manual_dir / f"{symbol}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = df["date"].astype(str)
    return df.sort_values("date").reset_index(drop=True)


def _build_market_feature_table(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    m = market_df.copy()
    if "date" not in m.columns:
        return pd.DataFrame()
    m = m.sort_values("date").reset_index(drop=True)
    m_close = pd.to_numeric(m["close"], errors="coerce")
    m_high = pd.to_numeric(m.get("high"), errors="coerce")
    m_low = pd.to_numeric(m.get("low"), errors="coerce")

    m["market_ret_5d"] = m_close.pct_change(5)
    m["market_ret_20d"] = m_close.pct_change(20)
    m["market_ma50"] = m_close.rolling(50, min_periods=50).mean()
    m["market_ma200"] = m_close.rolling(200, min_periods=200).mean()
    m["market_trend_50"] = m_close / m["market_ma50"] - 1
    m["market_trend_200"] = m_close / m["market_ma200"] - 1
    m["market_regime"] = (m_close >= m["market_ma200"]).astype(float)
    m["market_vol_20"] = m_close.pct_change().rolling(20, min_periods=20).std()

    if m_high.notna().any() and m_low.notna().any():
        high_low = m_high - m_low
        high_close = (m_high - m_close.shift()).abs()
        low_close = (m_low - m_close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        m["market_atr_14"] = tr.rolling(14, min_periods=14).mean()
        m["market_atr_pct"] = m["market_atr_14"] / m_close

    cols = [
        "date",
        "market_ret_5d",
        "market_ret_20d",
        "market_trend_50",
        "market_trend_200",
        "market_regime",
        "market_vol_20",
        "market_atr_pct",
    ]
    return m[cols].copy()


def _compute_model_scores(
    hist: pd.DataFrame,
    market_feat: pd.DataFrame,
    factory: FeatureFactory,
    model: dict,
) -> pd.DataFrame:
    feats = factory.create_all_features(hist, market_df=None)
    if feats.empty:
        return pd.DataFrame()
    base = hist[["date", "close"]].copy()
    df = pd.concat([base, feats], axis=1)
    if market_feat is not None and not market_feat.empty:
        df = df.merge(market_feat, on="date", how="left")
        close = pd.to_numeric(df["close"], errors="coerce")
        df["rel_strength_5d"] = close.pct_change(5) - df.get("market_ret_5d")
        df["rel_strength_20d"] = close.pct_change(20) - df.get("market_ret_20d")

    features = model.get("features") or []
    for f in features:
        if f not in df.columns:
            df[f] = np.nan
    X = df[features].apply(pd.to_numeric, errors="coerce")
    impute = model.get("impute", {}) or {}
    for f in features:
        X[f] = X[f].fillna(float(impute.get(f, 0.0)))
    booster = model.get("_model")
    preds = booster.predict(X.to_numpy(dtype=float))
    out = df[["date"]].copy()
    out["model_score"] = preds
    return out


def _max_drawdown(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    return float(dd.min())


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest rule+model BUY intersection")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--months", type=int, default=4)
    parser.add_argument("--start", default="", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="", help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--manual-dir", default="./data/manual_hist")
    parser.add_argument("--index-symbol", default="000300")
    parser.add_argument("--model-min", type=float, default=-1)
    parser.add_argument("--combo-rule", type=float, default=0.6)
    parser.add_argument("--combo-model", type=float, default=0.4)
    parser.add_argument("--exclude-chi-next", action="store_true")
    parser.add_argument("--exclude-star", action="store_true")
    parser.add_argument("--mainboard-only", action="store_true")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--out", default="./output/backtest_rule_model_4m.csv")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--limit-up", type=float, default=9.8)
    parser.add_argument("--hold-days", type=int, default=0)
    parser.add_argument("--exit-open", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    signals_cfg = cfg.get("signals", {}) or {}
    universe_cfg = cfg.get("universe", {}) or {}
    decision_cfg = cfg.get("decision", {}) or {}
    weights = cfg.get("score_weights", {}) or {}
    sector_cfg = cfg.get("sector_boost", {}) or {}

    configured_model = (signals_cfg.get("model_ref", {}) or {}).get("path", "")
    model_path = str(args.model or configured_model or "").strip()
    if not model_path:
        model_path = "./models/lightgbm_fd2_excess_y2_latest.json"

    model = None
    if not args.rules_only:
        model = load_ref_model(model_path)
        if not model:
            raise RuntimeError(f"model not found: {model_path}")

    manual_dir = Path(args.manual_dir)
    if not manual_dir.exists():
        raise RuntimeError(f"manual_hist dir not found: {manual_dir}")

    symbols = _list_symbols(manual_dir)
    symbols = filter_symbols_by_market(symbols, cfg.get("market_scope", ["sh", "sz"]))
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = bool(cfg.get("exclude_chi_next", False))
    mainboard_only = bool(cfg.get("mainboard_only", True))
    if args.exclude_star:
        exclude_star = True
    if args.exclude_chi_next:
        exclude_chi_next = True
    if args.mainboard_only:
        mainboard_only = True
    symbols = filter_symbols_by_board(
        symbols,
        exclude_star=exclude_star,
        exclude_chi_next=exclude_chi_next,
        mainboard_only=mainboard_only,
    )
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    index_hist = _load_hist(manual_dir, args.index_symbol)
    if index_hist.empty:
        raise RuntimeError("index history missing for market features")
    market_feat = _build_market_feature_table(index_hist)

    trade_calendar_file = str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv"))
    trade_days = load_trade_calendar(trade_calendar_file)
    last_idx_day = dt.date.fromisoformat(str(index_hist["date"].iloc[-1]))
    end_dt = dt.date.fromisoformat(str(args.end)) if str(args.end).strip() else last_idx_day
    if str(args.start).strip():
        start_dt = dt.date.fromisoformat(str(args.start))
    else:
        start_dt = _subtract_months(end_dt, args.months)
    trade_days = [d for d in trade_days if start_dt.isoformat() <= d <= end_dt.isoformat()]
    if not trade_days:
        raise RuntimeError("no trade days in range")

    # Build panel with rule features + model scores
    panel_rows = []
    if model is None:
        feature_set = str((cfg.get("features") or {}).get("set") or "legacy").strip().lower()
    else:
        feature_set = str((model.get("meta") or {}).get("feature_set") or "legacy").strip().lower()
    factory = FeatureFactory(feature_set=feature_set)
    if args.rules_only:
        forward_days = 2
    else:
        forward_days = int((model.get("meta") or {}).get("forward_days", 2))
    hold_days = int(args.hold_days) if int(args.hold_days) > 0 else int(forward_days)
    exit_open = bool(args.exit_open)
    exit_shift = hold_days + 1 if exit_open else hold_days

    for i, sym in enumerate(symbols, 1):
        hist = _load_hist(manual_dir, sym)
        if hist.empty:
            continue
        # buffer for indicators
        hist = hist[hist["date"].between((start_dt - dt.timedelta(days=220)).isoformat(), end_dt.isoformat())].copy()
        if hist.empty:
            continue
        rule_feat = compute_features_for_symbol(hist, signals_cfg)
        if rule_feat.empty:
            continue
        rule_feat = rule_feat.copy()
        rule_feat["symbol"] = sym
        if "name" not in rule_feat.columns:
            rule_feat["name"] = ""
        if not args.rules_only:
            model_scores = _compute_model_scores(hist, market_feat, factory, model)
            if not model_scores.empty:
                rule_feat = rule_feat.merge(model_scores, on="date", how="left")
        rule_feat["fwd_ret"] = rule_feat["close"].shift(-exit_shift) / rule_feat["close"] - 1
        rule_feat = rule_feat.dropna(subset=["fwd_ret"])
        panel_rows.append(rule_feat)
        if i % 200 == 0:
            print(f"[info] loaded {i}/{len(symbols)} symbols")

    if not panel_rows:
        raise RuntimeError("no data loaded")

    panel = pd.concat(panel_rows, ignore_index=True)
    panel = panel[panel["date"].between(start_dt.isoformat(), end_dt.isoformat())].copy()
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["entry_open"] = panel.groupby("symbol")["open"].shift(-1)
    panel["entry_date"] = panel.groupby("symbol")["date"].shift(-1)
    if exit_open:
        panel["exit_open"] = panel.groupby("symbol")["open"].shift(-exit_shift)
    else:
        panel["exit_close"] = panel.groupby("symbol")["close"].shift(-exit_shift)
    panel["exit_date"] = panel.groupby("symbol")["date"].shift(-exit_shift)

    model_ref_cfg = signals_cfg.get("model_ref") or {}
    model_min = args.model_min if args.model_min >= 0 else float(model_ref_cfg.get("show_min", 0.7))

    preselect_n = int(signals_cfg.get("preselect_n", 300))
    min_price = float(universe_cfg.get("min_price", 5.0))
    max_price = float(universe_cfg.get("max_price", 0.0))
    min_avg_amount_20 = float(universe_cfg.get("min_avg_amount_20", 1e8))
    exclude_st = bool(universe_cfg.get("exclude_st", True))

    portfolio = args.capital
    equity_curve = []
    trade_log = []
    step = max(exit_shift, 1)
    fee = float(args.fee)
    slippage = float(args.slippage)
    limit_up_pct = float(args.limit_up)

    idx = 0
    while idx < len(trade_days) - exit_shift:
        day = trade_days[idx]
        entry_day = trade_days[idx + 1] if idx + 1 < len(trade_days) else None
        exit_day = trade_days[idx + exit_shift] if idx + exit_shift < len(trade_days) else None
        daily = panel[panel["date"] == day].copy()
        if daily.empty:
            idx += step
            continue
        if entry_day and exit_day:
            daily = daily[
                (daily["entry_date"] == entry_day)
                & (daily["exit_date"] == exit_day)
                & (pd.to_numeric(daily["entry_open"], errors="coerce") > 0)
                & (
                    pd.to_numeric(
                        daily["exit_open"] if exit_open else daily["exit_close"],
                        errors="coerce",
                    )
                    > 0
                )
            ].copy()
            if daily.empty:
                idx += step
                continue

        daily = daily[daily["close"] >= min_price]
        if max_price > 0:
            daily = daily[daily["close"] <= max_price]
        if min_avg_amount_20 > 0 and "avg_amount_20" in daily.columns:
            daily = daily[daily["avg_amount_20"] >= min_avg_amount_20]
        if exclude_st and "name" in daily.columns:
            daily = daily[~daily["name"].astype(str).str.contains("ST|\\*ST|退", regex=True)]

        if daily.empty:
            idx += step
            continue

        if "amount" in daily.columns:
            daily = daily.sort_values("amount", ascending=False).head(preselect_n)
        else:
            daily = daily.head(preselect_n)

        vol_ratio_min = float(signals_cfg.get("vol_ratio_min", 1.5))
        daily["vol_ok"] = (daily["vol_ratio"] >= vol_ratio_min).astype(int)
        rs_source = str(signals_cfg.get("rs_source", "ret"))
        rs_vals = daily["rs_raw"] if rs_source in ("ret", "return", "ret_n", "ret20") else daily["pct_chg"]
        daily["rs_proxy"] = rs_vals.fillna(0).rank(pct=True)

        if sector_cfg.get("enabled"):
            map_file = str(sector_cfg.get("map_file", "./data/sector_map.csv"))
            daily = apply_sector_map(daily, map_file)

        ranked = score_and_rank(daily, weights, top_n=preselect_n, sector_cfg=sector_cfg)
        ranked = apply_short_term_decision(ranked, decision_cfg)

        ranked = ranked[ranked["action"] == "BUY"]

        if ranked.empty:
            idx += step
            continue

        if args.rules_only:
            ranked = ranked.sort_values("score", ascending=False).head(args.topk)
        else:
            if "model_score" in ranked.columns:
                ranked = ranked[ranked["model_score"] >= model_min]
            if ranked.empty:
                idx += step
                continue
            combo = ranked["score"] * args.combo_rule + ranked["model_score"] * 100.0 * args.combo_model
            ranked = ranked.assign(combo=combo).sort_values("combo", ascending=False).head(args.topk)

        exit_col = "exit_open" if exit_open else "exit_close"
        prev_close = pd.to_numeric(ranked["close"], errors="coerce")
        entry_open = pd.to_numeric(ranked["entry_open"], errors="coerce")
        exit_price = pd.to_numeric(ranked[exit_col], errors="coerce")
        limit_up = prev_close * (1 + limit_up_pct / 100.0)
        tradable = entry_open < limit_up
        ranked = ranked[tradable].copy()
        if ranked.empty:
            idx += step
            continue
        entry_open = pd.to_numeric(ranked["entry_open"], errors="coerce")
        exit_price = pd.to_numeric(ranked[exit_col], errors="coerce")
        gross_ret = exit_price / entry_open - 1.0
        cost_in = fee + slippage
        cost_out = fee + slippage
        net_ret = (1 + gross_ret) * (1 - cost_out) / (1 + cost_in) - 1.0
        avg_ret = float(net_ret.mean())
        portfolio *= (1 + avg_ret)
        equity_curve.append(portfolio)
        trade_log.append({"date": day, "avg_ret": avg_ret, "count": len(ranked), "portfolio": portfolio})

        idx += step

    if not trade_log:
        raise RuntimeError("no trades executed")

    total_return = portfolio / args.capital - 1
    wins = sum(1 for t in trade_log if t["avg_ret"] > 0)
    win_rate = wins / len(trade_log)
    mdd = _max_drawdown([t["portfolio"] for t in trade_log])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trade_log).to_csv(out_path, index=False)

    print(f"[ok] range: {start_dt.isoformat()} ~ {end_dt.isoformat()}")
    exit_label = "open" if exit_open else "close"
    print(f"[ok] trades: {len(trade_log)} | topk: {args.topk} | hold_days: {hold_days} | exit: {exit_label}")
    print(f"[ok] capital: {args.capital:.2f} -> {portfolio:.2f}")
    print(f"[ok] total_return: {total_return*100:.2f}% | win_rate: {win_rate*100:.2f}% | mdd: {mdd*100:.2f}%")
    if not args.rules_only:
        print(f"[ok] model: {model_path}")
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
