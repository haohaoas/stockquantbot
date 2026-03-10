from __future__ import annotations

import argparse
import calendar
import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from backtest_optimize import load_config
from features.feature_factory import FeatureFactory
from ml.ref_model import load_ref_model


def _add_months(d: dt.date, months: int) -> dt.date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    day = min(d.day, last_day)
    return dt.date(y, m, day)


def _subtract_months(d: dt.date, months: int) -> dt.date:
    return _add_months(d, -months)


def _load_index_history(manual_dir: Path, index_symbol: str) -> pd.DataFrame:
    p = manual_dir / f"{index_symbol}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
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
    df = pd.read_csv(p)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = df["date"].astype(str)
    return df.sort_values("date").reset_index(drop=True)


def _compute_features(
    hist: pd.DataFrame,
    market_feat: pd.DataFrame | None,
    factory: FeatureFactory,
) -> pd.DataFrame:
    feats = factory.create_all_features(hist, market_df=None)
    if feats.empty:
        return pd.DataFrame()
    base = hist[["date", "open", "close"]].copy()
    out = pd.concat([base, feats], axis=1)
    if market_feat is not None and not market_feat.empty:
        out = out.merge(market_feat, on="date", how="left")
        close = pd.to_numeric(out["close"], errors="coerce")
        out["rel_strength_5d"] = close.pct_change(5) - out.get("market_ret_5d")
        out["rel_strength_20d"] = close.pct_change(20) - out.get("market_ret_20d")
    return out


def _compute_predictions(df: pd.DataFrame, model: dict) -> pd.Series:
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
    return pd.Series(preds, index=df.index)


def _max_drawdown(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    return float(dd.min())


def _resolve_default_model_from_config(config_path: str) -> str:
    fallback = "./models/lightgbm_fd2_excess_y2_latest.json"
    try:
        cfg = load_config(config_path)
    except Exception:
        return fallback
    model_path = (
        (cfg.get("signals", {}) or {})
        .get("model_ref", {})
        .get("path", "")
    )
    model_path = str(model_path or "").strip()
    return model_path or fallback


def run_backtest(
    model_path: str,
    manual_dir: str,
    index_symbol: str,
    months: int,
    capital: float,
    topk: int,
    max_symbols: int,
    start_date: str | None,
    end_date: str | None,
    fee: float,
    slippage: float,
    limit_up_pct: float,
) -> dict:
    model = load_ref_model(model_path)
    if not model:
        raise RuntimeError(f"model not found: {model_path}")

    manual_dir_path = Path(manual_dir)
    if not manual_dir_path.exists():
        raise RuntimeError(f"manual_hist dir not found: {manual_dir}")

    market_df = _load_index_history(manual_dir_path, index_symbol)
    if market_df.empty:
        raise RuntimeError(f"index history missing: {index_symbol}")
    market_feat = _build_market_feature_table(market_df)

    if end_date:
        end = dt.date.fromisoformat(end_date)
    else:
        end = dt.date.fromisoformat(str(market_df["date"].iloc[-1]))

    if start_date:
        start = dt.date.fromisoformat(start_date)
    else:
        start = _subtract_months(end, months)
    # extra buffer for rolling indicators
    start_buffer = start - dt.timedelta(days=220)

    trade_days = [d for d in market_df["date"].astype(str).tolist() if start.isoformat() <= d <= end.isoformat()]
    if not trade_days:
        raise RuntimeError("no trade days in range")

    symbols = _list_symbols(manual_dir_path)
    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    feature_set = str((model.get("meta") or {}).get("feature_set") or "legacy").strip().lower()
    factory = FeatureFactory(feature_set=feature_set)
    rows = []
    for i, sym in enumerate(symbols, 1):
        hist = _load_hist(manual_dir_path, sym)
        if hist.empty:
            continue
        hist = hist[hist["date"].between(start_buffer.isoformat(), end.isoformat())].copy()
        if hist.empty:
            continue
        df = _compute_features(hist, market_feat, factory)
        if df.empty:
            continue
        df = df[df["date"].between(start.isoformat(), end.isoformat())].copy()
        if df.empty:
            continue
        meta = model.get("meta", {}) or {}
        fwd = int(meta.get("forward_days", 2))
        df["fwd_ret"] = df["close"].shift(-fwd) / df["close"] - 1
        df["symbol"] = sym
        df = df.dropna(subset=["fwd_ret"])
        if df.empty:
            continue
        df["model_score"] = _compute_predictions(df, model)
        rows.append(df)
        if i % 200 == 0:
            print(f"[info] features {i}/{len(symbols)}")

    if not rows:
        raise RuntimeError("no data to backtest")

    data = pd.concat(rows, ignore_index=True)

    meta = model.get("meta", {}) or {}
    fwd = int(meta.get("forward_days", 2))
    step = fwd if fwd > 0 else 1

    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    data["entry_open"] = data.groupby("symbol")["open"].shift(-1)
    data["entry_date"] = data.groupby("symbol")["date"].shift(-1)
    data["exit_close"] = data.groupby("symbol")["close"].shift(-fwd)
    data["exit_date"] = data.groupby("symbol")["date"].shift(-fwd)

    portfolio = capital
    equity_curve = []
    trade_log = []
    idx = 0
    while idx < len(trade_days) - fwd:
        day = trade_days[idx]
        entry_day = trade_days[idx + 1]
        exit_day = trade_days[idx + fwd]
        daily = data[data["date"] == day]
        if daily.empty:
            idx += step
            continue
        daily = daily[
            (daily["entry_date"] == entry_day)
            & (daily["exit_date"] == exit_day)
            & (pd.to_numeric(daily["entry_open"], errors="coerce") > 0)
            & (pd.to_numeric(daily["exit_close"], errors="coerce") > 0)
        ].copy()
        if daily.empty:
            idx += step
            continue
        pick = daily.sort_values("model_score", ascending=False).head(topk)
        if pick.empty:
            idx += step
            continue
        # Strict entry at next-day open; skip if limit-up at open.
        prev_close = pd.to_numeric(pick["close"], errors="coerce")
        entry_open = pd.to_numeric(pick["entry_open"], errors="coerce")
        exit_close = pd.to_numeric(pick["exit_close"], errors="coerce")
        limit_up = prev_close * (1 + limit_up_pct / 100.0)
        tradable = entry_open < limit_up
        pick = pick[tradable].copy()
        if pick.empty:
            idx += step
            continue

        entry_open = pd.to_numeric(pick["entry_open"], errors="coerce")
        exit_close = pd.to_numeric(pick["exit_close"], errors="coerce")
        gross_ret = exit_close / entry_open - 1.0
        cost_in = fee + slippage
        cost_out = fee + slippage
        net_ret = (1 + gross_ret) * (1 - cost_out) / (1 + cost_in) - 1.0
        avg_ret = float(net_ret.mean())
        portfolio *= (1 + avg_ret)
        equity_curve.append(portfolio)
        trade_log.append(
            {
                "date": day,
                "avg_ret": avg_ret,
                "topk": len(pick),
                "portfolio": portfolio,
            }
        )
        idx += step

    if not trade_log:
        raise RuntimeError("no trades executed")

    total_return = portfolio / capital - 1
    wins = sum(1 for t in trade_log if t["avg_ret"] > 0)
    win_rate = wins / len(trade_log)
    mdd = _max_drawdown([t["portfolio"] for t in trade_log])

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "capital": capital,
        "final": portfolio,
        "total_return": total_return,
        "trades": len(trade_log),
        "win_rate": win_rate,
        "max_drawdown": mdd,
        "forward_days": fwd,
        "topk": topk,
        "fee": fee,
        "slippage": slippage,
        "limit_up_pct": limit_up_pct,
        "log": trade_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="")
    parser.add_argument("--manual-dir", default="./data/manual_hist")
    parser.add_argument("--index-symbol", default="000300")
    parser.add_argument("--months", type=int, default=4)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--out", default="./output/backtest_model.csv")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--limit-up", type=float, default=9.8)
    args = parser.parse_args()
    model_path = args.model.strip() or _resolve_default_model_from_config(args.config)

    res = run_backtest(
        model_path=model_path,
        manual_dir=args.manual_dir,
        index_symbol=args.index_symbol,
        months=args.months,
        capital=args.capital,
        topk=args.topk,
        max_symbols=args.max_symbols,
        start_date=args.start or None,
        end_date=args.end or None,
        fee=args.fee,
        slippage=args.slippage,
        limit_up_pct=args.limit_up,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["log"]).to_csv(out_path, index=False)

    print(f"[ok] range: {res['start']} ~ {res['end']}")
    print(f"[ok] trades: {res['trades']} | topk: {res['topk']} | forward_days: {res['forward_days']}")
    print(f"[ok] model: {model_path}")
    print(f"[ok] capital: {res['capital']:.2f} -> {res['final']:.2f}")
    print(f"[ok] total_return: {res['total_return']*100:.2f}% | win_rate: {res['win_rate']*100:.2f}% | mdd: {res['max_drawdown']*100:.2f}%")
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
