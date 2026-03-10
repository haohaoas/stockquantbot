from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_optimize import (
    load_config,
    load_universe_info,
    load_trade_calendar,
    pick_trade_days,
)
from sqdata.universe import filter_symbols_by_market, filter_symbols_by_board
from train_lightgbm import _build_panel


def _spearman_ic(x: pd.Series, y: pd.Series) -> float | None:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 20:
        return None
    xr = x[mask].rank()
    yr = y[mask].rank()
    return float(xr.corr(yr))


def _quantile_means(values: pd.Series, ret: pd.Series, q: int) -> list[float] | None:
    values = pd.to_numeric(values, errors="coerce")
    ret = pd.to_numeric(ret, errors="coerce")
    mask = values.notna() & ret.notna()
    if mask.sum() < max(20, q * 5):
        return None
    try:
        bins = pd.qcut(values[mask], q, labels=False, duplicates="drop")
    except Exception:
        return None
    if bins is None:
        return None
    df = pd.DataFrame({"bin": bins, "ret": ret[mask].values})
    means = df.groupby("bin")["ret"].mean()
    out = []
    for i in range(q):
        out.append(float(means.get(i, np.nan)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--forward-days", type=int, default=1)
    parser.add_argument("--label-mode", type=str, default="excess")
    parser.add_argument("--max-symbols", type=int, default=1870)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--corr-thresh", type=float, default=0.9)
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    cfg = load_config("config/default.yaml")

    universe_file = str(cfg.get("universe_file", "./data/universe.csv"))
    market_scope = cfg.get("market_scope", ["sh", "sz"])
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = True
    mainboard_only = True

    symbols, name_map = load_universe_info(universe_file)
    symbols = filter_symbols_by_market(symbols, market_scope)
    symbols = filter_symbols_by_board(
        symbols,
        exclude_star=exclude_star,
        exclude_chi_next=exclude_chi_next,
        mainboard_only=mainboard_only,
    )
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    trade_days = load_trade_calendar(str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv")))
    today = dt.date.today()
    trade_days = [d for d in trade_days if d <= today.isoformat()]
    end_date = dt.date.fromisoformat(trade_days[-1])
    trade_days = pick_trade_days(trade_days, end_date, args.years)
    if not trade_days:
        raise SystemExit("no trade days")
    start_date = trade_days[0]
    end_date_str = trade_days[-1]

    hist_dir = Path("data/manual_hist")
    market_df = None
    try:
        market_df = pd.read_csv(hist_dir / "000300.csv")
        if "date" in market_df.columns:
            market_df["date"] = market_df["date"].astype(str).str.slice(0, 10)
    except Exception:
        market_df = None

    feature_set = str((cfg.get("features") or {}).get("set", "legacy") or "legacy")
    panel = _build_panel(
        symbols,
        name_map,
        start_date,
        end_date_str,
        args.forward_days,
        hist_dir,
        market_df,
        label_mode=args.label_mode,
        feature_set=feature_set,
    )
    if panel.empty:
        raise SystemExit("panel empty")

    # filters aligned with training
    universe_cfg = cfg.get("universe", {}) or {}
    min_price = float(universe_cfg.get("min_price", 5.0))
    max_price = float(universe_cfg.get("max_price", 0.0))
    min_avg_amount_20 = float(universe_cfg.get("min_avg_amount_20", 0.0))
    exclude_st = bool(universe_cfg.get("exclude_st", True))

    panel = panel[panel["date"].isin(set(trade_days))].copy()
    panel = panel[panel["close"] >= min_price]
    if max_price > 0:
        panel = panel[panel["close"] <= max_price]
    if exclude_st and "name" in panel.columns:
        panel = panel[~panel["name"].astype(str).str.contains("ST|\\*ST|退", regex=True)]
    if min_avg_amount_20 > 0 and "avg_amount_20" in panel.columns:
        panel = panel[panel["avg_amount_20"] >= min_avg_amount_20]

    feature_cols = [c for c in panel.columns if c not in ("date", "symbol", "name", "future_ret", "future_ret_mkt", "close")]
    if not feature_cols:
        raise SystemExit("no feature columns")

    grouped = panel.groupby("date")
    dates = sorted(grouped.groups.keys())

    ic_rows = []
    q_rows = []

    missing = panel[feature_cols].isna().mean().to_dict()

    for col in feature_cols:
        ic_list = []
        obs_list = []
        q_acc = []
        for d in dates:
            g = grouped.get_group(d)
            if len(g) < 20:
                continue
            ic = _spearman_ic(g[col], g["future_ret"])
            if ic is not None:
                ic_list.append(ic)
                obs_list.append(int(g[col].notna().sum()))
            q_means = _quantile_means(g[col], g["future_ret"], args.quantiles)
            if q_means is not None:
                q_acc.append(q_means)

        if len(ic_list) < args.min_days:
            continue

        ic_mean = float(np.mean(ic_list))
        ic_std = float(np.std(ic_list, ddof=1)) if len(ic_list) > 1 else 0.0
        ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        obs_avg = float(np.mean(obs_list)) if obs_list else 0.0

        ic_rows.append(
            {
                "factor": col,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "n_days": len(ic_list),
                "obs_avg": obs_avg,
                "missing_rate": float(missing.get(col, 0.0)),
            }
        )

        if q_acc:
            q_arr = np.array(q_acc, dtype=float)
            q_mean = np.nanmean(q_arr, axis=0)
            row = {"factor": col}
            for i in range(args.quantiles):
                row[f"q{i+1}"] = float(q_mean[i]) if i < len(q_mean) else np.nan
            row["top_minus_bottom"] = float(q_mean[-1] - q_mean[0]) if len(q_mean) >= 2 else np.nan
            q_rows.append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_df = pd.DataFrame(ic_rows).sort_values(["ic_ir", "ic_mean"], ascending=[False, False])
    ic_path = out_dir / "factor_ic.csv"
    ic_df.to_csv(ic_path, index=False)

    q_df = pd.DataFrame(q_rows).sort_values(["top_minus_bottom"], ascending=False)
    q_path = out_dir / "factor_quantiles.csv"
    q_df.to_csv(q_path, index=False)

    # correlation pairs
    sample = panel[feature_cols]
    if len(sample) > 200000:
        sample = sample.sample(200000, random_state=7)
    corr = sample.corr()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and abs(v) >= float(args.corr_thresh):
                pairs.append({"f1": cols[i], "f2": cols[j], "corr": float(v)})
    pair_df = pd.DataFrame(pairs).sort_values("corr", ascending=False)
    pair_path = out_dir / "factor_corr_pairs.csv"
    pair_df.to_csv(pair_path, index=False)

    print(f"[ok] saved: {ic_path}")
    print(f"[ok] saved: {q_path}")
    print(f"[ok] saved: {pair_path}")
    print(f"[info] factors: {len(ic_df)} | quantiles: {len(q_df)} | corr_pairs: {len(pair_df)}")


if __name__ == "__main__":
    main()
