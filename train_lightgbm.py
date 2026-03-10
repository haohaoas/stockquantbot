from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from features.feature_factory import FeatureFactory
from backtest_optimize import (
    load_config,
    load_universe_info,
    load_trade_calendar,
    pick_trade_days,
    build_rolling_windows,
    add_months,
)
from sqdata.universe import filter_symbols_by_market, filter_symbols_by_board


def _load_hist_from_dir(symbol: str, hist_dir: Path) -> pd.DataFrame:
    p = hist_dir / f"{str(symbol).zfill(6)}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=col_map)
    keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount"] if c in df.columns]
    if "date" not in keep:
        return pd.DataFrame()
    df = df[keep].copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _build_panel(
    symbols: list[str],
    name_map: dict[str, str],
    start_date: str,
    end_date: str,
    forward_days: int,
    hist_dir: Path,
    market_df: pd.DataFrame | None,
    label_mode: str = "raw",
    feature_set: str = "legacy",
) -> pd.DataFrame:
    factory = FeatureFactory(feature_set=feature_set)
    rows = []
    market_map: dict[str, float] = {}
    if label_mode == "excess" and market_df is not None and not market_df.empty:
        m = market_df.copy()
        if "date" in m.columns and "close" in m.columns:
            m = m.sort_values("date")
            m_close = pd.to_numeric(m["close"], errors="coerce")
            m["future_ret_mkt"] = (m_close.shift(-forward_days) / m_close - 1) * 100
            market_map = {str(r["date"])[:10]: float(r["future_ret_mkt"]) for _, r in m.iterrows() if pd.notna(r["future_ret_mkt"])}
    if label_mode == "excess" and not market_map:
        print("[warn] market data missing; fallback to raw label.")
    for i, sym in enumerate(symbols):
        hist = _load_hist_from_dir(sym, hist_dir)
        if hist.empty:
            continue
        hist = hist[(hist["date"] >= start_date) & (hist["date"] <= end_date)].copy()
        if hist.empty:
            continue
        feats = factory.create_all_features(hist, market_df=market_df)
        panel = pd.concat([hist[["date", "close"]], feats], axis=1)
        panel["symbol"] = str(sym).zfill(6)
        panel["name"] = name_map.get(str(sym).zfill(6), "")
        panel["future_ret"] = (panel["close"].shift(-forward_days) / panel["close"] - 1) * 100
        if label_mode == "excess" and market_map:
            panel["future_ret_mkt"] = panel["date"].astype(str).str.slice(0, 10).map(market_map)
            panel["future_ret"] = panel["future_ret"] - panel["future_ret_mkt"]
        rows.append(panel)
        if (i + 1) % 200 == 0:
            print(f"[info] loaded {i+1}/{len(symbols)} symbols")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _impute(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    impute = np.nanmedian(X, axis=0)
    impute = np.where(np.isnan(impute), 0.0, impute)
    X = np.where(np.isnan(X), impute, X)
    return X, impute


def _trim_for_forward(days: list[str], fwd: int) -> list[str]:
    if fwd <= 0:
        return days
    if len(days) <= fwd:
        return []
    return days[:-fwd]


def build_rolling_day_windows(
    trade_days: list[str],
    train_days: int = 360,
    val_days: int = 60,
    step_days: int = 20,
) -> list[tuple[list[str], list[str]]]:
    if not trade_days:
        return []
    windows: list[tuple[list[str], list[str]]] = []
    max_i = len(trade_days) - val_days
    for i in range(train_days, max_i + 1, step_days):
        train_range = trade_days[i - train_days : i]
        val_range = trade_days[i : i + val_days]
        if train_range and val_range:
            windows.append((train_range, val_range))
    return windows


def _prepare_rank_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    rank_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[int], pd.DataFrame]:
    d = df.dropna(subset=["date", "future_ret"]).copy()
    d = d.sort_values("date")
    X = d[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(d["future_ret"], errors="coerce")
    mask = y.notna()
    d = d.loc[mask].copy()
    X = X.loc[mask]
    y = y.loc[mask]

    rel = d.groupby("date")["future_ret"].rank(pct=True)
    bins = np.floor(rel * rank_bins).astype(int).clip(0, rank_bins - 1)
    uniq = sorted(pd.unique(bins))
    mapping = {v: i for i, v in enumerate(uniq)}
    y_int = bins.map(mapping).astype(int)
    group_sizes = d.groupby("date").size().tolist()
    return X.to_numpy(dtype=float), y_int.to_numpy(dtype=float), group_sizes, d


def _prepare_reg_data(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    d = df.dropna(subset=["future_ret"]).copy()
    X = d[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(d["future_ret"], errors="coerce")
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    d = d.loc[mask].copy()
    return X.to_numpy(dtype=float), y.to_numpy(dtype=float), d


def _build_threshold_series(
    df: pd.DataFrame,
    threshold: float,
    threshold_mode: str = "fixed",
    atr_col: str = "atr_percent",
) -> pd.Series:
    if threshold_mode == "atr":
        atr = pd.to_numeric(df.get(atr_col), errors="coerce").fillna(0.0)
        return atr * 100.0 * float(threshold)
    return pd.Series(float(threshold), index=df.index)


def _prepare_cls_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    label_mode: str = "threshold",
    threshold_mode: str = "fixed",
    drop_neutral: bool = False,
    pos_quantile: float = 0.8,
    neg_quantile: float = 0.2,
    atr_col: str = "atr_percent",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    d = df.dropna(subset=["future_ret"]).copy()
    y_ret = pd.to_numeric(d["future_ret"], errors="coerce")
    mask = y_ret.notna()
    d = d.loc[mask].copy()
    y_ret = y_ret.loc[mask]

    if label_mode == "quantile":
        pos_q = float(pos_quantile)
        neg_q = float(neg_quantile)
        if not (0.0 < neg_q < pos_q < 1.0):
            raise ValueError("quantile mode requires 0 < neg_quantile < pos_quantile < 1")
        hi = d.groupby("date")["future_ret"].transform(lambda s: s.quantile(pos_q))
        lo = d.groupby("date")["future_ret"].transform(lambda s: s.quantile(neg_q))
        keep = (y_ret >= hi) | (y_ret <= lo)
        d = d.loc[keep].copy()
        y_ret = y_ret.loc[keep]
        hi = hi.loc[keep]
        y_bin = (y_ret >= hi).astype(int)
    else:
        thr = _build_threshold_series(d, threshold, threshold_mode, atr_col)
        if drop_neutral and float(threshold) > 0:
            keep = y_ret.abs() > thr
            d = d.loc[keep].copy()
            y_ret = y_ret.loc[keep]
            thr = thr.loc[keep]
        y_bin = (y_ret > thr).astype(int)
    X = d[feature_cols].apply(pd.to_numeric, errors="coerce")
    return X.to_numpy(dtype=float), y_bin.to_numpy(dtype=float), d


def _compute_sample_weight(d: pd.DataFrame, recent_months: int, recent_weight_mult: float) -> np.ndarray | None:
    if recent_months <= 0 or recent_weight_mult <= 1.0 or d is None or d.empty:
        return None
    try:
        dates = pd.to_datetime(d["date"]).dt.date
    except Exception:
        return None
    if dates.empty:
        return None
    end = max(dates)
    cutoff = add_months(end, -int(recent_months))
    weights = np.where(dates >= cutoff, float(recent_weight_mult), 1.0)
    return weights.astype(float)


def _compute_topk_hit(
    df: pd.DataFrame,
    pred: np.ndarray,
    topk: int,
    threshold: float = 0.0,
    threshold_mode: str = "fixed",
    atr_col: str = "atr_percent",
) -> float:
    d = df.dropna(subset=["date", "future_ret"]).copy()
    if d.empty:
        return 0.0
    d = d.sort_values("date")
    d["pred"] = pred
    thr = _build_threshold_series(d, threshold, threshold_mode, atr_col)
    hits = []
    for _, g in d.groupby("date"):
        if g.empty:
            continue
        top = g.sort_values("pred", ascending=False).head(topk)
        hit = (top["future_ret"] > thr.loc[top.index]).mean()
        hits.append(hit)
    return float(np.mean(hits)) if hits else 0.0


def _compute_precision_at_k(
    df: pd.DataFrame,
    pred: np.ndarray,
    topk: int,
    ret_threshold: float = 0.0,
) -> float:
    d = df.dropna(subset=["date", "future_ret"]).copy()
    if d.empty:
        return 0.0
    d = d.sort_values("date")
    d["pred"] = pred
    hits = []
    for _, g in d.groupby("date"):
        if g.empty:
            continue
        top = g.sort_values("pred", ascending=False).head(topk)
        hits.append((top["future_ret"] > float(ret_threshold)).mean())
    return float(np.mean(hits)) if hits else 0.0


def _compute_topk_mean_return(
    df: pd.DataFrame,
    pred: np.ndarray,
    topk: int,
) -> float:
    d = df.dropna(subset=["date", "future_ret"]).copy()
    if d.empty:
        return 0.0
    d = d.sort_values("date")
    d["pred"] = pred
    rets = []
    for _, g in d.groupby("date"):
        if g.empty:
            continue
        top = g.sort_values("pred", ascending=False).head(topk)
        rets.append(float(top["future_ret"].mean()))
    return float(np.mean(rets)) if rets else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM model with custom feature factory")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--universe-file", default="")
    parser.add_argument("--use-trade-pool", action="store_true", help="Use data/trade_pool.csv if available.")
    parser.add_argument("--manual-hist-dir", default="data/manual_hist")
    parser.add_argument("--market-symbol", default="000300")
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--forward-days", type=int, default=5)
    parser.add_argument("--task", choices=["rank", "reg", "cls"], default="rank")
    parser.add_argument("--label-mode", choices=["raw", "excess"], default="excess", help="raw: stock return; excess: stock - market return.")
    parser.add_argument(
        "--cls-threshold",
        type=float,
        default=0.0,
        help="Threshold for future_ret. If 0<abs(x)<1 and mode=fixed, treated as percent (e.g. 0.01 => 1%%).",
    )
    parser.add_argument(
        "--cls-label-mode",
        choices=["threshold", "quantile"],
        default="threshold",
        help="threshold: fixed/ATR threshold labels; quantile: daily top/bottom quantiles as labels.",
    )
    parser.add_argument(
        "--cls-threshold-mode",
        choices=["fixed", "atr"],
        default="fixed",
        help="fixed: constant threshold; atr: threshold * atr_percent (volatility-adjusted).",
    )
    parser.add_argument("--cls-drop-neutral", action="store_true", help="Drop samples with |future_ret| <= threshold.")
    parser.add_argument("--cls-pos-quantile", type=float, default=0.8, help="Positive quantile for cls-label-mode=quantile.")
    parser.add_argument("--cls-neg-quantile", type=float, default=0.2, help="Negative quantile for cls-label-mode=quantile.")
    parser.add_argument(
        "--cls-balance",
        choices=["auto", "none"],
        default="auto",
        help="auto: set scale_pos_weight by class ratio; none: no balancing.",
    )
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--train-months", type=int, default=8)
    parser.add_argument("--val-months", type=int, default=4)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--rolling-days", action="store_true")
    parser.add_argument("--train-days", type=int, default=360)
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=20)
    parser.add_argument("--train-start", default="", help="Training start date (YYYY-MM-DD).")
    parser.add_argument("--train-end", default="", help="Training end date (YYYY-MM-DD).")
    parser.add_argument("--no-refit", action="store_true", help="Do not refit on full data; keep best rolling window model.")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--no-filter", action="store_true", help="Disable board/price/ST/amount filters (use full universe).")
    parser.add_argument("--exclude-chi-next", action="store_true", help="Exclude ChiNext (创业板) symbols.")
    parser.add_argument("--mainboard-only", action="store_true", help="Restrict to mainboard symbols only.")
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--feature-fraction", type=float, default=0.9, help="LightGBM feature fraction (colsample_bytree).")
    parser.add_argument("--bagging-fraction", type=float, default=0.9, help="LightGBM bagging fraction (subsample).")
    parser.add_argument("--bagging-freq", type=int, default=1, help="LightGBM bagging frequency.")
    parser.add_argument("--early-stopping-rounds", type=int, default=0, help="Enable early stopping when > 0.")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--precision-k", type=int, default=20, help="K for Precision@K and avg-return@K.")
    parser.add_argument("--rank-bins", type=int, default=20, help="Number of relevance bins per date for ranking.")
    parser.add_argument("--recent-weight-months", type=int, default=3, help="Months to upweight recent samples (0 to disable).")
    parser.add_argument("--recent-weight-mult", type=float, default=1.5, help="Weight multiplier for recent samples.")
    parser.add_argument("--out", default="models/lightgbm_model.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    universe_file = str(cfg.get("universe_file", "./data/universe.csv"))
    if args.universe_file:
        universe_file = args.universe_file
    elif args.use_trade_pool:
        trade_pool = Path("data/trade_pool.csv")
        if trade_pool.exists():
            universe_file = str(trade_pool)

    trade_calendar_file = str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv"))
    universe_cfg = cfg.get("universe", {}) or {}
    market_scope = cfg.get("market_scope", ["sh", "sz"])
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = bool(cfg.get("exclude_chi_next", False))
    mainboard_only = bool(cfg.get("mainboard_only", True))
    if args.exclude_chi_next:
        exclude_chi_next = True
    if args.mainboard_only:
        mainboard_only = True
    if args.no_filter:
        exclude_star = False
        exclude_chi_next = False
        mainboard_only = False

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
    train_end = args.train_end.strip() or ""
    if train_end:
        end_date = dt.date.fromisoformat(train_end)
        trade_days = [d for d in trade_days if d <= end_date.isoformat()]
    else:
        end_date = dt.date.fromisoformat(trade_days[-1])
    trade_days = pick_trade_days(trade_days, end_date, args.years)
    train_start = args.train_start.strip() or ""
    if train_start:
        trade_days = [d for d in trade_days if d >= train_start]
    if not trade_days:
        raise RuntimeError("no trade days selected for training range")

    start_date = trade_days[0]
    end_date_str = trade_days[-1]

    hist_dir = Path(args.manual_hist_dir)
    market_df = _load_hist_from_dir(args.market_symbol, hist_dir)
    if market_df.empty:
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
        raise RuntimeError("no historical data loaded")

    # Basic filters
    min_price = float(universe_cfg.get("min_price", 5.0))
    max_price = float(universe_cfg.get("max_price", 0.0))
    min_avg_amount_20 = float(universe_cfg.get("min_avg_amount_20", 0.0))
    exclude_st = bool(universe_cfg.get("exclude_st", True))
    if args.no_filter:
        min_price = 0.0
        max_price = 0.0
        min_avg_amount_20 = 0.0
        exclude_st = False

    panel = panel[panel["date"].isin(set(trade_days))].copy()
    panel = panel[panel["close"] >= min_price]
    if max_price > 0:
        panel = panel[panel["close"] <= max_price]
    if exclude_st and "name" in panel.columns:
        panel = panel[~panel["name"].astype(str).str.contains("ST|\\*ST|退", regex=True)]

    if min_avg_amount_20 > 0 and "avg_amount_20" in panel.columns:
        panel = panel[panel["avg_amount_20"] >= min_avg_amount_20]

    if panel.empty:
        raise RuntimeError("panel empty after filters")

    # Guard against leakage columns (future market return is used only for label adjustment).
    feature_cols = [
        c
        for c in panel.columns
        if c not in ("date", "symbol", "name", "future_ret", "future_ret_mkt", "close")
    ]

    # Normalize cls threshold to percent units (future_ret is in percent).
    cls_threshold = float(args.cls_threshold)
    if args.cls_threshold_mode == "fixed":
        if 0 < abs(cls_threshold) < 1:
            cls_threshold *= 100.0
    elif args.cls_threshold_mode == "atr" and cls_threshold <= 0:
        cls_threshold = 0.5
        print("[warn] cls-threshold <= 0 for atr mode; using 0.5 as default.")

    if args.rolling_days:
        windows = build_rolling_day_windows(trade_days, args.train_days, args.val_days, args.step_days)
        if not windows:
            raise RuntimeError("rolling day windows empty")
    elif args.rolling:
        windows = build_rolling_windows(trade_days, args.train_months, args.val_months, args.step_months)
        if not windows:
            raise RuntimeError("rolling windows empty")
    else:
        windows = [(trade_days[:-1], trade_days[-1:])]

    import lightgbm as lgb  # type: ignore

    metrics = []
    best_score = -1.0
    best_model = None
    best_impute = None
    best_window = None
    for train_days, val_days in windows:
        train_use = _trim_for_forward(train_days, args.forward_days)
        val_use = _trim_for_forward(val_days, args.forward_days)
        train_df = panel[panel["date"].isin(train_use)].copy()
        val_df = panel[panel["date"].isin(val_use)].copy()
        if train_df.empty or val_df.empty:
            continue

        if args.task == "rank":
            X_train, y_train, g_train, train_used = _prepare_rank_data(train_df, feature_cols, rank_bins=args.rank_bins)
            X_val, y_val, g_val, val_used = _prepare_rank_data(val_df, feature_cols, rank_bins=args.rank_bins)
        elif args.task == "cls":
            X_train, y_train, train_used = _prepare_cls_data(
                train_df,
                feature_cols,
                cls_threshold,
                label_mode=args.cls_label_mode,
                threshold_mode=args.cls_threshold_mode,
                drop_neutral=args.cls_drop_neutral,
                pos_quantile=args.cls_pos_quantile,
                neg_quantile=args.cls_neg_quantile,
            )
            X_val, y_val, val_used = _prepare_cls_data(
                val_df,
                feature_cols,
                cls_threshold,
                label_mode=args.cls_label_mode,
                threshold_mode=args.cls_threshold_mode,
                drop_neutral=args.cls_drop_neutral,
                pos_quantile=args.cls_pos_quantile,
                neg_quantile=args.cls_neg_quantile,
            )
            g_train = None
            g_val = None
        else:
            X_train, y_train, train_used = _prepare_reg_data(train_df, feature_cols)
            X_val, y_val, val_used = _prepare_reg_data(val_df, feature_cols)
            g_train = None
            g_val = None

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        X_train, impute = _impute(X_train)
        X_val = np.where(np.isnan(X_val), impute, X_val)
        train_weight = _compute_sample_weight(train_used, args.recent_weight_months, args.recent_weight_mult)

        if args.task == "rank":
            model = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                subsample=args.bagging_fraction,
                colsample_bytree=args.feature_fraction,
                bagging_freq=args.bagging_freq,
                random_state=7,
            )
            fit_kwargs = {"group": g_train, "sample_weight": train_weight}
            if args.early_stopping_rounds > 0:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["eval_group"] = [g_val]
                fit_kwargs["callbacks"] = [lgb.early_stopping(args.early_stopping_rounds, verbose=False)]
            model.fit(X_train, y_train, **fit_kwargs)
            pred = model.predict(X_val)
        elif args.task == "cls":
            scale_pos_weight = 1.0
            if args.cls_balance == "auto":
                pos = float(np.sum(y_train))
                neg = float(len(y_train) - pos)
                if pos > 0 and neg > 0:
                    scale_pos_weight = neg / pos
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                subsample=args.bagging_fraction,
                colsample_bytree=args.feature_fraction,
                bagging_freq=args.bagging_freq,
                scale_pos_weight=scale_pos_weight,
                random_state=7,
            )
            fit_kwargs = {"sample_weight": train_weight}
            if args.early_stopping_rounds > 0:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(args.early_stopping_rounds, verbose=False)]
            model.fit(X_train, y_train, **fit_kwargs)
            pred = model.predict_proba(X_val)[:, 1]
        else:
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                subsample=args.bagging_fraction,
                colsample_bytree=args.feature_fraction,
                bagging_freq=args.bagging_freq,
                random_state=7,
            )
            fit_kwargs = {"sample_weight": train_weight}
            if args.early_stopping_rounds > 0:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(args.early_stopping_rounds, verbose=False)]
            model.fit(X_train, y_train, **fit_kwargs)
            pred = model.predict(X_val)

        topk_threshold = cls_threshold if args.cls_label_mode == "threshold" else 0.0
        topk_threshold_mode = args.cls_threshold_mode if args.cls_label_mode == "threshold" else "fixed"
        topk_hit = _compute_topk_hit(
            val_used,
            pred,
            args.topk,
            threshold=topk_threshold,
            threshold_mode=topk_threshold_mode,
        )
        precision_k = _compute_precision_at_k(
            val_used,
            pred,
            max(1, int(args.precision_k)),
            ret_threshold=0.0,
        )
        topk_ret = _compute_topk_mean_return(
            val_used,
            pred,
            max(1, int(args.precision_k)),
        )
        corr = float(pd.Series(pred).corr(pd.Series(y_val))) if len(pred) == len(y_val) and len(pred) > 2 else 0.0
        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2))) if len(pred) == len(y_val) and len(pred) > 0 else 0.0
        metrics.append({"topk": topk_hit, "corr": corr, "rmse": rmse, "precision_k": precision_k, "topk_ret": topk_ret})
        if topk_hit > best_score:
            best_score = topk_hit
            best_model = model
            best_impute = impute
            if train_use and val_use:
                best_window = (train_use[0], train_use[-1], val_use[0], val_use[-1])

    if not metrics:
        raise RuntimeError("no valid windows for evaluation")

    avg_topk = float(np.mean([m["topk"] for m in metrics])) if metrics else 0.0
    avg_corr = float(np.mean([m["corr"] for m in metrics])) if metrics else 0.0
    avg_rmse = float(np.mean([m["rmse"] for m in metrics])) if metrics else 0.0
    avg_precision_k = float(np.mean([m["precision_k"] for m in metrics])) if metrics else 0.0
    avg_topk_ret = float(np.mean([m["topk_ret"] for m in metrics])) if metrics else 0.0

    if args.no_refit:
        if best_model is None or best_impute is None:
            raise RuntimeError("no valid rolling model trained")
        model = best_model
        impute = best_impute
    else:
        # Final model fit on full data
        if args.task == "rank":
            X_full, y_full, g_full, full_used = _prepare_rank_data(panel, feature_cols, rank_bins=args.rank_bins)
        elif args.task == "cls":
            X_full, y_full, full_used = _prepare_cls_data(
                panel,
                feature_cols,
                cls_threshold,
                label_mode=args.cls_label_mode,
                threshold_mode=args.cls_threshold_mode,
                drop_neutral=args.cls_drop_neutral,
                pos_quantile=args.cls_pos_quantile,
                neg_quantile=args.cls_neg_quantile,
            )
            g_full = None
        else:
            X_full, y_full, full_used = _prepare_reg_data(panel, feature_cols)
            g_full = None

        X_full, impute = _impute(X_full)
        full_weight = _compute_sample_weight(full_used, args.recent_weight_months, args.recent_weight_mult)
        if args.task == "rank":
            model = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                subsample=args.bagging_fraction,
                colsample_bytree=args.feature_fraction,
                bagging_freq=args.bagging_freq,
                random_state=7,
            )
            model.fit(X_full, y_full, group=g_full, sample_weight=full_weight)
        elif args.task == "cls":
            scale_pos_weight = 1.0
            if args.cls_balance == "auto":
                pos = float(np.sum(y_full))
                neg = float(len(y_full) - pos)
                if pos > 0 and neg > 0:
                    scale_pos_weight = neg / pos
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                subsample=args.bagging_fraction,
                colsample_bytree=args.feature_fraction,
                bagging_freq=args.bagging_freq,
                scale_pos_weight=scale_pos_weight,
                random_state=7,
            )
            model.fit(X_full, y_full, sample_weight=full_weight)
        else:
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                subsample=args.bagging_fraction,
                colsample_bytree=args.feature_fraction,
                bagging_freq=args.bagging_freq,
                random_state=7,
            )
            model.fit(X_full, y_full, sample_weight=full_weight)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_file = out_path.with_suffix(".lgb.txt")
    model.booster_.save_model(str(model_file))

    pos_rate = None
    if args.task == "cls":
        if "y_full" in locals() and len(y_full) > 0:
            pos_rate = float(np.sum(y_full) / len(y_full))

    meta = {
        "type": "lightgbm",
        "features": feature_cols,
        "model_path": model_file.name,
        "impute": {f: float(v) for f, v in zip(feature_cols, impute)},
        "meta": {
            "task": args.task,
            "forward_days": int(args.forward_days),
            "years": int(args.years),
            "feature_set": feature_set,
            "val_topk": avg_topk,
            "val_corr": avg_corr,
            "val_rmse": avg_rmse,
            "val_precision_k": avg_precision_k,
            "val_topk_ret": avg_topk_ret,
            "cls_threshold": cls_threshold if args.task == "cls" else None,
            "cls_threshold_mode": args.cls_threshold_mode if args.task == "cls" else None,
            "cls_label_mode": args.cls_label_mode if args.task == "cls" else None,
            "cls_drop_neutral": bool(args.cls_drop_neutral) if args.task == "cls" else None,
            "cls_pos_quantile": args.cls_pos_quantile if args.task == "cls" and args.cls_label_mode == "quantile" else None,
            "cls_neg_quantile": args.cls_neg_quantile if args.task == "cls" and args.cls_label_mode == "quantile" else None,
            "cls_balance": args.cls_balance if args.task == "cls" else None,
            "pos_rate": pos_rate,
            "label_mode": args.label_mode,
            "recent_weight_months": int(args.recent_weight_months),
            "recent_weight_mult": float(args.recent_weight_mult),
            "feature_fraction": float(args.feature_fraction),
            "bagging_fraction": float(args.bagging_fraction),
            "bagging_freq": int(args.bagging_freq),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "train_start": start_date,
            "train_end": end_date_str,
            "train_months": int(args.train_months) if args.rolling else None,
            "val_months": int(args.val_months) if args.rolling else None,
            "step_months": int(args.step_months) if args.rolling else None,
            "rolling": bool(args.rolling),
            "rolling_days": bool(args.rolling_days),
            "train_days": int(args.train_days) if args.rolling_days else None,
            "val_days": int(args.val_days) if args.rolling_days else None,
            "step_days": int(args.step_days) if args.rolling_days else None,
            "no_refit": bool(args.no_refit),
            "best_window": best_window,
        },
    }

    import json
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved model meta: {out_path}")
    print(
        f"[info] val topk: {avg_topk:.4f} | corr: {avg_corr:.4f} | rmse: {avg_rmse:.4f} "
        f"| p@{max(1, int(args.precision_k))}: {avg_precision_k:.4f} | ret@{max(1, int(args.precision_k))}: {avg_topk_ret:.4f}"
    )


if __name__ == "__main__":
    main()
