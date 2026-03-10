from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_optimize import load_config, load_universe_info, load_trade_calendar, pick_trade_days, build_rolling_windows
from sqdata.universe import filter_symbols_by_market, filter_symbols_by_board
from train_lightgbm import (
    _build_panel,
    _load_hist_from_dir,
    _prepare_cls_data,
    _impute,
    _compute_topk_hit,
    _compute_sample_weight,
)


def _resolve_universe_file(cfg: dict, override: str) -> str:
    if override:
        return override
    trade_pool = Path("data/trade_pool.csv")
    if trade_pool.exists():
        return str(trade_pool)
    return str(cfg.get("universe_file", "./data/universe.csv"))


def _filter_panel(panel: pd.DataFrame, trade_days: list[str], cfg: dict) -> pd.DataFrame:
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
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM CLS grid search (fast tuning).")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--universe-file", default="")
    parser.add_argument("--exclude-chi-next", action="store_true", help="Exclude ChiNext (创业板) symbols.")
    parser.add_argument("--mainboard-only", action="store_true", help="Restrict to mainboard symbols only.")
    parser.add_argument("--manual-hist-dir", default="data/manual_hist")
    parser.add_argument("--market-symbol", default="000300")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--forward-days", type=int, default=2)
    parser.add_argument("--label-mode", choices=["raw", "excess"], default="excess")
    parser.add_argument("--cls-threshold", type=float, default=1.5)
    parser.add_argument("--cls-threshold-mode", choices=["fixed", "atr"], default="atr")
    parser.add_argument("--cls-drop-neutral", action="store_true")
    parser.add_argument("--cls-balance", choices=["auto", "none"], default="auto")
    parser.add_argument("--rolling", action="store_true", default=True)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=4)
    parser.add_argument("--step-months", type=int, default=2)
    parser.add_argument("--max-symbols", type=int, default=800)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--recent-weight-months", type=int, default=3)
    parser.add_argument("--recent-weight-mult", type=float, default=1.5)
    parser.add_argument("--out", default="output/lightgbm_tune.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    universe_file = _resolve_universe_file(cfg, args.universe_file)

    market_scope = cfg.get("market_scope", ["sh", "sz"])
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = bool(cfg.get("exclude_chi_next", False))
    mainboard_only = bool(cfg.get("mainboard_only", True))
    if args.exclude_chi_next:
        exclude_chi_next = True
    if args.mainboard_only:
        mainboard_only = True

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

    trade_calendar_file = str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv"))
    trade_days = load_trade_calendar(trade_calendar_file)
    today = dt.date.today()
    trade_days = [d for d in trade_days if d <= today.isoformat()]
    if not trade_days:
        raise RuntimeError("trade_calendar is empty.")
    end_date = dt.date.fromisoformat(trade_days[-1])
    trade_days = pick_trade_days(trade_days, end_date, args.years)
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

    panel = _filter_panel(panel, trade_days, cfg)
    if panel.empty:
        raise RuntimeError("panel empty after filters")

    feature_cols = [
        c
        for c in panel.columns
        if c not in ("date", "symbol", "name", "future_ret", "future_ret_mkt", "close")
    ]

    if args.rolling:
        windows = build_rolling_windows(trade_days, args.train_months, args.val_months, args.step_months)
        if not windows:
            raise RuntimeError("rolling windows empty")
    else:
        windows = [(trade_days[:-1], trade_days[-1:])]

    import lightgbm as lgb  # type: ignore

    # Small grid (12 combos)
    grid_lr = [0.03, 0.05, 0.08]
    grid_leaves = [31, 63]
    grid_child = [20, 40]
    grid_estimators = [200]

    results = []

    def _cls_threshold_norm() -> float:
        thr = float(args.cls_threshold)
        if args.cls_threshold_mode == "fixed" and 0 < abs(thr) < 1:
            thr *= 100.0
        elif args.cls_threshold_mode == "atr" and thr <= 0:
            thr = 0.5
        return thr

    cls_thr = _cls_threshold_norm()

    for lr in grid_lr:
        for leaves in grid_leaves:
            for child in grid_child:
                for n_est in grid_estimators:
                    metrics = []
                    for train_days, val_days in windows:
                        train_df = panel[panel["date"].isin(train_days)].copy()
                        val_df = panel[panel["date"].isin(val_days)].copy()
                        if train_df.empty or val_df.empty:
                            continue

                        X_train, y_train, train_used = _prepare_cls_data(
                            train_df,
                            feature_cols,
                            cls_thr,
                            threshold_mode=args.cls_threshold_mode,
                            drop_neutral=args.cls_drop_neutral,
                        )
                        X_val, y_val, val_used = _prepare_cls_data(
                            val_df,
                            feature_cols,
                            cls_thr,
                            threshold_mode=args.cls_threshold_mode,
                            drop_neutral=args.cls_drop_neutral,
                        )
                        if len(X_train) == 0 or len(X_val) == 0:
                            continue

                        X_train, impute = _impute(X_train)
                        X_val = np.where(np.isnan(X_val), impute, X_val)
                        train_weight = _compute_sample_weight(train_used, args.recent_weight_months, args.recent_weight_mult)

                        scale_pos_weight = 1.0
                        if args.cls_balance == "auto":
                            pos = float(np.sum(y_train))
                            neg = float(len(y_train) - pos)
                            if pos > 0 and neg > 0:
                                scale_pos_weight = neg / pos

                        model = lgb.LGBMClassifier(
                            objective="binary",
                            n_estimators=n_est,
                            learning_rate=lr,
                            num_leaves=leaves,
                            min_child_samples=child,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            scale_pos_weight=scale_pos_weight,
                            random_state=7,
                        )
                        model.fit(X_train, y_train, sample_weight=train_weight)
                        pred = model.predict_proba(X_val)[:, 1]

                        topk_hit = _compute_topk_hit(
                            val_used,
                            pred,
                            args.topk,
                            threshold=cls_thr,
                            threshold_mode=args.cls_threshold_mode,
                        )
                        corr = float(pd.Series(pred).corr(pd.Series(y_val))) if len(pred) == len(y_val) and len(pred) > 2 else 0.0
                        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2))) if len(pred) == len(y_val) and len(pred) > 0 else 0.0
                        metrics.append({"topk": topk_hit, "corr": corr, "rmse": rmse})

                    if not metrics:
                        continue
                    avg_topk = float(np.mean([m["topk"] for m in metrics]))
                    avg_corr = float(np.mean([m["corr"] for m in metrics]))
                    avg_rmse = float(np.mean([m["rmse"] for m in metrics]))

                    results.append(
                        {
                            "learning_rate": lr,
                            "num_leaves": leaves,
                            "min_child_samples": child,
                            "n_estimators": n_est,
                            "val_topk": avg_topk,
                            "corr": avg_corr,
                            "rmse": avg_rmse,
                        }
                    )
                    print(
                        f"[grid] lr={lr} leaves={leaves} child={child} n={n_est} "
                        f"topk={avg_topk:.4f} corr={avg_corr:.4f} rmse={avg_rmse:.4f}"
                    )

    if not results:
        raise RuntimeError("no grid results")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results).sort_values(["val_topk", "corr"], ascending=[False, False])
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[ok] saved grid results: {out_path}")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
