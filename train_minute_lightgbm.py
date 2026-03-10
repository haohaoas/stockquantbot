from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def _load_day_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "datetime" not in df.columns:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    if df.empty:
        return pd.DataFrame()
    for c in ("open", "close", "high", "low", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close", "high", "low"]).sort_values("datetime").reset_index(drop=True)
    return df


def _build_day_features(
    df: pd.DataFrame,
    symbol: str,
    horizon_bars: int,
    target_ret: float,
    key_signal_only: bool,
    entry_start: str,
    entry_end: str,
) -> pd.DataFrame:
    if df.empty or len(df) < max(30, horizon_bars + 5):
        return pd.DataFrame()

    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float).fillna(0.0)

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)

    out["ma3"] = close.rolling(3, min_periods=3).mean()
    out["ma6"] = close.rolling(6, min_periods=6).mean()
    out["ma12"] = close.rolling(12, min_periods=12).mean()

    out["dist_ma3"] = close / out["ma3"] - 1.0
    out["dist_ma6"] = close / out["ma6"] - 1.0
    out["dist_ma12"] = close / out["ma12"] - 1.0

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    out["atr_pct"] = out["atr14"] / close.replace(0, np.nan)

    vol_ma5 = volume.rolling(5, min_periods=5).mean()
    out["vol_ratio5"] = volume / vol_ma5.replace(0, np.nan)
    out["range_pct"] = (high - low) / close.replace(0, np.nan)

    hhv12_prev = high.rolling(12, min_periods=12).max().shift(1)
    breakout = close > (hhv12_prev * 1.0015)
    pullback = (close / out["ma12"] - 1.0).abs() <= 0.005
    out["key_signal"] = (breakout | pullback).astype(int)

    out["bar_idx"] = np.arange(len(out), dtype=float)
    out["bar_pos"] = out["bar_idx"] / max(1.0, float(len(out) - 1))
    open0 = float(out["open"].iloc[0])
    out["ret_from_open"] = close / open0 - 1.0 if open0 > 0 else np.nan

    out["future_ret"] = close.shift(-horizon_bars) / close - 1.0
    out["y"] = (out["future_ret"] >= target_ret).astype(int)

    # Entry window filter (default aligned with intraday_v2).
    st_h, st_m = [int(x) for x in entry_start.split(":")]
    ed_h, ed_m = [int(x) for x in entry_end.split(":")]
    start_min = st_h * 60 + st_m
    end_min = ed_h * 60 + ed_m
    mins = out["datetime"].dt.hour * 60 + out["datetime"].dt.minute
    out = out[(mins >= start_min) & (mins <= end_min)].copy()

    if key_signal_only:
        out = out[out["key_signal"] == 1].copy()

    out["symbol"] = str(symbol).zfill(6)
    out["date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out = out.dropna(
        subset=[
            "ret_1",
            "ret_3",
            "ret_6",
            "dist_ma3",
            "dist_ma6",
            "dist_ma12",
            "atr_pct",
            "vol_ratio5",
            "range_pct",
            "ret_from_open",
            "future_ret",
        ]
    ).copy()
    return out


def _topk_hit_per_date(df_val: pd.DataFrame, pred: np.ndarray, topk: int) -> float:
    if df_val.empty or topk <= 0:
        return float("nan")
    t = df_val[["date"]].copy()
    t["pred"] = pred
    t["y"] = pd.to_numeric(df_val["y"], errors="coerce").fillna(0).astype(int)
    vals = []
    for _, g in t.groupby("date"):
        gg = g.sort_values("pred", ascending=False).head(topk)
        if not gg.empty:
            vals.append(float(gg["y"].mean()))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minute-level LightGBM classifier (V1)")
    parser.add_argument("--minute-dir", default="data/manual_minute")
    parser.add_argument("--interval", default="m5")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--max-days-per-symbol", type=int, default=0)
    parser.add_argument("--horizon-bars", type=int, default=6)
    parser.add_argument("--target-ret", type=float, default=0.002)
    parser.add_argument("--key-signal-only", action="store_true", default=True)
    parser.add_argument("--no-key-signal-only", dest="key_signal_only", action="store_false")
    parser.add_argument("--entry-start", default="09:00")
    parser.add_argument("--entry-end", default="14:00")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-rows", type=int, default=800000)
    parser.add_argument("--out", default="models/minute_lightgbm_v1.json")
    args = parser.parse_args()

    base = Path(args.minute_dir)
    if not base.exists():
        raise SystemExit(f"[error] minute dir not found: {base}")

    syms = sorted([p.name for p in base.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 6])
    if args.max_symbols and args.max_symbols > 0:
        syms = syms[: int(args.max_symbols)]
    if not syms:
        raise SystemExit("[error] no symbols with minute data.")

    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=int(args.years) * 365)

    frames: list[pd.DataFrame] = []
    loaded_symbols = 0
    for i, sym in enumerate(syms, start=1):
        sym_dir = base / sym
        files = sorted(sym_dir.glob(f"*_{args.interval}.csv"))
        if not files:
            continue
        selected: list[Path] = []
        for f in files:
            try:
                d = dt.datetime.strptime(f.name.split("_", 1)[0], "%Y-%m-%d").date()
            except Exception:
                continue
            if d < start_d or d > end_d:
                continue
            selected.append(f)
        if args.max_days_per_symbol and args.max_days_per_symbol > 0:
            selected = selected[-int(args.max_days_per_symbol) :]
        if not selected:
            continue

        sym_rows = []
        for f in selected:
            day_df = _load_day_file(f)
            if day_df.empty:
                continue
            feat = _build_day_features(
                day_df,
                symbol=sym,
                horizon_bars=int(args.horizon_bars),
                target_ret=float(args.target_ret),
                key_signal_only=bool(args.key_signal_only),
                entry_start=str(args.entry_start),
                entry_end=str(args.entry_end),
            )
            if not feat.empty:
                sym_rows.append(feat)
        if sym_rows:
            loaded_symbols += 1
            frames.append(pd.concat(sym_rows, ignore_index=True))
        if i % 100 == 0:
            print(f"[info] scanned {i}/{len(syms)} symbols, loaded={loaded_symbols}")

    if not frames:
        raise SystemExit("[error] no minute training rows generated.")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["date", "symbol", "datetime"]).reset_index(drop=True)
    print(f"[info] rows={len(data)} symbols={data['symbol'].nunique()} dates={data['date'].nunique()}")

    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_6",
        "dist_ma3",
        "dist_ma6",
        "dist_ma12",
        "atr_pct",
        "vol_ratio5",
        "range_pct",
        "bar_pos",
        "ret_from_open",
    ]
    data = data.dropna(subset=feature_cols + ["y"]).copy()
    if data.empty:
        raise SystemExit("[error] training rows empty after dropna.")

    dts = pd.to_datetime(data["date"], errors="coerce")
    max_dt = dts.max()
    val_start = max_dt - pd.DateOffset(months=int(args.val_months))
    train_start = val_start - pd.DateOffset(months=int(args.train_months))
    train_mask = (dts >= train_start) & (dts < val_start)
    val_mask = dts >= val_start

    train_df = data[train_mask].copy()
    val_df = data[val_mask].copy()

    if train_df.empty or val_df.empty:
        # Fallback by date split.
        ud = sorted(pd.to_datetime(data["date"]).unique())
        cut_idx = int(len(ud) * 0.8)
        cut = ud[max(1, min(cut_idx, len(ud) - 1))]
        train_df = data[pd.to_datetime(data["date"]) < cut].copy()
        val_df = data[pd.to_datetime(data["date"]) >= cut].copy()
        print("[warn] month split empty, fallback to 80/20 date split.")

    if train_df.empty or val_df.empty:
        raise SystemExit("[error] train/val split empty.")

    if args.max_train_rows and len(train_df) > int(args.max_train_rows):
        train_df = train_df.sample(int(args.max_train_rows), random_state=int(args.seed))
        print(f"[info] train rows sampled to {len(train_df)}")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["y"].to_numpy(dtype=np.int32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["y"].to_numpy(dtype=np.int32)

    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    spw = (neg / pos) if (pos > 0 and neg > 0) else 1.0

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        random_state=int(args.seed),
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, p)) if len(np.unique(y_val)) > 1 else float("nan")
    ll = float(log_loss(y_val, p)) if len(np.unique(y_val)) > 1 else float("nan")
    acc = float(accuracy_score(y_val, (p >= 0.5).astype(int)))
    topk_hit = _topk_hit_per_date(val_df, p, int(args.topk))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = out_path.with_suffix(".txt")
    model.booster_.save_model(str(model_path))

    metrics = {
        "val_auc": auc,
        "val_logloss": ll,
        "val_acc": acc,
        "val_topk_hit": topk_hit,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_pos_rate": float(y_train.mean()) if len(y_train) else float("nan"),
        "val_pos_rate": float(y_val.mean()) if len(y_val) else float("nan"),
        "symbols": int(data["symbol"].nunique()),
        "dates": int(data["date"].nunique()),
    }
    meta = {
        "task": "minute_cls",
        "interval": args.interval,
        "horizon_bars": int(args.horizon_bars),
        "target_ret": float(args.target_ret),
        "feature_cols": feature_cols,
        "model_path": str(model_path),
        "metrics": metrics,
        "args": vars(args),
        "generated_at": dt.datetime.now().isoformat(),
    }
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] saved model: {model_path}")
    print(f"[ok] saved meta: {out_path}")
    print(
        f"[info] val auc={metrics['val_auc']:.4f} | logloss={metrics['val_logloss']:.4f} | "
        f"acc={metrics['val_acc']:.4f} | topk_hit={metrics['val_topk_hit']:.4f}"
    )


if __name__ == "__main__":
    main()

