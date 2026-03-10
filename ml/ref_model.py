from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from features.feature_factory import FeatureFactory
from sqdata.akshare_fetcher import fetch_hist
from sqdata.minute_fetcher import fetch_tencent_minute, fetch_biying_minute_history


_MINUTE_RT_CACHE: dict[str, dict] = {}
_MINUTE_RT_CACHE_LOCK = threading.Lock()


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_ref_model(meta_path: str) -> dict | None:
    p = Path(meta_path)
    if not p.exists():
        return None
    data = _read_json(p)
    if not data:
        return None
    model_type = str(data.get("type", "") or "lightgbm").lower()
    if model_type != "lightgbm":
        return None

    model_path = str(data.get("model_path", "") or "")
    if not model_path:
        return None
    model_file = Path(model_path)
    if not model_file.is_absolute():
        # Prefer path relative to current workspace, fallback to meta file dir.
        cwd_file = Path(model_path).resolve()
        if cwd_file.exists():
            model_file = cwd_file
        else:
            model_file = (p.parent / model_path).resolve()
    if not model_file.exists():
        return None

    if not data.get("features") and data.get("feature_cols"):
        data["features"] = list(data.get("feature_cols") or [])
    if "meta" not in data or not isinstance(data.get("meta"), dict):
        data["meta"] = {}
    if data.get("task") and not data["meta"].get("task"):
        data["meta"]["task"] = str(data.get("task"))

    try:
        import lightgbm as lgb  # type: ignore
    except Exception:
        return None
    booster = lgb.Booster(model_file=str(model_file))
    data["_model"] = booster
    return data


def predict_ref_score(df: pd.DataFrame, model: dict) -> pd.Series | None:
    if df is None or df.empty or not model:
        return None
    features = model.get("features") or []
    if not features:
        return None
    missing = [f for f in features if f not in df.columns]
    if missing:
        return None

    X = df[features].apply(pd.to_numeric, errors="coerce")
    impute = model.get("impute", {}) or {}
    for f in features:
        fill = impute.get(f, 0.0)
        X[f] = X[f].fillna(float(fill))

    booster = model.get("_model")
    if booster is None:
        return None
    preds = booster.predict(X.to_numpy(dtype=float))

    task = str((model.get("meta") or {}).get("task", "")).lower()
    if task in {"cls", "minute_cls"}:
        score = pd.Series(preds, index=df.index)
    else:
        score = pd.Series(preds, index=df.index).rank(pct=True).fillna(0.0)
    return score


def build_ref_features(
    symbols: list[str],
    trade_date: str,
    signals_cfg: dict,
    market_df: pd.DataFrame | None = None,
    market_symbol: str = "000300",
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    hist_days = int(signals_cfg.get("hist_days", 120))
    hist_days = max(hist_days, 120)
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    use_cache = bool(signals_cfg.get("hist_cache", True))
    cache_ttl_sec = int(signals_cfg.get("hist_cache_ttl_sec", 6 * 3600))
    cache_dir = str(signals_cfg.get("hist_cache_dir", "./cache/hist"))
    allow_akshare_fallback = bool(signals_cfg.get("allow_akshare_fallback", True))
    use_parallel = bool(signals_cfg.get("parallel", True))
    max_workers = int(signals_cfg.get("max_workers", 8))

    feature_set = str(signals_cfg.get("feature_set") or "legacy").strip().lower()
    factory = FeatureFactory(feature_set=feature_set)
    if market_df is None or market_df.empty:
        # Try to fetch index history for market-relative features.
        hist_days_idx = max(hist_days, 260)
        try:
            market_df = fetch_hist(
                market_symbol,
                end_date=trade_date,
                hist_days=hist_days_idx,
                use_proxy=use_proxy,
                proxy=proxy,
                cache_dir=cache_dir,
                cache_ttl_sec=cache_ttl_sec,
                use_cache=use_cache,
                allow_akshare_fallback=allow_akshare_fallback,
            )
        except Exception:
            market_df = None
    min_hist = max(factory.return_windows + factory.ma_windows + [20, 26, 14]) + 5
    rows: list[pd.Series] = []
    lock = threading.Lock()

    def _compute(sym: str) -> None:
        try:
            hist = fetch_hist(
                sym,
                end_date=trade_date,
                hist_days=hist_days,
                use_proxy=use_proxy,
                proxy=proxy,
                cache_dir=cache_dir,
                cache_ttl_sec=cache_ttl_sec,
                use_cache=use_cache,
                allow_akshare_fallback=allow_akshare_fallback,
            )
        except Exception:
            return

        if hist is None or hist.empty or len(hist) < min_hist:
            return
        feats = factory.create_all_features(hist, market_df=market_df)
        if feats.empty:
            return
        last = feats.iloc[-1].copy()
        last["symbol"] = str(sym).zfill(6)
        with lock:
            rows.append(last)

    if use_parallel and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for sym in symbols:
                ex.submit(_compute, str(sym))
    else:
        for sym in symbols:
            _compute(str(sym))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.set_index("symbol")
    return df


def _parse_hhmm(text: str, default_minutes: int) -> int:
    s = str(text or "").strip()
    if not s:
        return default_minutes
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return default_minutes


def _load_minute_day_file(path: Path) -> pd.DataFrame:
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
    for c in ("open", "close", "high", "low", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close", "high", "low"]).sort_values("datetime").reset_index(drop=True)
    return df


def _build_minute_day_features(
    day_df: pd.DataFrame,
    symbol: str,
    entry_start: str,
    entry_end: str,
) -> pd.DataFrame:
    if day_df.empty or len(day_df) < 20:
        return pd.DataFrame()
    out = day_df.copy()
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

    start_min = _parse_hhmm(entry_start, 9 * 60)
    end_min = _parse_hhmm(entry_end, 14 * 60)
    mins = out["datetime"].dt.hour * 60 + out["datetime"].dt.minute
    out = out[(mins >= start_min) & (mins <= end_min)].copy()
    out["symbol"] = str(symbol).zfill(6)

    use_cols = [
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
        "key_signal",
        "symbol",
        "datetime",
    ]
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
        ]
    )
    return out[use_cols].copy()


def _latest_minute_file(sym: str, minute_dir: str, interval: str) -> Path | None:
    p = Path(minute_dir) / str(sym).zfill(6)
    if not p.exists():
        return None
    files = sorted(p.glob(f"*_{interval}.csv"))
    if not files:
        return None
    return files[-1]


def _merge_minute_df(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.copy() if new is not None else pd.DataFrame()
    if new is None or new.empty:
        return old.copy()
    out = pd.concat([old, new], ignore_index=True)
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        out = out.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last")
        out = out.sort_values("datetime").reset_index(drop=True)
    return out


def _save_minute_day_file(df: pd.DataFrame, minute_dir: str, symbol: str, interval: str) -> None:
    if df is None or df.empty or "datetime" not in df.columns:
        return
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if out.empty:
        return
    out["date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    day = str(out["date"].iloc[-1])
    out = out[out["date"] == day].copy()
    if out.empty:
        return
    ddir = Path(minute_dir) / str(symbol).zfill(6)
    ddir.mkdir(parents=True, exist_ok=True)
    file_path = ddir / f"{day}_{interval}.csv"
    out.to_csv(file_path, index=False)


def _fetch_live_minute_for_symbol(
    symbol: str,
    interval: str,
    signals_cfg: dict,
    trade_date: str,
) -> pd.DataFrame:
    provider = str(signals_cfg.get("realtime_provider", "auto") or "auto").strip().lower()
    order = signals_cfg.get("realtime_provider_order") or []
    if provider == "auto":
        providers = [str(x).strip().lower() for x in order if str(x).strip()]
        if not providers:
            providers = ["biying", "tencent"]
    else:
        providers = [provider]

    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    timeout = int(signals_cfg.get("minute_live_timeout", 4) or 4)
    limit = int(signals_cfg.get("minute_live_limit", 320) or 320)
    ymd = str(trade_date or "").replace("-", "")[:8]

    for p in providers:
        try:
            if p == "tencent":
                df = fetch_tencent_minute(
                    symbol,
                    interval=interval,
                    limit=limit,
                    use_proxy=use_proxy,
                    proxy=proxy,
                    timeout=timeout,
                )
            elif p == "biying":
                licence = str(
                    signals_cfg.get("realtime_biying_licence")
                    or signals_cfg.get("biying_licence")
                    or ""
                ).strip()
                if not licence:
                    continue
                base_url = str(signals_cfg.get("realtime_biying_base_url") or "http://api.biyingapi.com")
                biying_timeout = int(signals_cfg.get("realtime_biying_timeout", timeout) or timeout)
                df = fetch_biying_minute_history(
                    symbol,
                    interval=interval,
                    licence=licence,
                    start=ymd,
                    end=ymd,
                    limit=0,
                    base_url=base_url,
                    use_proxy=use_proxy,
                    proxy=proxy,
                    timeout=biying_timeout,
                )
            else:
                continue
            if df is not None and not df.empty:
                df = df.copy()
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
                if not df.empty:
                    return df
        except Exception:
            continue
    return pd.DataFrame()


def _get_live_minute_cached(
    symbol: str,
    interval: str,
    signals_cfg: dict,
    trade_date: str,
) -> pd.DataFrame:
    try:
        cache_ttl = int(signals_cfg.get("minute_live_cache_sec", 180))
    except Exception:
        cache_ttl = 180
    key = f"{str(symbol).zfill(6)}|{interval}|{trade_date}"
    now = dt.datetime.now().timestamp()
    with _MINUTE_RT_CACHE_LOCK:
        ent = _MINUTE_RT_CACHE.get(key)
        if cache_ttl > 0 and ent and (now - float(ent.get("ts", 0.0))) <= cache_ttl:
            df = ent.get("df")
            if isinstance(df, pd.DataFrame):
                return df.copy()

    df = _fetch_live_minute_for_symbol(symbol, interval, signals_cfg, trade_date)
    with _MINUTE_RT_CACHE_LOCK:
        _MINUTE_RT_CACHE[key] = {"ts": now, "df": df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()}
    return df


def build_minute_ref_features(
    symbols: list[str],
    model: dict,
    signals_cfg: dict,
    trade_date: str = "",
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    args = model.get("args") or {}
    interval = str(model.get("interval") or args.get("interval") or "m5")
    minute_dir = str(signals_cfg.get("minute_dir") or args.get("minute_dir") or "./data/manual_minute")
    key_signal_only = bool(args.get("key_signal_only", False))
    entry_start = str(args.get("entry_start") or "09:00")
    entry_end = str(args.get("entry_end") or "14:00")
    max_workers = int(signals_cfg.get("minute_live_max_workers", 10) or 10)
    live_enabled = bool(signals_cfg.get("minute_live_update", True))
    live_only_today = bool(signals_cfg.get("minute_live_only_today", True))
    live_only_missing_today = bool(signals_cfg.get("minute_live_only_missing_today", False))
    live_max_symbols = int(signals_cfg.get("minute_live_max_symbols", 0) or 0)

    td: dt.date | None = None
    try:
        if trade_date:
            td = dt.date.fromisoformat(str(trade_date)[:10])
    except Exception:
        td = None

    rows: list[pd.Series] = []
    lock = threading.Lock()
    today = dt.date.today()
    allow_live = live_enabled and (not live_only_today or (td is not None and td == today))
    live_syms: set[str] = set()
    if allow_live:
        if live_max_symbols > 0:
            live_syms = {str(s).zfill(6) for s in symbols[:live_max_symbols]}
        else:
            live_syms = {str(s).zfill(6) for s in symbols}

    def _worker(sym: str) -> None:
        sym = str(sym).zfill(6)
        latest_file = _latest_minute_file(sym, minute_dir, interval)
        day_df = _load_minute_day_file(latest_file) if latest_file else pd.DataFrame()
        local_day = None
        if latest_file is not None:
            try:
                local_day = dt.datetime.strptime(latest_file.name.split("_", 1)[0], "%Y-%m-%d").date()
            except Exception:
                local_day = None

        if allow_live and sym in live_syms:
            need_live = True
            if live_only_missing_today and local_day is not None and td is not None and local_day >= td:
                need_live = False
            if need_live:
                live_df = _get_live_minute_cached(sym, interval, signals_cfg, str(trade_date)[:10])
                if live_df is not None and not live_df.empty:
                    live_df["datetime"] = pd.to_datetime(live_df["datetime"], errors="coerce")
                    live_df = live_df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
                    if not live_df.empty:
                        if day_df is None or day_df.empty:
                            day_df = live_df
                        else:
                            day_df = _merge_minute_df(day_df, live_df)
                        try:
                            _save_minute_day_file(day_df, minute_dir, sym, interval)
                        except Exception:
                            pass

        if day_df.empty:
            return
        feat_df = _build_minute_day_features(day_df, sym, entry_start, entry_end)
        if feat_df.empty:
            return
        if key_signal_only:
            key_df = feat_df[feat_df["key_signal"] == 1]
            row = key_df.iloc[-1] if not key_df.empty else feat_df.iloc[-1]
        else:
            row = feat_df.iloc[-1]
        with lock:
            rows.append(row)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for sym in symbols:
                ex.submit(_worker, sym)
    else:
        for sym in symbols:
            _worker(sym)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.set_index("symbol")
    return out
