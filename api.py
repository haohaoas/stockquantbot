from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time
import threading
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, Body, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app as app_module
from sqdata.news_fetcher import fetch_news_eastmoney, simple_sentiment
from sqdata.news_sentiment import get_market_news_sentiment
from sqdata.ai_explain import explain_row
from sqdata import akshare_fetcher as akshare_module

app = FastAPI(title="StockQuantBot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_WEB_DIST_DIR = Path(__file__).resolve().parent / "web" / "dist"
_WEB_INDEX_FILE = _WEB_DIST_DIR / "index.html"
_WEB_ASSETS_DIR = _WEB_DIST_DIR / "assets"
if _WEB_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_WEB_ASSETS_DIR)), name="web-assets")

_MARKET_CACHE: dict[str, dict[str, Any]] = {}
_MARKET_CACHE_LOCK = threading.Lock()
_NEWS_SUMMARY_CACHE: dict[str, dict[str, Any]] = {}
_SNAPSHOT_REFRESH_STATE: dict[str, dict[str, Any]] = {}
_SNAPSHOT_SCHEDULER_STARTED = False
_SNAPSHOT_SCHEDULER_LOCK = threading.Lock()
_REVIEW_JOURNAL_LOCK = threading.Lock()
_REVIEW_JOURNAL_PATH = Path("./data/review_journal.json")
_REVIEW_MONITOR_CFG_PATH = Path("./data/review_monitor_config.json")
_CACHE_BUST_SIGNAL_PATH = Path("./cache/data_refresh_signal.json")
_CACHE_BUST_STATE: dict[str, int] = {"mtime_ns": 0}
_SNAPSHOT_CACHE_DIR = Path("./cache/api_snapshots")


def _market_cache_key(
    mode: str,
    top_n: int | None,
    only_buy: bool,
    intraday: bool,
    model_identity: str | None,
    provider: str | None,
    model_independent: bool,
    model_sector: str | None,
) -> str:
    return f"{mode}|{top_n or ''}|{int(only_buy)}|{int(intraday)}|{model_identity or ''}|{provider or ''}|{int(model_independent)}|{(model_sector or '').strip()}"


def _resolve_model_identity(params: dict) -> str:
    # Keep API cache aligned with the actual model file in use, not only UI model key.
    model_key = str(params.get("model_key", "") or "").strip()
    model_ref_cfg = ((params.get("signals") or {}).get("model_ref") or {})
    model_path = str(model_ref_cfg.get("path", "") or "").strip()
    if not model_path:
        return model_key
    try:
        mtime = int(os.path.getmtime(model_path))
    except Exception:
        mtime = 0
    model_file = Path(model_path).name
    return f"{model_key}|{model_file}|{mtime}"


def _get_market_cache(key: str, ttl_sec: int) -> dict[str, Any] | None:
    if ttl_sec <= 0:
        return None
    now = time.time()
    with _MARKET_CACHE_LOCK:
        entry = _MARKET_CACHE.get(key)
        if entry and now - float(entry.get("ts", 0)) <= ttl_sec:
            return entry.get("data")

    disk_entry = _get_market_disk_cache_entry(key)
    if not disk_entry:
        return None
    if now - float(disk_entry.get("ts", 0)) > ttl_sec:
        return None
    return disk_entry.get("data")


def _get_market_cache_entry(key: str) -> dict[str, Any] | None:
    with _MARKET_CACHE_LOCK:
        entry = _MARKET_CACHE.get(key)
        if entry:
            return dict(entry)
    return _get_market_disk_cache_entry(key)


def _set_market_cache(key: str, data: dict[str, Any]) -> None:
    with _MARKET_CACHE_LOCK:
        _MARKET_CACHE[key] = {"ts": time.time(), "data": data}
    _set_market_disk_cache(key, data)


def _market_disk_cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _SNAPSHOT_CACHE_DIR / f"{digest}.json"


def _get_market_disk_cache_entry(key: str) -> dict[str, Any] | None:
    path = _market_disk_cache_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("key", "")) != key:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    return {"ts": float(payload.get("ts", 0) or 0), "data": data}


def _set_market_disk_cache(key: str, data: dict[str, Any]) -> None:
    try:
        _SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "ts": time.time(), "data": data}
        _market_disk_cache_path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _clear_runtime_caches() -> None:
    with _MARKET_CACHE_LOCK:
        _MARKET_CACHE.clear()
        _NEWS_SUMMARY_CACHE.clear()
        _SNAPSHOT_REFRESH_STATE.clear()
    try:
        app_module.clear_runtime_caches()
    except Exception:
        pass
    try:
        akshare_module.clear_runtime_caches()
    except Exception:
        pass


def _apply_external_cache_bust_if_needed() -> None:
    try:
        st = _CACHE_BUST_SIGNAL_PATH.stat()
    except Exception:
        return
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    if mtime_ns <= int(_CACHE_BUST_STATE.get("mtime_ns", 0) or 0):
        return
    _clear_runtime_caches()
    _CACHE_BUST_STATE["mtime_ns"] = mtime_ns


def _resolve_cache_ttl(cfg: dict) -> int:
    if not cfg:
        return 0
    if "api_cache_sec" in cfg:
        return int(cfg.get("api_cache_sec") or 0)
    output_cfg = cfg.get("output", {}) or {}
    if "api_cache_sec" in output_cfg:
        return int(output_cfg.get("api_cache_sec") or 0)
    return 0


def _resolve_top_n(cfg: dict, top_n: int | None) -> int:
    if top_n is not None:
        return int(top_n)
    output_cfg = (cfg or {}).get("output", {}) or {}
    return int(output_cfg.get("top_n", 20) or 20)


def _resolve_news_cache_ttl(cfg: dict) -> int:
    if not cfg:
        return 0
    news_cfg = cfg.get("news", {}) or {}
    if "summary_cache_sec" in news_cfg:
        return int(news_cfg.get("summary_cache_sec") or 0)
    return 0


def _get_news_cache(ttl_sec: int) -> dict[str, Any] | None:
    if ttl_sec <= 0:
        return None
    now = time.time()
    with _MARKET_CACHE_LOCK:
        entry = _NEWS_SUMMARY_CACHE.get("summary")
        if not entry:
            return None
        if now - float(entry.get("ts", 0)) > ttl_sec:
            return None
        return entry.get("data")


def _set_news_cache(data: dict[str, Any]) -> None:
    with _MARKET_CACHE_LOCK:
        _NEWS_SUMMARY_CACHE["summary"] = {"ts": time.time(), "data": data}


def _load_deepseek_key() -> str:
    secret = app_module.load_secret("config/secret.json")
    key = str(secret.get("deepseek_api_key", "") or "")
    if key:
        return key
    return str(os.environ.get("DEEPSEEK_API_KEY") or "")


def _find_row_in_cache(symbol: str) -> dict[str, Any] | None:
    norm = app_module._normalize_symbol(symbol)
    if not norm:
        return None
    best_row = None
    best_ts = 0.0
    with _MARKET_CACHE_LOCK:
        for entry in _MARKET_CACHE.values():
            ts = float(entry.get("ts", 0))
            data = entry.get("data") or {}
            for key in ("rows", "model_top"):
                rows = data.get(key) or []
                for row in rows:
                    row_sym = app_module._normalize_symbol(str(row.get("symbol", "")))
                    if row_sym == norm and ts >= best_ts:
                        best_ts = ts
                        best_row = row
    return best_row


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _build_params(
    mode: str,
    top_n: int | None,
    intraday: bool,
    model_key: str | None,
    provider: str | None,
    model_independent: bool,
    model_sector: str | None,
) -> tuple[dict, dict]:
    cfg = app_module.load_config("config/default.yaml")
    universe_cfg = cfg.get("universe", {}) or {}
    signals_cfg = cfg.get("signals", {}) or {}
    weights = cfg.get("score_weights", {}) or {}
    decision_cfg = cfg.get("decision", {}) or {}
    features_cfg = cfg.get("features", {}) or {}
    output_cfg = cfg.get("output", {}) or {}
    index_cfg = cfg.get("index", {}) or {}
    sector_cfg = cfg.get("sector_boost", {}) or {}
    news_cfg = cfg.get("news", {}) or {}
    deepseek_cfg = cfg.get("deepseek", {}) or {}

    universe_file = str(cfg.get("universe_file", "./data/universe.csv"))
    use_universe_file = bool(cfg.get("use_universe_file", True))
    allow_eastmoney_fallback = bool(cfg.get("allow_eastmoney_fallback", False))
    trade_calendar_file = str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv"))
    use_trade_calendar_file = bool(cfg.get("use_trade_calendar_file", True))
    market_scope = cfg.get("market_scope", ["sh", "sz"])
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = bool(cfg.get("exclude_chi_next", False))
    mainboard_only = bool(cfg.get("mainboard_only", True))
    watchlist_file = str(cfg.get("watchlist_file", "./data/watchlist.csv"))
    exclude_limit_up = bool(cfg.get("exclude_limit_up", True))
    limit_up_pct = float(cfg.get("limit_up_pct", 9.8))
    exclude_non_realtime_pct = bool(cfg.get("exclude_non_realtime_pct", True))
    if mode == "watchlist":
        # Watchlist should be tolerant of real-time gaps and limit-up filters.
        exclude_non_realtime_pct = False
        exclude_limit_up = False

    model_key = str(model_key or "").strip()
    model_ref_cfg = signals_cfg.get("model_ref") or {}
    model_opts = model_ref_cfg.get("options") or {}
    selected_model = ""
    if model_key and model_key in model_opts:
        candidate_path = str(model_opts.get(model_key) or "")
        if candidate_path and os.path.exists(candidate_path):
            model_ref_cfg = dict(model_ref_cfg)
            model_ref_cfg["path"] = candidate_path
            model_ref_cfg["selected"] = model_key
            signals_cfg = dict(signals_cfg)
            signals_cfg["model_ref"] = model_ref_cfg
            selected_model = model_key

    if model_independent and mode != "watchlist":
        model_ref_cfg = dict(model_ref_cfg)
        # Detach model candidate pool from rule preselect pool.
        # Full-universe(backtest_like) may be slower but matches "全量" expectation.
        model_ref_cfg["candidate_mode"] = "backtest_like"
        signals_cfg = dict(signals_cfg)
        signals_cfg["model_ref"] = model_ref_cfg

    signals = dict(signals_cfg)
    provider = str(provider or "").strip().lower()
    if provider in {"biying", "tencent", "sina", "netease", "auto"}:
        signals["realtime_provider"] = provider
        if provider != "auto":
            # Explicit source switch should disable provider fallback chain.
            signals["realtime_provider_order"] = [provider]
    if mode == "watchlist":
        symbols_list = app_module.load_watchlist(watchlist_file)
        signals["symbols"] = symbols_list
    else:
        # Ensure full-market mode is not constrained by config symbols.
        for k in ("symbols", "watchlist", "universe"):
            signals.pop(k, None)

    params = {
        "trade_date": "",
        "top_n": int(top_n) if top_n is not None else int(output_cfg.get("top_n", 20)),
        "universe": {
            "min_price": float(universe_cfg.get("min_price", 5.0)),
            "max_price": float(universe_cfg.get("max_price", 0.0)),
            "min_avg_amount_20": float(universe_cfg.get("min_avg_amount_20", 0.0)),
            "exclude_st": bool(universe_cfg.get("exclude_st", True)),
        },
        "signals": {**signals, "intraday_breakout": bool(intraday)},
        "universe_file": universe_file,
        "use_universe_file": bool(use_universe_file),
        "allow_eastmoney_fallback": bool(allow_eastmoney_fallback),
        "trade_calendar_file": trade_calendar_file,
        "use_trade_calendar_file": bool(use_trade_calendar_file),
        "market_scope": market_scope,
        "exclude_star": bool(exclude_star),
        "exclude_chi_next": bool(exclude_chi_next),
        "mainboard_only": bool(mainboard_only),
        "watchlist_file": watchlist_file,
        "exclude_limit_up": bool(exclude_limit_up),
        "limit_up_pct": float(limit_up_pct),
        "exclude_non_realtime_pct": bool(exclude_non_realtime_pct),
        "is_watchlist": mode == "watchlist",
        "regime_filter": bool(index_cfg.get("regime_filter", True)),
        "index_symbol": str(index_cfg.get("symbol", "000300")),
        "index_ma_window": int(index_cfg.get("ma_window", 200)),
        "features": features_cfg,
        "sector_boost": sector_cfg,
        "weights": weights,
        "decision": decision_cfg,
        "model_key": selected_model,
        "model_independent": bool(model_independent),
        "model_sector": str(model_sector or "").strip(),
        "news": news_cfg,
        "deepseek": deepseek_cfg,
    }
    return params, cfg


def _get_model_options(cfg: dict) -> dict[str, Any]:
    signals_cfg = (cfg or {}).get("signals", {}) or {}
    model_ref_cfg = signals_cfg.get("model_ref") or {}
    default_path = str(model_ref_cfg.get("path") or "")
    options = model_ref_cfg.get("options") or {}

    items: list[dict[str, str]] = []
    default_matched = False
    for key in sorted(options.keys()):
        path = str(options.get(key) or "")
        if not path or not os.path.exists(path):
            continue
        if default_path and path == default_path:
            default_matched = True
        items.append({"key": str(key), "label": str(key), "path": path})

    if not items:
        items.append({"key": "", "label": "default", "path": default_path})
    elif default_path and not default_matched and os.path.exists(default_path):
        items.insert(0, {"key": "", "label": "default", "path": default_path})
    elif default_path and not default_matched:
        # Keep default selectable to avoid empty UI, even if path is missing.
        items.insert(0, {"key": "", "label": "default", "path": default_path})

    return {
        "default_path": default_path,
        "items": items,
    }


def _append_market_notes(
    payload: dict[str, Any],
    *,
    params: dict,
    provider: str | None,
    model_independent: bool,
    model_sector: str | None,
) -> dict[str, Any]:
    notes = list(payload.get("notes") or [])
    if params.get("model_key"):
        notes.append(f"模型切换: {params['model_key']}")
    if provider:
        notes.append(f"行情源切换: {provider}")
    if model_independent:
        notes.append("模型候选池: 独立于规则池")
    if model_sector:
        notes.append(f"模型板块筛选: {model_sector}")
    payload["notes"] = notes
    return payload


def _resolve_runtime_context(cfg: dict) -> tuple[dt.datetime, bool]:
    now_cn = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    trade_calendar_file = str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv"))
    use_trade_calendar_file = bool(cfg.get("use_trade_calendar_file", True))
    calendar_dates = app_module._load_trade_calendar(trade_calendar_file) if use_trade_calendar_file else set()
    market_open = app_module._is_market_open_cn(now_cn) and app_module._is_trading_day_cn(now_cn.date(), calendar_dates)
    return now_cn, market_open


def _enrich_runtime_payload(
    payload: dict[str, Any],
    *,
    market_open: bool,
    now_cn: dt.datetime,
    extra_note: str | None = None,
) -> dict[str, Any]:
    out = dict(payload)
    out["market_open"] = market_open
    out["server_time"] = now_cn.isoformat()
    notes = list(out.get("notes") or [])
    if extra_note:
        notes.append(extra_note)
    out["notes"] = notes
    return out


def _refresh_model_top_quotes(
    payload: dict[str, Any],
    *,
    params: dict,
) -> dict[str, Any]:
    model_rows = payload.get("model_top") or []
    if not isinstance(model_rows, list) or not model_rows:
        return dict(payload)

    df = pd.DataFrame(model_rows)
    if df.empty or "symbol" not in df.columns:
        return dict(payload)

    signals_cfg = params.get("signals", {}) or {}
    provider = str(signals_cfg.get("realtime_provider", "auto") or "auto").strip().lower()
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    max_rows = min(len(df), int(params.get("top_n") or len(df)))

    try:
        refreshed = app_module.refresh_realtime_for_view(
            df,
            use_proxy=use_proxy,
            proxy=proxy,
            max_rows=max_rows,
            provider=provider,
        )
    except Exception:
        refreshed = df

    out = dict(payload)
    out["model_top"] = _df_to_records(refreshed)
    return out


def _compute_market_snapshot(
    *,
    mode: str,
    top_n: int | None,
    only_buy: bool,
    intraday: bool,
    model: str | None,
    provider: str | None,
    model_independent: bool,
    model_sector: str | None,
) -> tuple[dict[str, Any], dict, dict, str, int]:
    params, cfg = _build_params(mode, top_n, intraday, model, provider, model_independent, model_sector)
    cache_ttl = _resolve_cache_ttl(cfg)
    cache_top_n = _resolve_top_n(cfg, top_n)
    model_identity = _resolve_model_identity(params)
    cache_key = _market_cache_key(
        mode,
        cache_top_n,
        only_buy,
        intraday,
        model_identity,
        provider,
        bool(model_independent),
        model_sector,
    )
    result = app_module.compute_market(params)
    df = result.get("df", pd.DataFrame())
    if only_buy and not df.empty and "action" in df.columns:
        df = df[df["action"] == "BUY"].copy()
    payload = {
        "rows": _df_to_records(df),
        "model_top": _df_to_records(result.get("model_top", pd.DataFrame())),
        "notes": result.get("notes", []),
        "stats": result.get("stats", {}),
        "trade_date": str(app_module.resolve_trade_date("")),
    }
    payload = _refresh_model_top_quotes(payload, params=params)
    payload = _append_market_notes(
        payload,
        params=params,
        provider=provider,
        model_independent=bool(model_independent),
        model_sector=model_sector,
    )
    return payload, cfg, params, cache_key, cache_ttl


def _compute_model_top_snapshot(
    *,
    mode: str,
    top_n: int | None,
    intraday: bool,
    model: str | None,
    provider: str | None,
    model_independent: bool,
    model_sector: str | None,
) -> tuple[dict[str, Any], dict, dict, str, int]:
    params, cfg = _build_params(mode, top_n, intraday, model, provider, model_independent, model_sector)
    params["model_top_only"] = True
    cache_ttl = _resolve_cache_ttl(cfg)
    cache_top_n = _resolve_top_n(cfg, top_n)
    model_identity = _resolve_model_identity(params)
    cache_key = "model_top|" + _market_cache_key(
        mode,
        cache_top_n,
        False,
        intraday,
        model_identity,
        provider,
        bool(model_independent),
        model_sector,
    )
    result = app_module.compute_market(params)
    payload = {
        "model_top": _df_to_records(result.get("model_top", pd.DataFrame())),
        "notes": result.get("notes", []),
        "stats": result.get("stats", {}),
        "trade_date": str(app_module.resolve_trade_date("")),
    }
    payload = _refresh_model_top_quotes(payload, params=params)
    payload = _append_market_notes(
        payload,
        params=params,
        provider=provider,
        model_independent=bool(model_independent),
        model_sector=model_sector,
    )
    return payload, cfg, params, cache_key, cache_ttl


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload or {})
    out["rows"] = [dict(x) for x in (payload.get("rows") or [])] if isinstance(payload, dict) else []
    out["model_top"] = [dict(x) for x in (payload.get("model_top") or [])] if isinstance(payload, dict) else []
    out["notes"] = list(payload.get("notes") or []) if isinstance(payload, dict) else []
    out["stats"] = dict(payload.get("stats") or {}) if isinstance(payload, dict) and isinstance(payload.get("stats"), dict) else {}
    return out


def _filter_buy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        try:
            if str(row.get("action", "") or "").upper() == "BUY":
                out.append(dict(row))
        except Exception:
            continue
    return out


def _prepare_market_fallback_payload(
    payload: dict[str, Any],
    *,
    only_buy: bool,
    note: str,
) -> dict[str, Any]:
    out = _clone_payload(payload)
    if only_buy:
        out["rows"] = _filter_buy_rows(out.get("rows") or [])
    notes = list(out.get("notes") or [])
    notes.append(note)
    out["notes"] = notes
    return out


def _prepare_snapshot_context(
    *,
    kind: str,
    mode: str,
    top_n: int | None,
    only_buy: bool,
    intraday: bool,
    model: str | None,
    provider: str | None,
    model_independent: bool,
    model_sector: str | None,
) -> tuple[dict, dict, str, int]:
    params, cfg = _build_params(mode, top_n, intraday, model, provider, model_independent, model_sector)
    if kind == "model_top":
        params["model_top_only"] = True
    cache_ttl = _resolve_cache_ttl(cfg)
    cache_top_n = _resolve_top_n(cfg, top_n)
    model_identity = _resolve_model_identity(params)
    cache_key = _market_cache_key(
        mode,
        cache_top_n,
        only_buy if kind == "market" else False,
        intraday,
        model_identity,
        provider,
        bool(model_independent),
        model_sector,
    )
    if kind == "model_top":
        cache_key = "model_top|" + cache_key
    return params, cfg, cache_key, cache_ttl


def _snapshot_refresh_running(key: str) -> bool:
    with _MARKET_CACHE_LOCK:
        st = _SNAPSHOT_REFRESH_STATE.get(key) or {}
        return bool(st.get("running", False))


def _trigger_snapshot_refresh(
    key: str,
    runner,
    *,
    min_gap_sec: int = 5,
    start_delay_sec: float = 0.0,
) -> bool:
    now = time.time()
    with _MARKET_CACHE_LOCK:
        st = _SNAPSHOT_REFRESH_STATE.get(key) or {}
        if bool(st.get("running", False)):
            return False
        if now - float(st.get("last_trigger_ts", 0.0) or 0.0) < max(1, int(min_gap_sec)):
            return False
        _SNAPSHOT_REFRESH_STATE[key] = {
            **st,
            "running": True,
            "last_trigger_ts": now,
        }

    def _worker() -> None:
        err = ""
        try:
            payload = runner()
            if isinstance(payload, dict):
                _set_market_cache(key, payload)
        except Exception as e:
            err = str(e)
        finally:
            with _MARKET_CACHE_LOCK:
                st2 = _SNAPSHOT_REFRESH_STATE.get(key) or {}
                st2["running"] = False
                st2["last_finish_ts"] = time.time()
                st2["last_error"] = err
                _SNAPSHOT_REFRESH_STATE[key] = st2

    delay = max(0.0, float(start_delay_sec or 0.0))
    if delay > 0:
        timer = threading.Timer(delay, _worker)
        timer.daemon = True
        timer.name = f"snapshot-refresh-delay:{key[:24]}"
        timer.start()
    else:
        threading.Thread(target=_worker, daemon=True, name=f"snapshot-refresh:{key[:32]}").start()
    return True


def _default_provider(cfg: dict) -> str:
    signals_cfg = cfg.get("signals", {}) or {}
    provider = str(signals_cfg.get("realtime_provider", "auto") or "auto").strip().lower()
    return provider if provider in {"auto", "biying", "tencent", "sina", "netease"} else "auto"


def _resolve_refresh_intervals(cfg: dict) -> tuple[int, int, int]:
    output_cfg = cfg.get("output", {}) or {}
    market_sec = int(output_cfg.get("backend_market_refresh_sec", 300) or 300)
    model_sec = int(output_cfg.get("backend_model_refresh_sec", 300) or 300)
    model_independent_sec = int(output_cfg.get("backend_model_independent_refresh_sec", 24 * 3600) or (24 * 3600))
    return market_sec, model_sec, model_independent_sec


def _scheduled_model_keys(cfg: dict) -> list[str | None]:
    signals_cfg = (cfg or {}).get("signals", {}) or {}
    model_ref_cfg = signals_cfg.get("model_ref") or {}
    options = model_ref_cfg.get("options") or {}
    keys: list[str | None] = [None]
    for key in sorted(options.keys()):
        k = str(key or "").strip()
        if not k:
            continue
        keys.append(k)
    return keys


def _scheduled_market_specs(cfg: dict, provider: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for mode in ("all", "watchlist"):
        for model_key in _scheduled_model_keys(cfg):
            specs.append(
                {
                    "mode": mode,
                    "top_n": None,
                    "only_buy": False,
                    "intraday": False,
                    "model": model_key,
                    "provider": provider,
                    "model_independent": False,
                    "model_sector": None,
                }
            )
    return specs


def _find_market_fallback_entry(
    *,
    cfg: dict,
    mode: str,
    top_n: int | None,
    only_buy: bool,
    intraday: bool,
    model: str | None,
    provider: str | None,
    model_independent: bool,
    model_sector: str | None,
) -> tuple[dict[str, Any], str] | None:
    default_provider = _default_provider(cfg)
    candidates: list[tuple[dict[str, Any], str]] = []
    if only_buy:
        candidates.append(
            (
                {
                    "mode": mode,
                    "top_n": top_n,
                    "only_buy": False,
                    "intraday": intraday,
                    "model": model,
                    "provider": provider,
                    "model_independent": model_independent,
                    "model_sector": model_sector,
                },
                "只看 BUY 快照预热中，先展示基础快照筛选结果",
            )
        )
    provider_norm = str(provider or "").strip().lower()
    if default_provider and provider_norm and provider_norm != default_provider:
        candidates.append(
            (
                {
                    "mode": mode,
                    "top_n": top_n,
                    "only_buy": only_buy,
                    "intraday": intraday,
                    "model": model,
                    "provider": default_provider,
                    "model_independent": model_independent,
                    "model_sector": model_sector,
                },
                f"行情源 {provider_norm} 快照预热中，先展示 {default_provider} 快照",
            )
        )
        if only_buy:
            candidates.append(
                (
                    {
                        "mode": mode,
                        "top_n": top_n,
                        "only_buy": False,
                        "intraday": intraday,
                        "model": model,
                        "provider": default_provider,
                        "model_independent": model_independent,
                        "model_sector": model_sector,
                    },
                    f"行情源 {provider_norm} 和只看 BUY 快照预热中，先展示 {default_provider} 基础快照筛选结果",
                )
            )

    seen_keys: set[str] = set()
    for kwargs, note in candidates:
        _, _, candidate_key, _ = _prepare_snapshot_context(kind="market", **kwargs)
        if candidate_key in seen_keys:
            continue
        seen_keys.add(candidate_key)
        entry = _get_market_cache_entry(candidate_key)
        if entry is None or not entry.get("data"):
            continue
        payload = _prepare_market_fallback_payload(entry["data"], only_buy=only_buy, note=note)
        return payload, note
    return None


def _maybe_schedule_background_refresh(
    *,
    kind: str,
    mode: str,
    top_n: int | None,
    only_buy: bool,
    intraday: bool,
    model: str | None,
    provider: str | None,
    model_independent: bool,
    model_sector: str | None,
    force: bool = False,
    defer_sec: float = 0.0,
) -> tuple[str, bool]:
    _, cfg, cache_key, _ = _prepare_snapshot_context(
        kind=kind,
        mode=mode,
        top_n=top_n,
        only_buy=only_buy,
        intraday=intraday,
        model=model,
        provider=provider,
        model_independent=model_independent,
        model_sector=model_sector,
    )
    if kind == "market":
        runner = lambda: _compute_market_snapshot(
            mode=mode,
            top_n=top_n,
            only_buy=only_buy,
            intraday=intraday,
            model=model,
            provider=provider,
            model_independent=model_independent,
            model_sector=model_sector,
        )[0]
    else:
        runner = lambda: _compute_model_top_snapshot(
            mode=mode,
            top_n=top_n,
            intraday=intraday,
            model=model,
            provider=provider,
            model_independent=model_independent,
            model_sector=model_sector,
        )[0]
    min_gap = 0 if force else max(5, int(_resolve_cache_ttl(cfg) or 5))
    triggered = _trigger_snapshot_refresh(cache_key, runner, min_gap_sec=min_gap, start_delay_sec=defer_sec)
    return cache_key, triggered


def _snapshot_scheduler_loop() -> None:
    while True:
        try:
            _apply_external_cache_bust_if_needed()
            cfg = app_module.load_config("config/default.yaml")
            provider = _default_provider(cfg)
            market_sec, model_sec, model_independent_sec = _resolve_refresh_intervals(cfg)
            specs = [("market", kwargs, market_sec) for kwargs in _scheduled_market_specs(cfg, provider)]
            for model_key in _scheduled_model_keys(cfg):
                specs.append(
                    ("model_top", {"mode": "all", "top_n": None, "only_buy": False, "intraday": False, "model": model_key, "provider": provider, "model_independent": False, "model_sector": None}, model_sec)
                )
                specs.append(
                    ("model_top", {"mode": "all", "top_n": None, "only_buy": False, "intraday": False, "model": model_key, "provider": provider, "model_independent": True, "model_sector": None}, model_independent_sec)
                )
            now_cn, market_open = _resolve_runtime_context(cfg)
            today = str(app_module.resolve_trade_date(""))
            for kind, kwargs, interval_sec in specs:
                _, _, cache_key, _ = _prepare_snapshot_context(kind=kind, **kwargs)
                entry = _get_market_cache_entry(cache_key)
                if entry is None:
                    _maybe_schedule_background_refresh(kind=kind, force=True, **kwargs)
                    continue
                age = time.time() - float(entry.get("ts", 0.0) or 0.0)
                payload = entry.get("data") or {}
                trade_date = str(payload.get("trade_date", "") or "")
                due = age >= max(30, int(interval_sec))
                if kind == "market" and not market_open:
                    due = trade_date != today
                if kind == "model_top" and trade_date != today:
                    due = True
                if due:
                    _maybe_schedule_background_refresh(kind=kind, force=True, **kwargs)
        except Exception:
            pass
        time.sleep(30)


def _start_snapshot_scheduler() -> None:
    global _SNAPSHOT_SCHEDULER_STARTED
    with _SNAPSHOT_SCHEDULER_LOCK:
        if _SNAPSHOT_SCHEDULER_STARTED:
            return
        threading.Thread(target=_snapshot_scheduler_loop, daemon=True, name="snapshot-scheduler").start()
        _SNAPSHOT_SCHEDULER_STARTED = True


@app.get("/", include_in_schema=False)
def index():
    # Serve built web UI in production; fallback to Vite dev server for local development.
    if _WEB_INDEX_FILE.exists():
        return FileResponse(path=str(_WEB_INDEX_FILE))
    return RedirectResponse(url="http://localhost:5173", status_code=302)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.on_event("startup")
def on_startup() -> None:
    _apply_external_cache_bust_if_needed()
    _start_snapshot_scheduler()


@app.get("/api/market")
def get_market(
    mode: str = Query("all", pattern="^(all|watchlist)$"),
    top_n: int | None = Query(None),
    only_buy: bool = Query(False),
    intraday: bool = Query(False),
    model: str | None = Query(None),
    provider: str | None = Query(None, pattern="^(auto|biying|tencent|sina|netease)?$"),
    model_independent: bool = Query(False),
    model_sector: str | None = Query(None),
) -> dict:
    _apply_external_cache_bust_if_needed()
    _start_snapshot_scheduler()
    params, cfg, cache_key, cache_ttl = _prepare_snapshot_context(
        kind="market",
        mode=mode,
        top_n=top_n,
        only_buy=only_buy,
        intraday=intraday,
        model=model,
        provider=provider,
        model_independent=model_independent,
        model_sector=model_sector,
    )
    now_cn, market_open = _resolve_runtime_context(cfg)
    cached = _get_market_cache(cache_key, cache_ttl)
    if cached is not None:
        return _enrich_runtime_payload(cached, market_open=market_open, now_cn=now_cn, extra_note=f"缓存命中({cache_ttl}s)")

    entry = _get_market_cache_entry(cache_key)
    if entry is not None and entry.get("data"):
        _maybe_schedule_background_refresh(
            kind="market",
            mode=mode,
            top_n=top_n,
            only_buy=only_buy,
            intraday=intraday,
            model=model,
            provider=provider,
            model_independent=model_independent,
            model_sector=model_sector,
            force=True,
            defer_sec=0.25,
        )
        note = "展示缓存快照，后台刷新中" if _snapshot_refresh_running(cache_key) else "展示缓存快照，后台刷新已触发"
        payload = entry["data"]
        return _enrich_runtime_payload(payload, market_open=market_open, now_cn=now_cn, extra_note=note)

    fallback = _find_market_fallback_entry(
        cfg=cfg,
        mode=mode,
        top_n=top_n,
        only_buy=only_buy,
        intraday=intraday,
        model=model,
        provider=provider,
        model_independent=model_independent,
        model_sector=model_sector,
    )
    if fallback is not None:
        _maybe_schedule_background_refresh(
            kind="market",
            mode=mode,
            top_n=top_n,
            only_buy=only_buy,
            intraday=intraday,
            model=model,
            provider=provider,
            model_independent=model_independent,
            model_sector=model_sector,
            force=True,
            defer_sec=0.25,
        )
        payload = fallback[0]
        return _enrich_runtime_payload(payload, market_open=market_open, now_cn=now_cn)

    payload, _, _, _, _ = _compute_market_snapshot(
        mode=mode,
        top_n=top_n,
        only_buy=only_buy,
        intraday=intraday,
        model=model,
        provider=provider,
        model_independent=model_independent,
        model_sector=model_sector,
    )
    _set_market_cache(cache_key, payload)
    return _enrich_runtime_payload(payload, market_open=market_open, now_cn=now_cn)


@app.get("/api/model-top")
def get_model_top(
    mode: str = Query("all", pattern="^(all|watchlist)$"),
    top_n: int | None = Query(None),
    intraday: bool = Query(False),
    model: str | None = Query(None),
    provider: str | None = Query(None, pattern="^(auto|biying|tencent|sina|netease)?$"),
    model_independent: bool = Query(False),
    model_sector: str | None = Query(None),
) -> dict:
    _apply_external_cache_bust_if_needed()
    _start_snapshot_scheduler()
    params, cfg, cache_key, cache_ttl = _prepare_snapshot_context(
        kind="model_top",
        mode=mode,
        top_n=top_n,
        only_buy=False,
        intraday=intraday,
        model=model,
        provider=provider,
        model_independent=model_independent,
        model_sector=model_sector,
    )
    now_cn, market_open = _resolve_runtime_context(cfg)
    cached = _get_market_cache(cache_key, cache_ttl)
    if cached is not None:
        cached = _refresh_model_top_quotes(cached, params=params)
        return _enrich_runtime_payload(cached, market_open=market_open, now_cn=now_cn, extra_note=f"缓存命中({cache_ttl}s)")

    entry = _get_market_cache_entry(cache_key)
    if entry is not None and entry.get("data"):
        _maybe_schedule_background_refresh(
            kind="model_top",
            mode=mode,
            top_n=top_n,
            only_buy=False,
            intraday=intraday,
            model=model,
            provider=provider,
            model_independent=model_independent,
            model_sector=model_sector,
            force=True,
            defer_sec=0.25,
        )
        note = "展示缓存快照，后台刷新中" if _snapshot_refresh_running(cache_key) else "展示缓存快照，后台刷新已触发"
        payload = _refresh_model_top_quotes(entry["data"], params=params)
        return _enrich_runtime_payload(payload, market_open=market_open, now_cn=now_cn, extra_note=note)

    if model_independent:
        _, _, shared_cache_key, _ = _prepare_snapshot_context(
            kind="model_top",
            mode=mode,
            top_n=top_n,
            only_buy=False,
            intraday=intraday,
            model=model,
            provider=provider,
            model_independent=False,
            model_sector=model_sector,
        )
        shared_entry = _get_market_cache_entry(shared_cache_key)
        if shared_entry is not None and shared_entry.get("data"):
            _maybe_schedule_background_refresh(
                kind="model_top",
                mode=mode,
                top_n=top_n,
                only_buy=False,
                intraday=intraday,
                model=model,
                provider=provider,
                model_independent=True,
                model_sector=model_sector,
                force=True,
                defer_sec=0.25,
            )
            payload = _refresh_model_top_quotes(shared_entry["data"], params=params)
            return _enrich_runtime_payload(
                payload,
                market_open=market_open,
                now_cn=now_cn,
                extra_note="独立候选预热中，先展示共享候选快照",
            )

    if mode == "watchlist" or not model_independent:
        payload, _, _, _, _ = _compute_model_top_snapshot(
            mode=mode,
            top_n=top_n,
            intraday=intraday,
            model=model,
            provider=provider,
            model_independent=model_independent,
            model_sector=model_sector,
        )
        _set_market_cache(cache_key, payload)
        payload = _refresh_model_top_quotes(payload, params=params)
        return _enrich_runtime_payload(payload, market_open=market_open, now_cn=now_cn)

    _maybe_schedule_background_refresh(
        kind="model_top",
        mode=mode,
        top_n=top_n,
        only_buy=False,
        intraday=intraday,
        model=model,
        provider=provider,
        model_independent=model_independent,
        model_sector=model_sector,
        force=True,
        defer_sec=0.25,
    )
    placeholder = {
        "model_top": [],
        "notes": ["模型榜单首次生成中，后台已启动预热"],
        "stats": {},
        "trade_date": str(app_module.resolve_trade_date("")),
    }
    return _enrich_runtime_payload(placeholder, market_open=market_open, now_cn=now_cn)


@app.get("/api/model-options")
def get_model_options() -> dict:
    cfg = app_module.load_config("config/default.yaml")
    return _get_model_options(cfg)


@app.get("/api/watchlist")
def get_watchlist(include_name: bool = Query(False)) -> dict:
    cfg = app_module.load_config("config/default.yaml")
    watchlist_file = str(cfg.get("watchlist_file", "./data/watchlist.csv"))
    syms = app_module.load_watchlist(watchlist_file)
    if not include_name:
        return {"symbols": syms}
    mapping = _load_symbol_name_map()
    items = [{"symbol": s, "name": mapping.get(str(s).zfill(6), "")} for s in syms]
    return {"symbols": syms, "items": items}


_SYMBOL_NAME_CACHE: dict[str, str] | None = None
_NAME_SYMBOL_CACHE: dict[str, str] | None = None


def _load_symbol_name_map() -> dict[str, str]:
    global _SYMBOL_NAME_CACHE
    if _SYMBOL_NAME_CACHE is not None:
        return _SYMBOL_NAME_CACHE
    cfg = app_module.load_config("config/default.yaml")
    universe_file = str(cfg.get("universe_file", "./data/universe.csv"))
    try:
        df = pd.read_csv(universe_file)
        if "symbol" in df.columns and "name" in df.columns:
            mapping = {}
            for _, row in df[["symbol", "name"]].iterrows():
                sym = str(row["symbol"]).strip()
                name = str(row["name"]).strip()
                if not sym:
                    continue
                sym = sym.zfill(6)
                if sym and name and sym not in mapping:
                    mapping[sym] = name
            _SYMBOL_NAME_CACHE = mapping
            return mapping
    except Exception:
        pass
    _SYMBOL_NAME_CACHE = {}
    return _SYMBOL_NAME_CACHE


def _load_name_symbol_map() -> dict[str, str]:
    global _NAME_SYMBOL_CACHE
    if _NAME_SYMBOL_CACHE is not None:
        return _NAME_SYMBOL_CACHE
    reverse: dict[str, str] = {}
    for sym, name in _load_symbol_name_map().items():
        n = str(name or "").strip()
        if n and n not in reverse:
            reverse[n] = str(sym).zfill(6)
    _NAME_SYMBOL_CACHE = reverse
    return _NAME_SYMBOL_CACHE


def _parse_news_time(val: str) -> dt.datetime | None:
    if not val:
        return None
    s = str(val).strip()
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


@app.get("/api/news")
def get_news(
    symbol: str | None = Query(None),
    limit: int = Query(20),
) -> dict:
    cfg = app_module.load_config("config/default.yaml")
    signals_cfg = cfg.get("signals", {}) or {}
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    if not proxy and app_module._get_env_proxy():
        use_proxy = True
    elif use_proxy and not proxy:
        detected = app_module._detect_proxy()
        if detected:
            proxy = detected
    keyword = symbol or "A股"
    related_name = ""
    if symbol:
        mapping = _load_symbol_name_map()
        related_name = mapping.get(str(symbol).zfill(6), "")
    try:
        items = fetch_news_eastmoney(keyword, limit=int(limit), use_proxy=use_proxy, proxy=proxy)
        enriched = []
        for it in items:
            title = str(it.get("title", "") or "")
            content = str(it.get("content", "") or "")
            score, label = simple_sentiment(title + " " + content)
            enriched.append(
                {
                    **it,
                    "symbol": symbol or "",
                    "name": related_name,
                    "sentiment": label,
                    "sentiment_score": score,
                }
            )
        enriched.sort(key=lambda x: _parse_news_time(x.get("time", "")) or dt.datetime.min, reverse=True)
        return {"keyword": keyword, "items": enriched, "count": len(enriched)}
    except Exception as e:
        return {"keyword": keyword, "items": [], "count": 0, "error": str(e)}


@app.get("/api/news/summary")
def get_news_summary(limit: int = Query(30)) -> dict:
    cfg = app_module.load_config("config/default.yaml")
    signals_cfg = cfg.get("signals", {}) or {}
    deep_cfg = cfg.get("deepseek", {}) or {}
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    if not proxy and app_module._get_env_proxy():
        use_proxy = True
        proxy = app_module._get_env_proxy()
    elif use_proxy and not proxy:
        detected = app_module._detect_proxy()
        if detected:
            proxy = detected

    ttl = _resolve_news_cache_ttl(cfg)
    cached = _get_news_cache(ttl)
    if cached is not None:
        return {**cached, "cached": True}

    api_key = _load_deepseek_key() if bool(deep_cfg.get("enabled", False)) else ""
    base_url = str(deep_cfg.get("base_url", "") or "")
    model = str(deep_cfg.get("model", "deepseek-chat") or "deepseek-chat")

    summary = get_market_news_sentiment(
        limit=int(limit),
        use_proxy=use_proxy,
        proxy=proxy,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    payload = {"summary": summary, "cached": False}
    _set_news_cache(payload)
    return payload


class ExplainRequest(BaseModel):
    symbol: str
    row: dict[str, Any] | None = None


class ReviewRequest(BaseModel):
    mode: str = "all"
    review_date: str | None = None
    rows: list[dict[str, Any]] | None = None
    model_top: list[dict[str, Any]] | None = None
    notes: list[str] | None = None
    news_summary: dict[str, Any] | None = None
    watchlist: list[dict[str, Any]] | None = None
    operations: list[dict[str, Any]] | None = None
    note_text: str | None = None


@app.post("/api/explain")
def explain(req: ExplainRequest) -> dict:
    symbol = str(req.symbol or "").strip()
    if not symbol:
        return {"error": "symbol required"}

    row = req.row or _find_row_in_cache(symbol)
    if not row:
        return {"symbol": symbol, "error": "row not found"}

    cfg = app_module.load_config("config/default.yaml")
    signals_cfg = cfg.get("signals", {}) or {}
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    if not proxy and app_module._get_env_proxy():
        use_proxy = True
        proxy = app_module._get_env_proxy()
    elif use_proxy and not proxy:
        detected = app_module._detect_proxy()
        if detected:
            proxy = detected

    api_key = _load_deepseek_key()
    text, source, levels = explain_row(row, cfg, api_key=api_key, proxy=proxy if use_proxy else "")
    return {"symbol": symbol, "explain": text, "source": source, "levels": levels}


def _review_top_rows(rows: list[dict[str, Any]] | None, limit: int = 5) -> list[dict[str, Any]]:
    items = rows or []
    if not items:
        return []
    out = []
    for row in items[:limit]:
        out.append(
            {
                "symbol": str(row.get("symbol", "") or ""),
                "name": str(row.get("name", "") or ""),
                "action": str(row.get("action", "") or ""),
                "score": row.get("score"),
                "model_score": row.get("model_score"),
                "pct_chg": row.get("pct_chg"),
                "sector": str(row.get("sector", "") or ""),
                "reason": str(row.get("reason", "") or ""),
            }
        )
    return out


def _today_cn_str() -> str:
    return dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).date().isoformat()


def _norm_review_date(val: str | None) -> str:
    s = str(val or "").strip()
    if not s:
        return _today_cn_str()
    try:
        return dt.date.fromisoformat(s[:10]).isoformat()
    except Exception:
        return _today_cn_str()


def _load_review_journal() -> dict[str, Any]:
    if not _REVIEW_JOURNAL_PATH.exists():
        return {"days": {}}
    try:
        obj = json.loads(_REVIEW_JOURNAL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"days": {}}
    if not isinstance(obj, dict):
        return {"days": {}}
    if "days" not in obj or not isinstance(obj.get("days"), dict):
        obj["days"] = {}
    return obj


def _save_review_journal(journal: dict[str, Any]) -> None:
    _REVIEW_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REVIEW_JOURNAL_PATH.write_text(json.dumps(journal, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_review_day(journal: dict[str, Any], review_date: str) -> dict[str, Any]:
    days = journal.setdefault("days", {})
    if review_date not in days or not isinstance(days.get(review_date), dict):
        days[review_date] = {
            "messages": [],
            "operations": [],
            "note_text": "",
            "card": {},
            "draft_card": {},
            "error_tag": "",
            "updated_at": "",
        }
    day = days[review_date]
    if "messages" not in day or not isinstance(day.get("messages"), list):
        day["messages"] = []
    if "operations" not in day or not isinstance(day.get("operations"), list):
        day["operations"] = []
    if "note_text" not in day:
        day["note_text"] = ""
    if "card" not in day or not isinstance(day.get("card"), dict):
        day["card"] = {}
    if "draft_card" not in day or not isinstance(day.get("draft_card"), dict):
        day["draft_card"] = {}
    if "error_tag" not in day:
        day["error_tag"] = ""
    return day


def _normalize_operation(item: dict[str, Any], *, op_id: str | None = None) -> dict[str, Any]:
    symbol = app_module._normalize_symbol(item.get("symbol", ""))
    side = str(item.get("side", "BUY") or "BUY").upper()
    if side not in {"BUY", "SELL"}:
        side = "BUY"
    price = pd.to_numeric(item.get("price"), errors="coerce")
    qty = pd.to_numeric(item.get("qty"), errors="coerce")
    result_pct = pd.to_numeric(item.get("result_pct"), errors="coerce")
    now_s = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    oid = str(op_id or item.get("id") or f"{int(time.time() * 1000)}_{abs(hash(now_s)) % 100000}").strip()
    return {
        "id": oid,
        "symbol": symbol or "",
        "name": str(item.get("name", "") or ""),
        "side": side,
        "price": None if pd.isna(price) else float(price),
        "qty": None if pd.isna(qty) else int(qty),
        "result_pct": None if pd.isna(result_pct) else float(result_pct),
        "time": str(item.get("time", "") or ""),
        "reason": str(item.get("reason", "") or ""),
        "created_at": str(item.get("created_at", "") or now_s),
    }


def _normalize_chat_message(item: dict[str, Any]) -> dict[str, Any]:
    role = str(item.get("role", "user") or "user").strip().lower()
    if role not in {"user", "assistant"}:
        role = "user"
    text = str(item.get("text", "") or "").strip()
    if not text:
        text = ""
    now_s = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    mid = str(item.get("id", "") or "").strip()
    if not mid:
        seed = f"{role}|{text}|{now_s}"
        mid = hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]
    return {
        "id": mid,
        "role": role,
        "text": text,
        "created_at": str(item.get("created_at", "") or now_s),
    }


def _extract_symbols_from_text(text: str) -> list[str]:
    if not text:
        return []
    found = re.findall(r"(?<!\d)([0-9]{6})(?!\d)", text)
    out: list[str] = []
    for s in found:
        ns = app_module._normalize_symbol(s)
        if ns and ns not in out:
            out.append(ns)
    name_map = _load_name_symbol_map()
    if name_map:
        # Prefer longer names first so "洲际油气" beats shorter partial overlaps.
        for name, sym in sorted(name_map.items(), key=lambda item: len(item[0]), reverse=True):
            if name and name in text and sym not in out:
                out.append(sym)
    return out


def _extract_operations_from_text(text: str, review_date: str, *, fallback_symbols: list[str] | None = None) -> list[dict[str, Any]]:
    if not text:
        return []
    chunks = [x.strip() for x in re.split(r"[，,。；;\n]+", text) if x.strip()]
    out: list[dict[str, Any]] = []
    symbol_name_map = _load_symbol_name_map()
    last_symbols: list[str] = []
    for chunk in chunks:
        used_fallback_symbols = False
        syms = _extract_symbols_from_text(chunk)
        if syms:
            last_symbols = syms[:]
        elif last_symbols and re.search(r"(卖出|卖掉|卖了|止盈|止损|减仓|清仓|出了|买入|买进|买了|加仓|开仓|低吸|建仓|亏|赚|盈利|浮盈|浮亏)", chunk):
            syms = last_symbols[:]
        elif fallback_symbols and re.search(r"(卖出|卖掉|卖了|止盈|止损|减仓|清仓|出了|买入|买进|买了|加仓|开仓|低吸|建仓|亏|赚|盈利|浮盈|浮亏)", chunk):
            syms = fallback_symbols[:]
            used_fallback_symbols = True
        if not syms:
            continue
        side = ""
        if re.search(r"(卖出|卖掉|止盈|止损|减仓|清仓|出了|卖了)", chunk):
            side = "SELL"
        elif re.search(r"(买入|买进|加仓|开仓|低吸|买了|建仓)", chunk):
            side = "BUY"
        if not side:
            continue
        price = None
        m_price = re.search(r"(?:买入|买进|卖出|卖了|在|于|价|价格|均价|成本|@)\s*([0-9]+(?:\.[0-9]+)?)", chunk)
        if not m_price:
            m_price = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:元)?\s*(?:买入|买进|卖出|卖了|买了|开仓|建仓)", chunk)
        if not m_price:
            m_price = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*元", chunk)
        if m_price:
            try:
                price = float(m_price.group(1))
            except Exception:
                price = None

        qty = None
        m_qty = re.search(r"([0-9]{1,6})\s*(股|手)", chunk)
        if m_qty:
            try:
                q = int(m_qty.group(1))
                qty = q * 100 if m_qty.group(2) == "手" else q
            except Exception:
                qty = None

        result_pct = None
        m_ret = re.search(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*%", chunk)
        if m_ret:
            try:
                result_pct = float(m_ret.group(1))
            except Exception:
                result_pct = None
        if result_pct is None:
            m_pts = re.search(r"(亏|跌|回撤|损失|赚|盈利|浮盈|浮亏|涨)?\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:个点|个百分点|点)", chunk)
            if m_pts:
                try:
                    result_pct = float(m_pts.group(2))
                    direction = str(m_pts.group(1) or "")
                    if direction in {"亏", "跌", "回撤", "损失", "浮亏"}:
                        result_pct = -abs(result_pct)
                    elif direction in {"赚", "盈利", "浮盈", "涨"}:
                        result_pct = abs(result_pct)
                except Exception:
                    result_pct = None
        if result_pct is None:
            m_ret_plain = re.search(r"(?:盈亏|结果|亏损|浮亏|浮盈|盈利)\D*([+-]?[0-9]+(?:\.[0-9]+)?)\b", chunk)
            if m_ret_plain:
                try:
                    result_pct = float(m_ret_plain.group(1))
                except Exception:
                    result_pct = None

        reason = ""
        m_reason = re.search(r"(?:因为|原因|理由|纪律)[:：]?\s*(.+)$", chunk)
        if m_reason:
            reason = str(m_reason.group(1)).strip()

        tm = ""
        m_tm = re.search(r"\b([0-2]?[0-9]:[0-5][0-9])\b", chunk)
        if m_tm:
            tm = m_tm.group(1)

        # Cross-message fallback is reliable for sells/result updates, but a bare "然后就买了"
        # without price/qty is too ambiguous and often duplicates the previous buy.
        if used_fallback_symbols and side == "BUY" and price is None and qty is None and not tm:
            continue

        for sym in syms:
            seed = f"{review_date}|{sym}|{side}|{price}|{qty}|{tm}|{reason}"
            op_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:20]
            out.append(
                _normalize_operation(
                    {
                        "id": op_id,
                        "symbol": sym,
                        "name": symbol_name_map.get(sym, ""),
                        "side": side,
                        "price": price,
                        "qty": qty,
                        "result_pct": result_pct,
                        "time": tm,
                        "reason": reason,
                    },
                    op_id=op_id,
                )
            )
    if out:
        global_pcts = re.findall(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*%", text)
        if len(global_pcts) == 1:
            try:
                pct_val = float(global_pcts[0])
                for i in range(len(out)):
                    if pd.isna(pd.to_numeric(out[i].get("result_pct"), errors="coerce")):
                        out[i]["result_pct"] = pct_val
            except Exception:
                pass
    return out


def _extract_result_hint(text: str) -> tuple[float | None, bool]:
    if not text:
        return None, False
    explicit = re.search(
        r"(?:实际(?:盈亏)?|最终(?:盈亏|结果)?|总(?:盈亏|共)|账户|盈亏|结果|只亏|只赚|亏损|盈利|浮亏|浮盈)\D*([+-]?[0-9]+(?:\.[0-9]+)?)",
        text,
    )
    if explicit:
        try:
            val = float(explicit.group(1))
            prefix = explicit.group(0)
            if re.search(r"(只亏|亏损|浮亏)", prefix) and val > 0:
                val = -val
            elif re.search(r"(只赚|盈利|浮盈)", prefix) and val < 0:
                val = abs(val)
            return val, True
        except Exception:
            pass
    generic = re.search(r"(亏|跌|回撤|损失|赚|盈利|浮盈|浮亏|涨)?\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:个点|个百分点|点|%)", text)
    if generic:
        try:
            val = float(generic.group(2))
            direction = str(generic.group(1) or "")
            if direction in {"亏", "跌", "回撤", "损失", "浮亏"}:
                val = -abs(val)
            elif direction in {"赚", "盈利", "浮盈", "涨"}:
                val = abs(val)
            return val, False
        except Exception:
            pass
    return None, False


def _apply_result_hint_to_operations(
    operations: list[dict[str, Any]],
    *,
    symbols: list[str],
    result_pct: float,
    explicit_override: bool,
    prefer_sell: bool,
) -> list[dict[str, Any]]:
    if not operations or not symbols:
        return operations

    sell_exists = any(
        app_module._normalize_symbol(op.get("symbol", "")) in symbols and str(op.get("side", "")).upper() == "SELL"
        for op in operations
    )
    search_orders: list[str] = []
    if prefer_sell and sell_exists:
        search_orders.append("SELL")
    else:
        if prefer_sell:
            search_orders.append("SELL")
        search_orders.append("")
    seen: set[int] = set()
    for side in search_orders:
        for idx in range(len(operations) - 1, -1, -1):
            if idx in seen:
                continue
            op = operations[idx]
            sym = app_module._normalize_symbol(op.get("symbol", ""))
            if sym not in symbols:
                continue
            if side and str(op.get("side", "")).upper() != side:
                continue
            seen.add(idx)
            current = pd.to_numeric(op.get("result_pct"), errors="coerce")
            if explicit_override or pd.isna(current):
                operations[idx]["result_pct"] = float(result_pct)
                return operations
    return operations


def _rebuild_operations_from_messages(messages: list[dict[str, Any]], review_date: str) -> list[dict[str, Any]]:
    rebuilt: list[dict[str, Any]] = []
    recent_symbols: list[str] = []
    for msg in messages:
        if str(msg.get("role", "")).strip().lower() != "user":
            continue
        text = str(msg.get("text", "") or "").strip()
        if not text:
            continue
        explicit_symbols = _extract_symbols_from_text(text)
        active_symbols = explicit_symbols[:] if explicit_symbols else recent_symbols[:]
        extracted = _extract_operations_from_text(text, review_date, fallback_symbols=active_symbols)
        if extracted:
            rebuilt = _merge_operations(rebuilt, extracted)
            active_symbols = []
            for op in extracted:
                sym = app_module._normalize_symbol(op.get("symbol", ""))
                if sym and sym not in active_symbols:
                    active_symbols.append(sym)
        result_hint, explicit_override = _extract_result_hint(text)
        if result_hint is not None and active_symbols:
            rebuilt = _apply_result_hint_to_operations(
                rebuilt,
                symbols=active_symbols,
                result_pct=float(result_hint),
                explicit_override=explicit_override,
                prefer_sell=bool(re.search(r"(卖出|卖掉|卖了|止盈|止损|清仓|出了|亏|跌|回撤|盈亏|亏损|盈利|浮盈|浮亏)", text)),
            )
        if explicit_symbols:
            recent_symbols = explicit_symbols[:]
        elif active_symbols:
            recent_symbols = active_symbols[:]
    return rebuilt


def _hydrate_review_day_from_messages(day: dict[str, Any], review_date: str) -> bool:
    messages = day.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return False
    rebuilt = _rebuild_operations_from_messages(messages, review_date)
    if not rebuilt:
        return False
    existing = day.get("operations", [])
    def _sig(ops: list[dict[str, Any]]) -> list[tuple[str, str, Any, Any]]:
        out: list[tuple[str, str, Any, Any]] = []
        for op in ops:
            out.append(
                (
                    app_module._normalize_symbol(op.get("symbol", "")),
                    str(op.get("side", "")).upper(),
                    None if pd.isna(pd.to_numeric(op.get("price"), errors="coerce")) else float(pd.to_numeric(op.get("price"), errors="coerce")),
                    None if pd.isna(pd.to_numeric(op.get("result_pct"), errors="coerce")) else float(pd.to_numeric(op.get("result_pct"), errors="coerce")),
                )
            )
        return out
    if existing and _sig(existing) == _sig(rebuilt):
        return False
    day["operations"] = rebuilt
    card = _build_review_card(review_date, rebuilt)
    day["draft_card"] = card
    day["card"] = card
    day["error_tag"] = str(card.get("error_tag", "") or "")
    day["updated_at"] = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    return True


def _merge_operations(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [x for x in existing if isinstance(x, dict)]
    by_id = {str(x.get("id", "")): x for x in merged}
    for op in incoming:
        oid = str(op.get("id", "")).strip()
        if not oid:
            continue
        by_id[oid] = op
    out = list(by_id.values())
    out.sort(key=lambda x: str(x.get("created_at", "")))
    return out


def _load_symbol_market_snapshot(symbol: str, review_date: str) -> dict[str, Any]:
    sym = app_module._normalize_symbol(symbol)
    if not sym:
        return {"symbol": symbol}
    hist_path = Path("./data/manual_hist") / f"{sym}.csv"
    if not hist_path.exists():
        return {"symbol": sym}
    try:
        df = pd.read_csv(hist_path)
    except Exception:
        return {"symbol": sym}
    if df.empty or "date" not in df.columns:
        return {"symbol": sym}
    dfx = df.copy()
    dfx["date"] = dfx["date"].astype(str).str.slice(0, 10)
    dfx = dfx.sort_values("date").reset_index(drop=True)
    dfx = dfx[dfx["date"] <= review_date]
    if dfx.empty:
        return {"symbol": sym}
    for c in ("close", "high", "low"):
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    dfx["ma5"] = dfx["close"].rolling(5, min_periods=2).mean()
    dfx["ma20"] = dfx["close"].rolling(20, min_periods=5).mean()
    prev_close = dfx["close"].shift(1)
    tr1 = (dfx["high"] - dfx["low"]).abs()
    tr2 = (dfx["high"] - prev_close).abs()
    tr3 = (dfx["low"] - prev_close).abs()
    dfx["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    dfx["atr14"] = dfx["tr"].rolling(14, min_periods=5).mean()
    dfx["ret"] = dfx["close"].pct_change()
    dfx["vol20"] = dfx["ret"].rolling(20, min_periods=5).std()
    row = dfx.iloc[-1]
    close = pd.to_numeric(row.get("close"), errors="coerce")
    prev = pd.to_numeric(dfx.iloc[-2]["close"], errors="coerce") if len(dfx) >= 2 else pd.NA
    pct_chg = None
    if not pd.isna(close) and not pd.isna(prev) and float(prev) != 0:
        pct_chg = (float(close) / float(prev) - 1.0) * 100.0
    atr = pd.to_numeric(row.get("atr14"), errors="coerce")
    atr_pct = None
    if not pd.isna(atr) and not pd.isna(close) and float(close) != 0:
        atr_pct = float(atr) / float(close) * 100.0
    vol20 = pd.to_numeric(row.get("vol20"), errors="coerce")
    vol20_pct = None if pd.isna(vol20) else float(vol20) * 100.0
    out = {
        "symbol": sym,
        "date": str(row.get("date", "")),
        "close": None if pd.isna(close) else float(close),
        "pct_chg": pct_chg,
        "ma5": None if pd.isna(row.get("ma5")) else float(row.get("ma5")),
        "ma20": None if pd.isna(row.get("ma20")) else float(row.get("ma20")),
        "atr_pct": atr_pct,
        "vol20_pct": vol20_pct,
    }
    return out


def _pick_error_tag(operations: list[dict[str, Any]], market_ctx: dict[str, dict[str, Any]]) -> str:
    for op in operations:
        side = str(op.get("side", "")).upper()
        if side != "BUY":
            continue
        sym = app_module._normalize_symbol(op.get("symbol", ""))
        px = pd.to_numeric(op.get("price"), errors="coerce")
        snap = market_ctx.get(sym, {})
        ma20 = pd.to_numeric(snap.get("ma20"), errors="coerce")
        if not pd.isna(px) and not pd.isna(ma20) and float(px) > float(ma20) * 1.05:
            return "追涨"
    neg_ops = [x for x in operations if pd.to_numeric(x.get("result_pct"), errors="coerce") < -2.0]
    if neg_ops:
        return "无止损"
    small_win_sell = [
        x
        for x in operations
        if str(x.get("side", "")).upper() == "SELL"
        and 0 <= float(pd.to_numeric(x.get("result_pct"), errors="coerce") or 0) <= 1.0
    ]
    if small_win_sell:
        return "提前止盈"
    return "执行偏差"


def _month_error_stats(journal: dict[str, Any], month: str) -> list[dict[str, Any]]:
    days = journal.get("days", {}) if isinstance(journal, dict) else {}
    counter: Counter[str] = Counter()
    for d, item in days.items():
        if not str(d).startswith(month):
            continue
        tag = str((item or {}).get("error_tag", "") or "").strip()
        if tag:
            counter[tag] += 1
    return [{"tag": k, "count": int(v)} for k, v in counter.most_common(3)]


def _build_review_card(
    review_date: str,
    operations: list[dict[str, Any]],
    rows: list[dict[str, Any]] | None = None,
    model_top: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    uniq_symbols: list[str] = []
    for op in operations:
        sym = app_module._normalize_symbol(op.get("symbol", ""))
        if sym and sym not in uniq_symbols:
            uniq_symbols.append(sym)
    market_ctx = {sym: _load_symbol_market_snapshot(sym, review_date) for sym in uniq_symbols[:20]}

    op_cards: list[dict[str, Any]] = []
    for op in operations:
        sym = app_module._normalize_symbol(op.get("symbol", ""))
        snap = market_ctx.get(sym, {})
        op_cards.append(
            {
                "id": str(op.get("id", "")),
                "symbol": sym,
                "side": str(op.get("side", "")).upper(),
                "price": op.get("price"),
                "qty": op.get("qty"),
                "result_pct": op.get("result_pct"),
                "reason": str(op.get("reason", "") or ""),
                "close": snap.get("close"),
                "ma5": snap.get("ma5"),
                "ma20": snap.get("ma20"),
                "atr_pct": snap.get("atr_pct"),
                "vol20_pct": snap.get("vol20_pct"),
                "pct_chg": snap.get("pct_chg"),
            }
        )

    right_points: list[str] = []
    wrong_points: list[str] = []
    for op in op_cards:
        ret = pd.to_numeric(op.get("result_pct"), errors="coerce")
        if pd.isna(ret):
            continue
        if float(ret) >= 1.5:
            right_points.append(f"{op.get('symbol')} 执行结果 +{float(ret):.2f}%")
        if float(ret) <= -1.5:
            wrong_points.append(f"{op.get('symbol')} 回撤 {float(ret):.2f}%")
    if not right_points:
        right_points.append("有记录并复盘到具体标的，执行复盘流程合格")
    if not wrong_points:
        wrong_points.append("未见明显硬错误，继续关注入场一致性与止损纪律")

    error_tag = _pick_error_tag(operations, market_ctx)
    if error_tag == "追涨":
        wrong_points.append("存在偏离均线较大的追价行为")
    elif error_tag == "无止损":
        wrong_points.append("亏损单止损执行不够明确")
    elif error_tag == "提前止盈":
        wrong_points.append("盈利单存在过早离场迹象")

    tomorrow_plan = [
        "入场触发：仅在价格靠近MA20（±1%）且5分钟量比<=1.2时考虑开仓。",
        "回避触发：当日涨幅超过5%且价格偏离MA5超过3%时不追。",
        "风控触发：开仓后跌破MA20或单笔亏损达到-2%立即减仓/离场。",
    ]

    if rows:
        buy_rows = [r for r in rows if str(r.get("action", "")).upper() == "BUY"]
        if buy_rows:
            picks = "、".join(str(x.get("symbol", "")) for x in buy_rows[:3] if x.get("symbol"))
            if picks:
                right_points.append(f"规则侧候选：{picks}")
    if model_top:
        tops = "、".join(str(x.get("symbol", "")) for x in (model_top or [])[:3] if x.get("symbol"))
        if tops:
            right_points.append(f"模型Top：{tops}")

    return {
        "review_date": review_date,
        "operations": op_cards,
        "right_points": right_points[:5],
        "wrong_points": wrong_points[:5],
        "error_tag": error_tag,
        "tomorrow_plan": tomorrow_plan[:3],
        "market_context": list(market_ctx.values())[:20],
        "generated_at": dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    }


def _write_monitor_config(review_date: str, card: dict[str, Any]) -> str:
    plans = card.get("tomorrow_plan") or []
    rules = []
    for i, text in enumerate(plans, start=1):
        t = str(text or "").strip()
        if not t:
            continue
        rid = hashlib.md5(f"{review_date}|{t}|{i}".encode("utf-8")).hexdigest()[:12]
        rules.append({"id": rid, "enabled": True, "rule_text": t, "source": "review_assistant"})
    payload = {
        "updated_at": dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "review_date": review_date,
        "rules": rules,
    }
    _REVIEW_MONITOR_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REVIEW_MONITOR_CFG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(_REVIEW_MONITOR_CFG_PATH)


def _format_market_context_for_chat(market_context: list[dict[str, Any]], *, limit: int = 3) -> str:
    parts: list[str] = []
    for x in market_context[:limit]:
        sym = str(x.get("symbol", "") or "")
        close = pd.to_numeric(x.get("close"), errors="coerce")
        ma20 = pd.to_numeric(x.get("ma20"), errors="coerce")
        atr_pct = pd.to_numeric(x.get("atr_pct"), errors="coerce")
        if not sym or pd.isna(close):
            continue
        seg = f"{sym} 收{float(close):.2f}"
        if not pd.isna(ma20):
            seg += f"/MA20 {float(ma20):.2f}"
        if not pd.isna(atr_pct):
            seg += f"/波动{float(atr_pct):.2f}%"
        parts.append(seg)
    return "；".join(parts)


def _build_chat_context_payload(
    *,
    review_date: str,
    operations: list[dict[str, Any]],
    extracted_ops: list[dict[str, Any]],
    market_context: list[dict[str, Any]],
    card: dict[str, Any],
    month_stats: list[dict[str, Any]],
    note_text: str,
) -> dict[str, Any]:
    ops_summary: list[dict[str, Any]] = []
    for op in operations[-8:]:
        ops_summary.append(
            {
                "symbol": str(op.get("symbol", "") or ""),
                "side": str(op.get("side", "") or "").upper(),
                "price": op.get("price"),
                "qty": op.get("qty"),
                "result_pct": op.get("result_pct"),
                "time": str(op.get("time", "") or ""),
                "reason": str(op.get("reason", "") or ""),
            }
        )
    market_summary: list[dict[str, Any]] = []
    for snap in market_context[:5]:
        market_summary.append(
            {
                "symbol": str(snap.get("symbol", "") or ""),
                "date": str(snap.get("date", "") or ""),
                "close": snap.get("close"),
                "pct_chg": snap.get("pct_chg"),
                "ma20": snap.get("ma20"),
                "atr_pct": snap.get("atr_pct"),
            }
        )
    return {
        "review_date": review_date,
        "newly_recorded_operations": len(extracted_ops),
        "operation_count": len(operations),
        "operations": ops_summary,
        "market_context": market_summary,
        "card": {
            "right_points": (card.get("right_points") or [])[:3],
            "wrong_points": (card.get("wrong_points") or [])[:3],
            "error_tag": str(card.get("error_tag", "") or ""),
            "tomorrow_plan": (card.get("tomorrow_plan") or [])[:3],
        },
        "month_error_stats": month_stats[:3],
        "note_text": note_text[:200],
    }


def _build_local_chat_reply(
    *,
    user_text: str,
    review_date: str,
    extracted_ops: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    market_context: list[dict[str, Any]],
) -> str:
    text = str(user_text or "").strip()
    market_text = _format_market_context_for_chat(market_context)
    has_missing_result = any(pd.isna(pd.to_numeric(x.get("result_pct"), errors="coerce")) for x in operations)
    op_count = len(operations)

    if extracted_ops:
        lines = [f"已经记下 {len(extracted_ops)} 条操作。"]
        if market_text:
            lines.append(f"我顺手把行情也对上了：{market_text}。")
        if has_missing_result:
            lines.append("还差这些交易的大致盈亏幅度，你补个百分比我就能继续归因。")
        else:
            lines.append("你可以继续补充当时为什么买卖、仓位怎么想、情绪有没有波动，我再帮你拆执行问题。")
        return "".join(lines)

    if re.search(r"(你是真人吗|你不是真人|你是谁|你在干嘛|在吗|能聊天吗)", text):
        if op_count > 0:
            return f"不是人工，但我现在能正常接着聊，也会顺手记你的复盘。今天 {review_date} 已经记了 {op_count} 笔操作，想聊哪一笔都行。"
        return "不是人工，我是你的复盘助手。现在可以正常聊天；你想闲聊、吐槽，或者直接开始复盘都可以。"

    if re.search(r"(烦|好烦|郁闷|难受|崩|焦虑|心态炸|亏麻了|亏惨了)", text):
        if op_count > 0:
            return "先别急，我们可以按聊天的方式拆。你先说今天最难受的那一笔，我结合已经记下的交易帮你看，是节奏问题、仓位问题，还是卖点问题。"
        return "能聊。你先把最卡的点直接说出来就行，我先陪你把情绪和问题拆开；如果要转复盘，后面再补股票和交易细节。"

    if re.fullmatch(r"[?？!！。.]+", text):
        if op_count > 0:
            return f"我在，这边已经保留了你 {review_date} 的复盘记录。目前有 {op_count} 笔操作，你继续说，我会接着聊，不会只按模板回。"
        return "我在。你继续说就行，我会按正常聊天接，不会强行把每句话都当成复盘指令。"

    if market_text:
        if has_missing_result:
            return f"我先把你提到的内容和行情对上了：{market_text}。如果这是在复盘，下一步最关键的是补上盈亏幅度，我才能更准地判断执行问题。"
        return f"我先把相关内容对上了：{market_text}。如果你想继续复盘，可以接着说买卖理由、仓位变化，或者当时为什么犹豫。"

    if op_count > 0:
        if has_missing_result:
            return "我在跟着你的上下文。现在已经有交易记录了，但还差部分盈亏结果；你补一下大概百分比，我就能继续做归因。"
        return f"我在听。今天 {review_date} 已经有 {op_count} 笔交易记录了，你可以继续像聊天一样讲经过，我会边聊边整理复盘。"

    return "我在。你可以正常聊，也可以随时切回复盘；如果要开始复盘，直接把今天的买卖、价格、盈亏和理由按你习惯的话说出来就行。"


def _generate_chat_reply(
    *,
    user_text: str,
    review_date: str,
    day: dict[str, Any],
    extracted_ops: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    market_context: list[dict[str, Any]],
    card: dict[str, Any],
    month_stats: list[dict[str, Any]],
    cfg: dict,
    proxy: str = "",
    use_proxy: bool = False,
) -> tuple[str, str]:
    fallback = _build_local_chat_reply(
        user_text=user_text,
        review_date=review_date,
        extracted_ops=extracted_ops,
        operations=operations,
        market_context=market_context,
    )
    deep_cfg = cfg.get("deepseek", {}) or {}
    api_key = _load_deepseek_key()
    if not deep_cfg.get("enabled", False) or not api_key:
        return fallback, "local"

    history = []
    for msg in (day.get("messages") or [])[-12:]:
        role = str(msg.get("role", "") or "").strip().lower()
        content = str(msg.get("text", "") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content})

    context_payload = _build_chat_context_payload(
        review_date=review_date,
        operations=operations,
        extracted_ops=extracted_ops,
        market_context=market_context,
        card=card,
        month_stats=month_stats,
        note_text=str(day.get("note_text", "") or ""),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是A股短线复盘助手，同时也是一个可以正常多轮闲聊的中文助手。"
                "优先自然回应用户当前话题，不要每次都强行拉回股票代码或复盘流程。"
                "当用户在表达情绪、确认你是谁、闲聊、追问你在做什么时，直接像正常聊天一样回应。"
                "当用户提到股票、买卖、仓位、盈亏、纪律、复盘时，再结合上下文给出复盘式分析。"
                "如果复盘信息不完整，只追问1个最关键的问题；如果信息已足够，就给分析或下一步建议。"
                "回复要求：中文，口语化，简洁，2到5句，不用Markdown列表，不编造持仓或行情。"
            ),
        },
        {
            "role": "system",
            "content": f"当天复盘上下文：{json.dumps(context_payload, ensure_ascii=False)}",
        },
        *history,
    ]
    payload = {
        "model": str(deep_cfg.get("model", "deepseek-chat") or "deepseek-chat"),
        "messages": messages,
        "temperature": max(0.2, float(deep_cfg.get("temperature", 0.2) or 0.2)),
        "max_tokens": max(220, int(deep_cfg.get("max_tokens", 300) or 300)),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    proxies = {"http": proxy, "https": proxy} if use_proxy and proxy else None
    try:
        resp = requests.post(
            _resolve_deepseek_url(str(deep_cfg.get("base_url", "https://api.deepseek.com"))),
            headers=headers,
            json=payload,
            timeout=25,
            proxies=proxies,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        text = str(text or "").strip()
        if not text:
            return fallback, "local"
        return text, "deepseek"
    except Exception:
        return fallback, "local"


def _build_review_snapshot(req: ReviewRequest) -> dict[str, Any]:
    rows = req.rows or []
    model_top = req.model_top or []
    buy_rows = [r for r in rows if str(r.get("action", "")).upper() == "BUY"]
    watch_rows = [r for r in rows if str(r.get("action", "")).upper() == "WATCH"]
    top_watchlist = []
    for item in (req.watchlist or [])[:10]:
        top_watchlist.append(
            {
                "symbol": str(item.get("symbol", "") or ""),
                "name": str(item.get("name", "") or ""),
            }
        )
    operations = req.operations or []
    buy_ops = [x for x in operations if str(x.get("side", "")).upper() == "BUY"]
    sell_ops = [x for x in operations if str(x.get("side", "")).upper() == "SELL"]
    res_vals = [pd.to_numeric(x.get("result_pct"), errors="coerce") for x in operations]
    res_vals = [float(x) for x in res_vals if not pd.isna(x)]
    avg_result_pct = float(sum(res_vals) / len(res_vals)) if res_vals else None

    return {
        "mode": req.mode,
        "review_date": _norm_review_date(req.review_date),
        "buy_count": len(buy_rows),
        "watch_count": len(watch_rows),
        "row_count": len(rows),
        "notes": (req.notes or [])[:10],
        "news_summary": req.news_summary or {},
        "top_buy": _review_top_rows(buy_rows, limit=5),
        "top_watch": _review_top_rows(watch_rows, limit=5),
        "top_model": _review_top_rows(model_top, limit=5),
        "watchlist": top_watchlist,
        "operations": operations,
        "operation_count": len(operations),
        "buy_ops": len(buy_ops),
        "sell_ops": len(sell_ops),
        "avg_result_pct": avg_result_pct,
        "note_text": str(req.note_text or ""),
    }


def _local_review_text(snapshot: dict[str, Any]) -> str:
    news = snapshot.get("news_summary") or {}
    lines: list[str] = []
    lines.append(f"复盘日期：{snapshot.get('review_date', _today_cn_str())}。")
    lines.append(
        f"盘面信号：规则候选 {snapshot.get('row_count', 0)} 只，BUY {snapshot.get('buy_count', 0)} 只，WATCH {snapshot.get('watch_count', 0)} 只。"
    )
    market_sentiment = str(news.get("market_sentiment", "") or "neutral")
    risk_level = str(news.get("risk_level", "") or "medium")
    hot_sectors = "、".join(news.get("hot_sectors") or []) or "暂无"
    lines.append(f"市场情绪 {market_sentiment}，风险等级 {risk_level}，热点板块 {hot_sectors}。")
    op_count = int(snapshot.get("operation_count", 0) or 0)
    if op_count > 0:
        avg_result_pct = snapshot.get("avg_result_pct")
        if avg_result_pct is None:
            lines.append(f"执行记录：共 {op_count} 笔（买入 {snapshot.get('buy_ops', 0)} / 卖出 {snapshot.get('sell_ops', 0)}）。")
        else:
            lines.append(
                f"执行记录：共 {op_count} 笔（买入 {snapshot.get('buy_ops', 0)} / 卖出 {snapshot.get('sell_ops', 0)}），平均结果 {float(avg_result_pct):.2f}%。"
            )
    else:
        lines.append("执行记录为空：建议先补充当日买卖记录，再做行为复盘。")
    top_buy = snapshot.get("top_buy") or []
    if top_buy:
        picks = "；".join(
            f"{r.get('symbol')} {r.get('name')}({r.get('sector') or '--'}) score={float(r.get('score') or 0):.2f}"
            for r in top_buy[:3]
        )
        lines.append(f"规则侧最强信号：{picks}。")
    else:
        lines.append("规则侧没有明确 BUY，说明当前日线条件或分钟确认偏严格。")
    top_model = snapshot.get("top_model") or []
    if top_model:
        picks = "；".join(
            f"{r.get('symbol')} {r.get('name')} 模型={float(r.get('model_score') or 0) * 100:.2f}%"
            for r in top_model[:3]
        )
        lines.append(f"模型侧关注：{picks}。")
    if snapshot.get("watchlist"):
        wl = "、".join(f"{x.get('symbol')} {x.get('name')}".strip() for x in snapshot["watchlist"][:5])
        lines.append(f"自选观察池：{wl}。")
    notes = snapshot.get("notes") or []
    if notes:
        lines.append(f"系统备注：{'；'.join(str(x) for x in notes[:3])}。")
    note_text = str(snapshot.get("note_text", "") or "").strip()
    if note_text:
        lines.append(f"当日笔记：{note_text[:120]}。")
    lines.append("明确指导：明天优先选择规则与模型共振、且所属热点板块一致的标的，单笔先按半仓试错；若开盘后分钟确认未通过，直接放弃追单。")
    lines.append("复盘笔记建议：记录1条做对的执行、1条做错的执行、1条明日必须遵守的纪律。")
    return "".join(lines)


def _resolve_deepseek_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _generate_review_text(snapshot: dict[str, Any], cfg: dict, *, proxy: str = "", use_proxy: bool = False) -> tuple[str, str]:
    deep_cfg = cfg.get("deepseek", {}) or {}
    api_key = _load_deepseek_key()
    if not deep_cfg.get("enabled", False) or not api_key:
        return _local_review_text(snapshot), "local"

    prompt = (
        "你是A股短线复盘助手。请基于给定结构化快照，输出盘后复盘报告。"
        "必须包含四段："
        "1) 今日执行评估（结合操作记录），"
        "2) 信号与盘面总结（规则侧+模型侧+情绪），"
        "3) 明日明确指导（给出3条可执行动作，含入场/回避/仓位），"
        "4) 复盘笔记（给出可直接复制的三行笔记模板）。"
        "要求：不用Markdown列表，不编造数据，不输出免责声明，总长度控制在220~360字。"
        f"\n\n快照:{json.dumps(snapshot, ensure_ascii=False)}"
    )
    payload = {
        "model": str(deep_cfg.get("model", "deepseek-chat") or "deepseek-chat"),
        "messages": [
            {"role": "system", "content": "你是严谨的量化复盘助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(deep_cfg.get("temperature", 0.2)),
        "max_tokens": max(300, int(deep_cfg.get("max_tokens", 300) or 300)),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    proxies = {"http": proxy, "https": proxy} if use_proxy and proxy else None
    try:
        resp = requests.post(
            _resolve_deepseek_url(str(deep_cfg.get("base_url", "https://api.deepseek.com"))),
            headers=headers,
            json=payload,
            timeout=25,
            proxies=proxies,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        text = str(text or "").strip()
        if not text:
            return _local_review_text(snapshot), "local"
        return text, "deepseek"
    except Exception:
        return _local_review_text(snapshot), "local"


@app.post("/api/review")
def review(req: ReviewRequest) -> dict:
    cfg = app_module.load_config("config/default.yaml")
    signals_cfg = cfg.get("signals", {}) or {}
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    if not proxy and app_module._get_env_proxy():
        use_proxy = True
        proxy = app_module._get_env_proxy()
    elif use_proxy and not proxy:
        detected = app_module._detect_proxy()
        if detected:
            proxy = detected

    review_date = _norm_review_date(req.review_date)
    with _REVIEW_JOURNAL_LOCK:
        journal = _load_review_journal()
        day = _ensure_review_day(journal, review_date)
        operations = req.operations if req.operations is not None else day.get("operations", [])
        note_text = req.note_text if req.note_text is not None else str(day.get("note_text", "") or "")

        req_base = req.dict() if hasattr(req, "dict") else req.model_dump()
        req_base.update({"review_date": review_date, "operations": operations, "note_text": note_text})
        req_for_snapshot = ReviewRequest(**req_base)
        snapshot = _build_review_snapshot(req_for_snapshot)
        text, source = _generate_review_text(snapshot, cfg, proxy=proxy, use_proxy=use_proxy)

        card = _build_review_card(review_date, operations or [], rows=req.rows, model_top=req.model_top)
        monitor_path = _write_monitor_config(review_date, card)

        day["operations"] = [_normalize_operation(x) for x in (operations or []) if isinstance(x, dict)]
        day["card"] = card
        day["draft_card"] = card
        day["error_tag"] = str(card.get("error_tag", "") or "")
        day["note_text"] = note_text
        day["review_text"] = text
        day["review_source"] = source
        day["snapshot"] = snapshot
        day["updated_at"] = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        _save_review_journal(journal)

    return {
        "review": text,
        "source": source,
        "snapshot": snapshot,
        "card": card,
        "monitor_config": monitor_path,
    }


@app.post("/api/review/chat")
def review_chat(payload: dict = Body(...)) -> dict:
    review_date = _norm_review_date(payload.get("date"))
    text = str(payload.get("text", "") or "").strip()
    if not text:
        return {"error": "text required"}
    cfg = app_module.load_config("config/default.yaml")
    signals_cfg = cfg.get("signals", {}) or {}
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    if not proxy and app_module._get_env_proxy():
        use_proxy = True
        proxy = app_module._get_env_proxy()
    elif use_proxy and not proxy:
        detected = app_module._detect_proxy()
        if detected:
            proxy = detected

    with _REVIEW_JOURNAL_LOCK:
        journal = _load_review_journal()
        day = _ensure_review_day(journal, review_date)

        user_msg = _normalize_chat_message({"role": "user", "text": text})
        day["messages"].append(user_msg)

        recent_symbols: list[str] = []
        for msg in reversed(day.get("messages", [])[:-1]):
            if str(msg.get("role", "")).strip().lower() != "user":
                continue
            recent_symbols = _extract_symbols_from_text(str(msg.get("text", "") or "").strip())
            if recent_symbols:
                break
        extracted_ops = _extract_operations_from_text(text, review_date, fallback_symbols=recent_symbols)
        day["operations"] = _rebuild_operations_from_messages(day.get("messages", []), review_date)

        symbols = _extract_symbols_from_text(text)
        if not symbols:
            for op in day.get("operations", []):
                sym = app_module._normalize_symbol(op.get("symbol", ""))
                if sym and sym not in symbols:
                    symbols.append(sym)
                if len(symbols) >= 5:
                    break
        market_context = [_load_symbol_market_snapshot(sym, review_date) for sym in symbols[:5]]
        day["draft_card"] = _build_review_card(review_date, day.get("operations", []))
        month = review_date[:7]
        month_stats = _month_error_stats(journal, month)

        reply, source = _generate_chat_reply(
            user_text=text,
            review_date=review_date,
            day=day,
            extracted_ops=extracted_ops,
            operations=day.get("operations", []),
            market_context=market_context,
            card=day.get("draft_card", {}),
            month_stats=month_stats,
            cfg=cfg,
            proxy=proxy,
            use_proxy=use_proxy,
        )
        assistant_msg = _normalize_chat_message({"role": "assistant", "text": reply})
        day["messages"].append(assistant_msg)
        day["chat_source"] = source
        day["updated_at"] = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        _save_review_journal(journal)

        return {
            "date": review_date,
            "messages": day.get("messages", []),
            "operations": day.get("operations", []),
            "card": day.get("draft_card", {}),
            "error_stats_month": month_stats,
            "source": source,
        }


@app.get("/api/review/journal")
def get_review_journal(review_date: str | None = Query(None)) -> dict:
    d = _norm_review_date(review_date)
    with _REVIEW_JOURNAL_LOCK:
        journal = _load_review_journal()
        day = _ensure_review_day(journal, d)
        if _hydrate_review_day_from_messages(day, d):
            _save_review_journal(journal)
        month = d[:7]
        month_stats = _month_error_stats(journal, month)
        return {
            "date": d,
            "messages": day.get("messages", []),
            "operations": day.get("operations", []),
            "note_text": day.get("note_text", ""),
            "card": day.get("card") or day.get("draft_card") or {},
            "error_tag": day.get("error_tag", ""),
            "error_stats_month": month_stats,
        }


@app.post("/api/review/operation")
def add_review_operation(payload: dict = Body(...)) -> dict:
    d = _norm_review_date(payload.get("date"))
    op_raw = payload.get("operation") or {}
    op = _normalize_operation(op_raw)
    with _REVIEW_JOURNAL_LOCK:
        journal = _load_review_journal()
        day = _ensure_review_day(journal, d)
        day["operations"] = [x for x in day.get("operations", []) if str(x.get("id", "")) != op["id"]]
        day["operations"].append(op)
        day["updated_at"] = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        _save_review_journal(journal)
        return {"date": d, "operation": op, "operations": day["operations"]}


@app.delete("/api/review/operation")
def delete_review_operation(payload: dict = Body(...)) -> dict:
    d = _norm_review_date(payload.get("date"))
    op_id = str(payload.get("id", "") or "").strip()
    if not op_id:
        return {"error": "id required"}
    with _REVIEW_JOURNAL_LOCK:
        journal = _load_review_journal()
        day = _ensure_review_day(journal, d)
        before = len(day.get("operations", []))
        day["operations"] = [x for x in day.get("operations", []) if str(x.get("id", "")) != op_id]
        removed = before - len(day["operations"])
        day["updated_at"] = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        _save_review_journal(journal)
        return {"date": d, "removed": removed, "operations": day["operations"]}


@app.post("/api/review/note")
def save_review_note(payload: dict = Body(...)) -> dict:
    d = _norm_review_date(payload.get("date"))
    note_text = str(payload.get("note_text", "") or "")
    with _REVIEW_JOURNAL_LOCK:
        journal = _load_review_journal()
        day = _ensure_review_day(journal, d)
        day["note_text"] = note_text
        day["updated_at"] = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        _save_review_journal(journal)
        return {"date": d, "note_text": note_text}


@app.post("/api/watchlist")
def add_watchlist(payload: dict = Body(...)) -> dict:
    cfg = app_module.load_config("config/default.yaml")
    watchlist_file = str(cfg.get("watchlist_file", "./data/watchlist.csv"))
    symbols = payload.get("symbols") or []
    new_syms = [app_module._normalize_symbol(s) for s in symbols]
    new_syms = [s for s in new_syms if s]
    if not new_syms:
        return {"added": [], "symbols": app_module.load_watchlist(watchlist_file)}
    watchlist = app_module.load_watchlist(watchlist_file)
    merged = watchlist + [s for s in new_syms if s not in set(watchlist)]
    app_module.save_watchlist(watchlist_file, merged)
    return {"added": new_syms, "symbols": merged}


@app.delete("/api/watchlist")
def delete_watchlist(payload: dict = Body(...)) -> dict:
    cfg = app_module.load_config("config/default.yaml")
    watchlist_file = str(cfg.get("watchlist_file", "./data/watchlist.csv"))
    symbols = payload.get("symbols") or []
    if symbols == "*":
        app_module.save_watchlist(watchlist_file, [])
        return {"removed": "*", "symbols": []}
    remove_syms = {app_module._normalize_symbol(s) for s in symbols}
    remove_syms = {s for s in remove_syms if s}
    watchlist = app_module.load_watchlist(watchlist_file)
    remain = [s for s in watchlist if s not in remove_syms]
    app_module.save_watchlist(watchlist_file, remain)
    return {"removed": list(remove_syms), "symbols": remain}
