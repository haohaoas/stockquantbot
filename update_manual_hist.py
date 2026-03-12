from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from backtest_optimize import load_config, load_universe_info, load_trade_calendar, pick_trade_days
from sqdata.akshare_fetcher import fetch_hist
from sqdata.tencent_fetcher import fetch_realtime
from sqdata.universe import filter_symbols_by_market, filter_symbols_by_board


def _normalize_hist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
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


def _read_hist(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return _normalize_hist(df)


def _read_last_date_fast(path: Path) -> str:
    """Fast path: read latest date from csv tail without loading full file."""
    if not path.exists() or path.stat().st_size <= 0:
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            pos = max(0, size - 4096)
            f.seek(pos)
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        if not lines:
            return ""
        # Skip potential header line and scan from bottom.
        for line in reversed(lines):
            head = line.split(",", 1)[0].strip().strip('"').strip("'")
            # Accept YYYY-MM-DD or YYYY-MM-DD HH:MM:SS prefix.
            if len(head) >= 10:
                cand = head[:10]
                try:
                    dt.date.fromisoformat(cand)
                    return cand
                except Exception:
                    continue
    except Exception:
        return ""
    return ""


def _merge_hist(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    out = pd.concat([old, new], ignore_index=True)
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def _covers_range(df: pd.DataFrame, start_date: str, end_date: str) -> bool:
    if df is None or df.empty:
        return False
    try:
        min_d = df["date"].min()
        max_d = df["date"].max()
    except Exception:
        return False
    return bool(min_d <= start_date and max_d >= end_date)


def _resolve_dates(trade_days: list[str], years: int, end_date: str | None) -> tuple[str, str]:
    today = dt.date.today()
    if trade_days:
        trade_days = [d for d in trade_days if d <= today.isoformat()]
    if end_date:
        end_dt = dt.date.fromisoformat(end_date)
    elif trade_days:
        end_dt = dt.date.fromisoformat(trade_days[-1])
    else:
        end_dt = today
    days = pick_trade_days(trade_days, end_dt, years) if trade_days else []
    if days:
        start = days[0]
        end = days[-1]
    else:
        start = (end_dt - dt.timedelta(days=365 * years)).isoformat()
        end = end_dt.isoformat()
    return start, end


def _is_market_open_cn(now: dt.datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return (570 <= t <= 690) or (780 <= t <= 900)


def _remove_file(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        return False
    return False


def _clear_post_update_disk_caches(cfg: dict) -> dict[str, int]:
    signals_cfg = cfg.get("signals", {}) or {}
    model_ref_cfg = signals_cfg.get("model_ref", {}) or {}
    removed = {"preselect": 0, "model_scores": 0}

    preselect_cache_file = str(signals_cfg.get("preselect_cache_file", "./cache/preselect_symbols.json") or "").strip()
    if preselect_cache_file and _remove_file(Path(preselect_cache_file)):
        removed["preselect"] = 1

    model_cache_dir = str(model_ref_cfg.get("cache_dir", "./cache/model_scores") or "").strip()
    if model_cache_dir:
        cache_dir = Path(model_cache_dir)
        if cache_dir.exists():
            for fp in cache_dir.glob("*.csv"):
                if _remove_file(fp):
                    removed["model_scores"] += 1
    return removed


def _write_refresh_signal(end_date: str, stats: dict[str, int], *, signal_path: str = "./cache/data_refresh_signal.json") -> None:
    p = Path(signal_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "end_date": str(end_date),
        "stats": {k: int(v) for k, v in stats.items()},
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _post_refresh_prewarm(
    cfg: dict,
    *,
    api_base: str,
    provider: str,
    timeout: int,
) -> tuple[int, int]:
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        return 0, 0

    signals_cfg = cfg.get("signals", {}) or {}
    model_ref_cfg = signals_cfg.get("model_ref") or {}
    model_options = model_ref_cfg.get("options") or {}
    targets: list[str | None] = [None]
    for key in sorted(model_options.keys()):
        k = str(key or "").strip()
        if k:
            targets.append(k)

    session = requests.Session()
    ok = 0
    fail = 0

    urls = [
        f"{base}/api/market?mode=all&provider={provider}&top_n=20&only_buy=false&intraday=false&model_independent=false",
    ]
    for model_key in targets:
        q_model = f"&model={model_key}" if model_key else ""
        urls.append(
            f"{base}/api/model-top?mode=all&provider={provider}&top_n=20{q_model}&model_independent=false"
        )
        urls.append(
            f"{base}/api/model-top?mode=all&provider={provider}&top_n=20{q_model}&model_independent=true"
        )

    def _hit(url: str) -> bool:
        try:
            resp = session.get(url, timeout=max(5, int(timeout)))
            return bool(resp.ok)
        except Exception:
            return False

    max_workers = min(6, max(1, len(urls)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_hit, url) for url in urls]
        for fut in as_completed(futures):
            try:
                if fut.result():
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
    return ok, fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill manual_hist using Tencent kline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--universe-file", default="", help="Override universe_file (default from config).")
    parser.add_argument("--use-trade-pool", action="store_true", help="Use data/trade_pool.csv if available.")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbols", default="", help="Comma-separated symbols to update (override universe).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=4, help="Per-request timeout seconds for history fetch.")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--manual-hist-dir", default="data/manual_hist")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only-stale",
        dest="only_stale",
        action="store_true",
        default=True,
        help="Only refresh symbols whose latest date < end_date. Enabled by default.",
    )
    parser.add_argument(
        "--full-range",
        dest="only_stale",
        action="store_false",
        help="Disable stale-only mode and refresh by full target range.",
    )
    parser.add_argument(
        "--stale-lookback-days",
        type=int,
        default=0,
        help="When stale-only mode is enabled, backfill from latest_local_date - N days (default: 0).",
    )
    parser.add_argument("--retry-direct", action="store_true", help="Retry without proxy when proxy fetch returns empty.")
    parser.add_argument("--fill-rt", action="store_true", help="Append today's bar from realtime quote if still missing.")
    parser.add_argument("--use-proxy", action="store_true")
    parser.add_argument("--proxy", default="")
    parser.add_argument("--skip-post-refresh", action="store_true", help="Skip cache bust + API prewarm after update.")
    parser.add_argument("--prewarm-api-base", default="http://127.0.0.1:8000", help="Local API base used for post-update prewarm.")
    parser.add_argument("--prewarm-timeout", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    universe_file = str(cfg.get("universe_file", "./data/universe.csv"))
    if args.universe_file:
        universe_file = args.universe_file
    elif args.use_trade_pool:
        trade_pool = Path("data/trade_pool.csv")
        if trade_pool.exists():
            universe_file = str(trade_pool)
    market_scope = cfg.get("market_scope", ["sh", "sz"])
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = bool(cfg.get("exclude_chi_next", False))
    mainboard_only = bool(cfg.get("mainboard_only", True))

    symbols, _ = load_universe_info(universe_file)
    symbols = filter_symbols_by_market(symbols, market_scope)
    symbols = filter_symbols_by_board(
        symbols,
        exclude_star=exclude_star,
        exclude_chi_next=exclude_chi_next,
        mainboard_only=mainboard_only,
    )
    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    trade_calendar_file = str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv"))
    trade_days = load_trade_calendar(trade_calendar_file)
    explicit_end = (args.end_date or "").strip()
    start_date, end_date = _resolve_dates(trade_days, args.years, explicit_end or None)
    if not explicit_end and trade_days:
        now_cn = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        today_iso = now_cn.date().isoformat()
        # Default to last fully-closed trading day, to avoid intraday full-market refresh.
        if end_date == today_iso:
            t_min = now_cn.hour * 60 + now_cn.minute
            if _is_market_open_cn(now_cn) or t_min < (17 * 60):
                prev_days = [d for d in trade_days if d < today_iso]
                if prev_days:
                    end_date = prev_days[-1]
                    start_date, _ = _resolve_dates(trade_days, args.years, end_date)
                    print(f"[info] auto end-date adjusted to last closed trading day: {end_date}")
    # Add a buffer to ensure enough rows for moving averages.
    start_dt = dt.date.fromisoformat(start_date) - dt.timedelta(days=30)
    start_buffer = start_dt.isoformat()

    signals_cfg = cfg.get("signals", {}) or {}
    use_proxy = bool(signals_cfg.get("use_proxy", False)) or bool(args.use_proxy)
    proxy = args.proxy or str(signals_cfg.get("proxy", "") or "")
    if use_proxy and not proxy:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""

    allow_akshare_fallback = bool(cfg.get("allow_eastmoney_fallback", False))

    out_dir = Path(args.manual_hist_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "empty": 0, "error": 0}
    lock = threading.Lock()

    def _log_progress(i: int, total: int) -> None:
        if i % 50 == 0 or i == total:
            print(
                f"[info] {i}/{total} ok={stats['ok']} skip={stats['skip']} empty={stats['empty']} error={stats['error']}"
            )

    def _append_realtime_bar(sym: str, existing: pd.DataFrame) -> pd.DataFrame:
        if existing is None or existing.empty:
            return existing
        try:
            last = str(existing["date"].max())[:10]
        except Exception:
            return existing
        if last >= end_date:
            return existing
        today_cn = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai")).date()
        if end_date != today_cn.isoformat():
            return existing
        now_cn = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        if _is_market_open_cn(now_cn):
            return existing
        try:
            q = fetch_realtime(sym, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=True)
        except Exception:
            return existing
        if not q or q.price <= 0:
            return existing
        row = {
            "date": end_date,
            "open": q.open,
            "close": q.price,
            "high": q.high,
            "low": q.low,
            "volume": q.volume,
            "amount": q.amount,
        }
        add = pd.DataFrame([row])
        merged = _merge_hist(existing, add)
        return merged

    def _fetch_one(sym: str) -> tuple[str, str]:
        sym = str(sym).zfill(6)
        out_path = out_dir / f"{sym}.csv"
        existing: pd.DataFrame | None = None
        fetch_start = start_buffer
        latest_local_date = ""
        stale_append_mode = False
        if not args.force:
            if args.only_stale:
                max_d = _read_last_date_fast(out_path)
                if max_d:
                    try:
                        if max_d >= end_date:
                            return sym, "skip"
                        max_dt = dt.date.fromisoformat(max_d)
                        lookback_days = max(0, int(args.stale_lookback_days))
                        if lookback_days == 0 and not args.fill_rt:
                            # Fast incremental append path: fetch from next day and append only new rows.
                            stale_append_mode = True
                            latest_local_date = max_d
                            fetch_start = max((max_dt + dt.timedelta(days=1)), dt.date.fromisoformat(start_buffer)).isoformat()
                        else:
                            fetch_start = max((max_dt - dt.timedelta(days=lookback_days)), dt.date.fromisoformat(start_buffer)).isoformat()
                    except Exception:
                        fetch_start = start_buffer
            else:
                existing = _read_hist(out_path)
                if _covers_range(existing, start_date, end_date):
                    return sym, "skip"
        try:
            df = fetch_hist(
                sym,
                start=fetch_start,
                end=end_date,
                timeout=max(1, int(args.timeout)),
                use_proxy=use_proxy,
                proxy=proxy,
                allow_akshare_fallback=allow_akshare_fallback,
                manual_hist_dir=None,
            )
        except Exception:
            df = pd.DataFrame()
        df = _normalize_hist(df)
        if (df is None or df.empty) and args.retry_direct and (use_proxy or proxy):
            try:
                df = fetch_hist(
                    sym,
                    start=fetch_start,
                    end=end_date,
                    timeout=max(1, int(args.timeout)),
                    use_proxy=False,
                    proxy="",
                    allow_akshare_fallback=allow_akshare_fallback,
                    manual_hist_dir=None,
                )
            except Exception:
                df = pd.DataFrame()
            df = _normalize_hist(df)
        if df.empty:
            return sym, "empty"
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
        if df.empty:
            return sym, "empty"
        if stale_append_mode:
            df_new = df[df["date"] > latest_local_date].copy() if latest_local_date else df
            if df_new.empty:
                return sym, "skip"
            keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount"] if c in df_new.columns]
            df_new = df_new[keep].sort_values("date").reset_index(drop=True)
            exists_nonempty = out_path.exists() and out_path.stat().st_size > 0
            df_new.to_csv(
                out_path,
                mode="a" if exists_nonempty else "w",
                header=not exists_nonempty,
                index=False,
                encoding="utf-8",
            )
            if args.sleep > 0:
                time.sleep(args.sleep)
            return sym, "ok"
        if existing is None:
            existing = _read_hist(out_path)
        merged = _merge_hist(existing, df)
        if args.fill_rt:
            merged = _append_realtime_bar(sym, merged)
        merged.to_csv(out_path, index=False, encoding="utf-8")
        if args.sleep > 0:
            time.sleep(args.sleep)
        return sym, "ok"

    total = len(symbols)
    if total == 0:
        print("[error] symbol list empty.")
        return

    if args.workers <= 1:
        for i, sym in enumerate(symbols, start=1):
            _, status = _fetch_one(sym)
            with lock:
                stats[status] += 1
            _log_progress(i, total)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_fetch_one, sym): sym for sym in symbols}
            for i, fut in enumerate(as_completed(futures), start=1):
                status = "error"
                try:
                    _, status = fut.result()
                except Exception:
                    status = "error"
                with lock:
                    stats[status] += 1
                _log_progress(i, total)

    print(
        f"[ok] done: ok={stats['ok']} skip={stats['skip']} empty={stats['empty']} error={stats['error']} "
        f"range={start_date}~{end_date}"
    )

    if args.skip_post_refresh:
        return

    removed = _clear_post_update_disk_caches(cfg)
    _write_refresh_signal(end_date, stats)
    print(
        f"[info] cache cleared: preselect={removed['preselect']} model_score_files={removed['model_scores']}"
    )

    provider = str((cfg.get("signals", {}) or {}).get("realtime_provider", "tencent") or "tencent").strip().lower()
    ok, fail = _post_refresh_prewarm(
        cfg,
        api_base=args.prewarm_api_base,
        provider=provider if provider in {"auto", "biying", "tencent", "sina", "netease"} else "tencent",
        timeout=args.prewarm_timeout,
    )
    print(f"[info] post-refresh prewarm: ok={ok} fail={fail} base={args.prewarm_api_base}")


if __name__ == "__main__":
    main()
