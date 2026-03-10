from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import pandas as pd

from app import load_config, load_watchlist, _is_trading_day_cn, _load_trade_calendar, compute_market
from sqdata.calendar import resolve_trade_date
from sqdata.universe import load_universe_symbols, filter_symbols_by_market, filter_symbols_by_board
from sqdata.minute_fetcher import fetch_tencent_minute, fetch_biying_minute_history


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).copy()
        return df
    except Exception:
        return pd.DataFrame()


def _normalize_minute_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        out = out.dropna(subset=["datetime"]).copy()
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    for c in ("open", "close", "high", "low", "volume", "amount"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _merge_minute(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    old_n = _normalize_minute_df(old)
    new_n = _normalize_minute_df(new)
    if old_n.empty:
        return new_n
    if new_n.empty:
        return old_n
    out = pd.concat([old_n, new_n], ignore_index=True)
    if "datetime" in out.columns:
        out = out.drop_duplicates(subset=["datetime"], keep="last")
    return out.sort_values("datetime").reset_index(drop=True)


def _resolve_symbols(args, cfg: dict) -> list[str]:
    if args.symbols:
        return [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]

    if args.source == "model_top":
        top_n = args.top_n or int((cfg.get("output", {}) or {}).get("top_n", 20))
        params = _build_market_params(cfg, top_n)
        result = compute_market(params)
        model_top = result.get("model_top")
        if model_top is not None and not model_top.empty and "symbol" in model_top.columns:
            return [str(s).zfill(6) for s in model_top["symbol"].tolist()]
        return []

    if args.source == "watchlist":
        watchlist_file = args.watchlist_file or str(cfg.get("watchlist_file", "./data/watchlist.csv"))
        if watchlist_file:
            syms = load_watchlist(watchlist_file)
            if syms:
                return [str(s).zfill(6) for s in syms]
        return []

    universe_file = args.universe_file or str(cfg.get("universe_file", "./data/universe.csv"))
    symbols = load_universe_symbols(universe_file)
    symbols = filter_symbols_by_market(symbols, cfg.get("market_scope", ["sh", "sz"]))
    symbols = filter_symbols_by_board(
        symbols,
        exclude_star=bool(cfg.get("exclude_star", True)),
        exclude_chi_next=bool(cfg.get("exclude_chi_next", False)),
        mainboard_only=bool(cfg.get("mainboard_only", True)),
    )
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    return [str(s).zfill(6) for s in symbols]


def _build_market_params(cfg: dict, top_n: int) -> dict:
    universe_cfg = cfg.get("universe", {}) or {}
    signals_cfg = cfg.get("signals", {}) or {}
    weights = cfg.get("score_weights", {}) or {}
    decision_cfg = cfg.get("decision", {}) or {}
    index_cfg = cfg.get("index", {}) or {}
    sector_cfg = cfg.get("sector_boost", {}) or {}

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

    # Ensure full-market mode
    signals = dict(signals_cfg)
    for k in ("symbols", "watchlist", "universe"):
        signals.pop(k, None)

    return {
        "trade_date": "",
        "top_n": int(top_n),
        "universe": {
            "min_price": float(universe_cfg.get("min_price", 5.0)),
            "max_price": float(universe_cfg.get("max_price", 0.0)),
            "min_avg_amount_20": float(universe_cfg.get("min_avg_amount_20", 0.0)),
            "exclude_st": bool(universe_cfg.get("exclude_st", True)),
        },
        "signals": signals,
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
        "is_watchlist": False,
        "regime_filter": bool(index_cfg.get("regime_filter", True)),
        "index_symbol": str(index_cfg.get("symbol", "000300")),
        "index_ma_window": int(index_cfg.get("ma_window", 200)),
        "sector_boost": sector_cfg,
        "weights": weights,
        "decision": decision_cfg,
    }


def _ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and store minute kline into data/manual_minute")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--source", default="model_top", choices=["model_top", "watchlist", "all"])
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--watchlist-file", default="")
    parser.add_argument("--universe-file", default="")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--provider", default="tencent", choices=["tencent", "biying"])
    parser.add_argument("--interval", default="m5")
    parser.add_argument("--limit", type=int, default=320)
    parser.add_argument("--years", type=int, default=0, help="only for biying history; 0 means disabled")
    parser.add_argument("--start-date", default="", help="YYYYMMDD, only for biying history")
    parser.add_argument("--end-date", default="", help="YYYYMMDD, only for biying history")
    parser.add_argument("--biying-licence", default="")
    parser.add_argument("--biying-base-url", default="")
    parser.add_argument("--out-dir", default="data/manual_minute")
    parser.add_argument("--skip-non-trading", action="store_true", default=True)
    parser.add_argument("--no-skip-non-trading", dest="skip_non_trading", action="store_false")
    parser.add_argument("--use-proxy", action="store_true")
    parser.add_argument("--proxy", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    signals_cfg = cfg.get("signals", {}) or {}
    use_proxy = bool(args.use_proxy) or bool(signals_cfg.get("use_proxy", False))
    proxy = args.proxy or str(signals_cfg.get("proxy", "") or "")
    if use_proxy and not proxy:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""

    use_history_range = (
        args.provider == "biying"
        and (bool(args.start_date) or bool(args.end_date) or (args.years and args.years > 0))
    )

    if args.skip_non_trading and not use_history_range:
        trade_date = resolve_trade_date("")
        trade_calendar_file = str(cfg.get("trade_calendar_file", "./data/trade_calendar.csv"))
        calendar_dates = _load_trade_calendar(trade_calendar_file)
        if trade_date:
            trade_dt = dt.date.fromisoformat(trade_date)
            if not _is_trading_day_cn(trade_dt, calendar_dates):
                print("[info] non-trading day; skip.")
                return

    symbols = _resolve_symbols(args, cfg)
    if not symbols:
        print("[error] symbol list empty.")
        return

    biying_licence = ""
    biying_base_url = ""
    start_date = ""
    end_date = ""
    effective_limit = int(args.limit)
    if args.provider == "biying":
        biying_licence = str(
            args.biying_licence
            or signals_cfg.get("realtime_biying_licence")
            or signals_cfg.get("biying_licence")
            or ""
        ).strip()
        biying_base_url = str(
            args.biying_base_url
            or signals_cfg.get("realtime_biying_base_url")
            or "http://api.biyingapi.com"
        ).strip()
        if not biying_licence:
            print("[error] biying licence is required. use --biying-licence or config signals.realtime_biying_licence")
            return
        if args.start_date:
            start_date = str(args.start_date).strip()
        if args.end_date:
            end_date = str(args.end_date).strip()
        if args.years and args.years > 0:
            end_d = dt.date.today()
            start_d = end_d - dt.timedelta(days=int(args.years) * 365)
            if not start_date:
                start_date = _ymd(start_d)
            if not end_date:
                end_date = _ymd(end_d)
        if start_date or end_date:
            print(f"[info] biying history range: {start_date or '(auto)'} ~ {end_date or '(auto)'}")
        if (start_date or end_date) and effective_limit == 320:
            # Keep Tencent default intact; for Biying history range, default should mean "all in range".
            effective_limit = 0
            print("[info] biying history mode: auto set --limit=0 (full range)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    empty = 0
    error = 0

    for idx, sym in enumerate(symbols, start=1):
        try:
            if args.provider == "biying":
                df = fetch_biying_minute_history(
                    sym,
                    interval=args.interval,
                    licence=biying_licence,
                    start=start_date,
                    end=end_date,
                    limit=effective_limit,
                    base_url=biying_base_url,
                    use_proxy=use_proxy,
                    proxy=proxy,
                )
            else:
                df = fetch_tencent_minute(
                    sym,
                    interval=args.interval,
                    limit=effective_limit,
                    use_proxy=use_proxy,
                    proxy=proxy,
                )
        except Exception:
            error += 1
            if idx % 50 == 0:
                print(f"[info] {idx}/{len(symbols)} ok={ok} empty={empty} error={error}")
            continue

        if df is None or df.empty:
            empty += 1
            if idx % 50 == 0:
                print(f"[info] {idx}/{len(symbols)} ok={ok} empty={empty} error={error}")
            continue

        # split by date, save per day file
        for date, group in df.groupby("date"):
            sym_dir = out_dir / sym
            sym_dir.mkdir(parents=True, exist_ok=True)
            out_path = sym_dir / f"{date}_{args.interval}.csv"
            old = _read_csv(out_path)
            merged = _merge_minute(old, group)
            merged.to_csv(out_path, index=False, encoding="utf-8")

        ok += 1
        if idx % 50 == 0:
            print(f"[info] {idx}/{len(symbols)} ok={ok} empty={empty} error={error}")

    print(f"[ok] done: ok={ok} empty={empty} error={error}")


if __name__ == "__main__":
    main()
