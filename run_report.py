from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from features.factors import build_factors
from report.to_md import write_md_report
from report.to_xlsx import write_xlsx_report
from sqdata.akshare_fetcher import fetch_a_share_daily_panel
from sqdata.calendar import resolve_trade_date
from sqdata.universe import load_universe_symbols, filter_symbols_by_market, filter_symbols_by_board
from sqdata.fetcher import get_realtime
from strategy.scorer import score_and_rank
from sqdata.sector_map import apply_sector_map
from strategy.decision import apply_short_term_decision
from strategy.universe import filter_universe


def load_config(path: str) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from e

    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def _detect_local_proxy() -> str:
    import socket

    candidates = [7890, 7891, 7892, 7893, 8888, 1080]
    for port in candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                s.close()
                return f"http://127.0.0.1:{port}"
            s.close()
        except Exception:
            continue
    return ""


def run_report(config_path: str = "config/default.yaml", no_cache: bool = False) -> dict:
    cfg = load_config(config_path)

    trade_date = resolve_trade_date(str(cfg.get("trade_date", "")).strip())
    universe_cfg = cfg.get("universe", {}) or {}
    signals_cfg = cfg.get("signals", {}) or {}
    weights = cfg.get("score_weights", {}) or {}
    output_cfg = cfg.get("output", {}) or {}
    decision_cfg = cfg.get("decision", {}) or {}
    universe_file = str(cfg.get("universe_file", "./data/universe.csv"))
    use_universe_file = bool(cfg.get("use_universe_file", True))
    allow_eastmoney_fallback = bool(cfg.get("allow_eastmoney_fallback", False))
    market_scope = cfg.get("market_scope", ["sh", "sz"])
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = bool(cfg.get("exclude_chi_next", False))
    mainboard_only = bool(cfg.get("mainboard_only", True))
    exclude_limit_up = bool(cfg.get("exclude_limit_up", True))
    limit_up_pct = float(cfg.get("limit_up_pct", 9.8))
    exclude_non_realtime_pct = bool(cfg.get("exclude_non_realtime_pct", True))
    realtime_refresh_top_n = int(cfg.get("realtime_refresh_top_n", 50))

    preselect_n = int(signals_cfg.get("preselect_n", 300))
    use_proxy_cfg = signals_cfg.get("use_proxy", None)
    if use_proxy_cfg is None:
        import os
        use_proxy = bool(
            os.environ.get("http_proxy")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("HTTPS_PROXY")
        )
    else:
        use_proxy = bool(use_proxy_cfg)

    proxy = str(signals_cfg.get("proxy", "") or "")
    if not proxy and use_proxy:
        import os
        proxy = (
            os.environ.get("http_proxy")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or ""
        )

    auto_proxy_note = ""
    if not proxy and not use_proxy:
        auto_proxy = _detect_local_proxy()
        if auto_proxy:
            use_proxy = True
            proxy = auto_proxy
            auto_proxy_note = f"自动检测到本地代理: {proxy}"
            print(f"[info] {auto_proxy_note}")

    signals_cfg = dict(signals_cfg)
    signals_cfg["use_proxy"] = use_proxy
    signals_cfg["proxy"] = proxy
    top_n = int(output_cfg.get("top_n", 20))
    out_dir = Path(output_cfg.get("out_dir", "./output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    if auto_proxy_note:
        notes.append(auto_proxy_note)
    symbols_cfg = signals_cfg.get("symbols") or signals_cfg.get("watchlist") or signals_cfg.get("universe")
    if (not symbols_cfg) and use_universe_file:
        symbols_list = load_universe_symbols(universe_file)
        if symbols_list:
            symbols_list = filter_symbols_by_market(symbols_list, market_scope)
            symbols_list = filter_symbols_by_board(
                symbols_list,
                exclude_star=exclude_star,
                exclude_chi_next=exclude_chi_next,
                mainboard_only=mainboard_only,
            )
            signals_cfg = dict(signals_cfg)
            signals_cfg["symbols"] = symbols_list
            notes.append(f"使用本地股票清单: {universe_file} (n={len(symbols_list)})")
        else:
            notes.append(f"本地股票清单为空: {universe_file}")
            notes.append("请运行: python update_universe.py 生成全市场清单。")
    try:
        spot = fetch_a_share_daily_panel(
            trade_date=trade_date,
            signals_cfg=signals_cfg,
            no_cache=no_cache,
            use_proxy=use_proxy,
            proxy=proxy,
            allow_eastmoney_fallback=allow_eastmoney_fallback,
        )
    except Exception as e:
        spot = None
        notes.append(f"行情面板获取失败: {e}")
        print(f"[warn] spot fetch failed: {e}")
    if spot is None or spot.empty:
        ranked = spot if spot is not None else None
        if not notes:
            notes.append("行情面板为空，可能是数据源不可用或网络受限。")
    else:
        signals_cfg = dict(signals_cfg)
        signals_cfg["allow_akshare_fallback"] = allow_eastmoney_fallback
        print(f"[info] spot rows: {len(spot)}")

        symbols_cfg = signals_cfg.get("symbols") or signals_cfg.get("watchlist") or signals_cfg.get("universe")
        symbols_list = []
        if isinstance(symbols_cfg, str):
            symbols_list = [s.strip() for s in symbols_cfg.split(",") if s.strip()]
        elif isinstance(symbols_cfg, list):
            symbols_list = [str(s).strip() for s in symbols_cfg if str(s).strip()]

        if symbols_list:
            # User-specified symbols: skip liquidity filter to avoid wiping all candidates.
            universe_cfg = dict(universe_cfg)
            universe_cfg["min_avg_amount_20"] = 0
            notes.append("已检测到自选 symbol 列表，已关闭成交额过滤。")

        spot = filter_universe(spot, universe_cfg)
        print(f"[info] after universe: {len(spot)}")

        if "amount" in spot.columns:
            spot = spot.sort_values("amount", ascending=False).head(preselect_n)
        else:
            spot = spot.head(preselect_n)

        print(f"[info] after preselect: {len(spot)}")

        try:
            stats: dict = {}
            factors = build_factors(spot, signals_cfg, trade_date, no_cache=no_cache, stats=stats)
            sector_cfg = cfg.get("sector_boost", {}) or {}
            if sector_cfg.get("enabled"):
                map_file = str(sector_cfg.get("map_file", "./data/sector_map.csv"))
                factors = apply_sector_map(factors, map_file)
            ranked = score_and_rank(factors, weights, top_n=top_n, sector_cfg=sector_cfg)
            ranked = apply_short_term_decision(ranked, decision_cfg)
            if exclude_limit_up and "pct_chg" in ranked.columns:
                ranked = ranked[pd.to_numeric(ranked["pct_chg"], errors="coerce") < limit_up_pct].reset_index(drop=True)
            if "pct_source" in ranked.columns:
                src_counts = ranked["pct_source"].value_counts().to_dict()
                notes.append("涨跌幅来源统计: " + ", ".join(f"{k}={v}" for k, v in src_counts.items()))
                if exclude_non_realtime_pct:
                    filtered = ranked[ranked["pct_source"].isin(["spot", "spot_calc", "spot_fix"])].reset_index(drop=True)
                    if filtered.empty and not ranked.empty:
                        notes.append("实时涨跌幅不可用，已保留历史/估算数据。")
                    else:
                        ranked = filtered
            # Refresh realtime for top N
            try:
                n = min(len(ranked), realtime_refresh_top_n)
                for i in range(n):
                    sym = str(ranked.iloc[i].get("symbol", "")).strip()
                    if not sym:
                        continue
                    q = get_realtime(sym, provider="auto", use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=True)
                    if q.price and q.price > 0:
                        ranked.at[ranked.index[i], "close"] = q.price
                        ranked.at[ranked.index[i], "pct_chg"] = q.pct
                        ranked.at[ranked.index[i], "price_source"] = "spot_refresh"
                        ranked.at[ranked.index[i], "pct_source"] = "spot_refresh"
            except Exception:
                pass
            print(f"[info] factors rows: {len(factors)}")
            if stats:
                ok = stats.get("ok", 0)
                empty = stats.get("hist_empty", 0)
                short = stats.get("hist_short", 0)
                err = stats.get("hist_error", 0)
                notes.append(f"历史行情统计: ok={ok}, empty={empty}, short={short}, error={err}")
                if stats.get("hist_empty_symbols"):
                    notes.append("hist_empty 示例: " + ", ".join(stats["hist_empty_symbols"]))
                if stats.get("hist_short_symbols"):
                    notes.append("hist_short 示例: " + ", ".join(stats["hist_short_symbols"]))
                if stats.get("hist_error_msgs"):
                    notes.append("hist_error 示例: " + " | ".join(stats["hist_error_msgs"]))
                if stats.get("spot_close_fixed") or stats.get("spot_pct_fixed"):
                    notes.append(
                        f"现价/涨跌幅纠正: close_fixed={stats.get('spot_close_fixed', 0)}, pct_fixed={stats.get('spot_pct_fixed', 0)}"
                    )
        except Exception as e:
            notes.append(f"因子/评分失败: {e}")
            print(f"[warn] factors/scoring failed: {e}")
            ranked = None

    md_path = out_dir / f"report_{trade_date}.md"
    xlsx_path = out_dir / f"report_{trade_date}.xlsx"

    if ranked is None:
        ranked = pd.DataFrame()

    write_md_report(ranked, trade_date, md_path, notes=notes)
    try:
        write_xlsx_report(ranked, trade_date, xlsx_path)
    except Exception as e:
        print(f"[warn] xlsx write failed: {e}")

    print(f"[ok] report generated for {trade_date}")
    print(f"[ok] md: {md_path}")
    print(f"[ok] xlsx: {xlsx_path}")

    return {"trade_date": trade_date, "md": str(md_path), "xlsx": str(xlsx_path), "rows": int(getattr(ranked, "shape", [0])[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily short-term stock report.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--no-cache", action="store_true", help="Disable local cache for data fetch.")
    args = parser.parse_args()

    run_report(config_path=args.config, no_cache=args.no_cache)


if __name__ == "__main__":
    main()
