from __future__ import annotations

from pathlib import Path
import argparse
import json
import datetime as dt

import requests

from sqdata.akshare_fetcher import fetch_hist
import app as app_module


def _fetch_tencent_index(
    symbol: str,
    start: str,
    end: str,
    adj: str = "qfq",
    *,
    proxy: str = "",
    debug: bool = False,
) -> list[list[str]]:
    if symbol.startswith(("sh", "sz")):
        mcode = symbol
    else:
        # Index codes: 399xxx are SZ, others default to SH for major indices (e.g. 000300)
        if symbol.startswith("399"):
            mcode = "sz" + symbol
        else:
            mcode = "sh" + symbol
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    count = 640
    param = f"{mcode},day,{start},{end},{count},{adj}"
    proxies = {"http": proxy, "https": proxy} if proxy else None
    r = requests.get(url, params={"param": param}, timeout=15, proxies=proxies)
    r.raise_for_status()
    text = r.text.strip()
    if debug:
        print(f"[debug] tencent url: {r.url}")
        print(f"[debug] tencent resp (head): {text[:200]}")
    if not text.startswith("{"):
        l = text.find("{")
        rpos = text.rfind("}")
        if l >= 0 and rpos > l:
            text = text[l : rpos + 1]
    j = json.loads(text)
    data = (j.get("data") or {}).get(mcode) or {}
    rows = data.get("qfqday") or data.get("day") or None
    if not rows:
        raise RuntimeError("Tencent kline empty")
    return rows


def _rows_to_df(rows: list[list[str]]) -> "pd.DataFrame":
    import pandas as pd

    out = []
    for row in rows:
        if not row or len(row) < 6:
            continue
        date_str, o, c, h, l, v = row[:6]
        amt = row[6] if len(row) > 6 else 0
        out.append([date_str, o, c, h, l, v, amt])
    df = pd.DataFrame(out, columns=["date", "open", "close", "high", "low", "volume", "amount"])
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch index history for excess-return labels")
    parser.add_argument("--symbol", default="000300")
    parser.add_argument("--hist-days", type=int, default=1200)
    parser.add_argument("--proxy", default="")
    parser.add_argument("--force-tencent", action="store_true", help="Disable AkShare fallback (Eastmoney).")
    args = parser.parse_args()

    symbol = str(args.symbol)
    proxy = args.proxy.strip()
    env_proxy = app_module._get_env_proxy()
    if not proxy:
        if env_proxy:
            app_module._ensure_no_proxy_defaults()
            proxy = ""
        else:
            detected = app_module._detect_proxy()
            if detected:
                app_module._set_env_proxy(detected)
                env_proxy = app_module._get_env_proxy()
                proxy = ""
    use_proxy = bool(proxy) or bool(env_proxy)
    print(f"[info] proxy: {proxy if proxy else ('ENV' if env_proxy else 'DIRECT')}")

    # Prefer direct Tencent kline (same as curl), fallback to fetch_hist.
    df = None
    try:
        end_date = dt.date.today().isoformat()
        start_date = (dt.date.today() - dt.timedelta(days=int(args.hist_days) * 2)).isoformat()
        proxy_for_direct = proxy or env_proxy or ""
        rows = _fetch_tencent_index(
            symbol,
            start_date,
            end_date,
            adj="qfq",
            proxy=proxy_for_direct,
            debug=True,
        )
        df = _rows_to_df(rows)
    except Exception as e:
        print(f"[warn] direct tencent kline failed: {e}")
        df = fetch_hist(
            symbol,
            hist_days=int(args.hist_days),
            use_proxy=use_proxy,
            proxy=proxy,
            allow_akshare_fallback=not args.force_tencent,
            manual_hist_dir=None,
        )
    if df is None or df.empty:
        raise SystemExit("[error] failed to fetch index history")

    out_dir = Path("data/manual_hist")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.csv"
    df.to_csv(out_path, index=False)
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
