import random
import time
import threading
import socket
import datetime as dt
import os
import re
from pathlib import Path

import requests
import pandas as pd
import urllib3.util.connection as urllib3_cn
import json
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

_HIST_MEM_CACHE: "OrderedDict[str, tuple[float, pd.DataFrame]]" = OrderedDict()
_HIST_MEM_LOCK = threading.Lock()
_HIST_MEM_MAX = 2048


def _hist_mem_get(key: str, ttl_sec: int) -> pd.DataFrame | None:
    if ttl_sec <= 0:
        return None
    now = time.time()
    with _HIST_MEM_LOCK:
        item = _HIST_MEM_CACHE.get(key)
        if not item:
            return None
        ts, df = item
        if ttl_sec > 0 and now - ts > ttl_sec:
            _HIST_MEM_CACHE.pop(key, None)
            return None
        _HIST_MEM_CACHE.move_to_end(key)
        return df


def _hist_mem_set(key: str, df: pd.DataFrame) -> None:
    with _HIST_MEM_LOCK:
        _HIST_MEM_CACHE[key] = (time.time(), df)
        _HIST_MEM_CACHE.move_to_end(key)
        while len(_HIST_MEM_CACHE) > _HIST_MEM_MAX:
            _HIST_MEM_CACHE.popitem(last=False)

def _eastmoney_spot_page(session: requests.Session, *, page: int, page_size: int, timeout: int) -> pd.DataFrame:
    hosts = [82, 80, 81, 83, 84, 85]
    host = hosts[(page - 1) % len(hosts)]
    url = f"https://{host}.push2.eastmoney.com/api/qt/clist/get"

    params = {
        "pn": str(page),
        "pz": str(page_size),
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f12,f14,f2,f3,f5,f6,f20,f21,f13",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://quote.eastmoney.com/",
        "Connection": "keep-alive",
    }

    r = session.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    diff = (j.get("data") or {}).get("diff") or []
    if not diff:
        return pd.DataFrame()

    df = pd.DataFrame(diff).rename(
        columns={
            "f12": "symbol",
            "f14": "name",
            "f2": "close",
            "f3": "pct_chg",
            "f5": "volume",
            "f6": "amount",
            "f20": "mkt_cap",
            "f21": "float_mkt_cap",
            "f13": "market",
        }
    )

    df["symbol"] = df["symbol"].astype(str)
    for c in ["close", "pct_chg", "volume", "amount", "mkt_cap", "float_mkt_cap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _build_session(use_proxy: bool, proxy: str | None) -> requests.Session:
    session = requests.Session()
    if proxy:
        if proxy.startswith("socks") and not _socks_available():
            raise RuntimeError("SOCKS proxy requested but PySocks not installed. Install with `pip install requests[socks]`.")
        session.proxies.update({"http": proxy, "https": proxy})
        session.trust_env = False
    else:
        session.trust_env = use_proxy
    return session


def _socks_available() -> bool:
    try:
        import socks  # noqa: F401
        return True
    except Exception:
        return False


def _with_proxy_env(proxy: str | None):
    class _ProxyEnv:
        def __enter__(self):
            self._old = {
                "http_proxy": os.environ.get("http_proxy"),
                "https_proxy": os.environ.get("https_proxy"),
                "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
                "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
            }
            if proxy:
                os.environ["http_proxy"] = proxy
                os.environ["https_proxy"] = proxy
                os.environ["HTTP_PROXY"] = proxy
                os.environ["HTTPS_PROXY"] = proxy
            return self

        def __exit__(self, exc_type, exc, tb):
            for k, v in self._old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return False

    return _ProxyEnv()


def fetch_spot_a_share(no_cache: bool = False, use_proxy: bool = False, proxy: str = "") -> pd.DataFrame:
    # Force IPv4: browsers fall back quickly, but Python may stick to IPv6 and get reset by the server.
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

    session = _build_session(use_proxy, proxy)

    page_size = 200
    timeout = 20
    max_pages = 30

    frames = []
    for page in range(1, max_pages + 1):
        last_err = None
        for attempt in range(4):
            try:
                dfp = _eastmoney_spot_page(session, page=page, page_size=page_size, timeout=timeout)
                if dfp.empty:
                    page = max_pages + 1
                    break
                frames.append(dfp)
                time.sleep(0.25 + random.uniform(0.0, 0.25))  # 降速，降低被掐概率
                break
            except Exception as e:
                last_err = e
                time.sleep(min(2.5, 0.4 * (2**attempt)) + random.uniform(0.0, 0.3))
        else:
            raise RuntimeError(f"Eastmoney spot fetch failed on page {page}: {last_err!r}")

        if page > max_pages:
            break

    if not frames:
        raise RuntimeError("Eastmoney spot fetch returned no data.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["symbol"].str.match(r"^(0|3|6)\d{5}$", na=False)].copy()
    return df


def _fetch_tencent_realtime_quotes(
    symbols: list[str],
    timeout: int = 10,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
    chunk_size: int = 200,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Fetch realtime quotes for a list of symbols via Tencent qt.gtimg.cn.

    Returns a DataFrame with columns compatible with the rest of the pipeline:
    symbol, name, close, pct_chg, volume, amount, market
    """
    def to_mcode(sym: str) -> str:
        s = str(sym).strip()
        if s.startswith(("sh", "sz")):
            return s
        if s.startswith(("6", "5", "9")):
            return f"sh{s}"
        return f"sz{s}"

    mcodes = [to_mcode(s) for s in symbols]

    # Tencent supports multiple codes separated by comma; keep chunks conservative
    chunk_size = max(50, min(int(chunk_size or 200), 800))
    max_workers = max(1, int(max_workers or 1))

    def _looks_like_code(val: str, symbol: str) -> bool:
        s = str(val).strip()
        if not s.isdigit():
            return False
        return s.zfill(6) == str(symbol).zfill(6)

    def _parse_text(text: str) -> list[dict]:
        parsed = []
        for line in text.splitlines():
            line = line.strip().strip(";")
            if not line or '="' not in line:
                continue
            try:
                left, payload = line.split('="', 1)
                payload = payload.rstrip('"')
                key = left.split("v_", 1)[1]
                arr = payload.split("~")

                name = arr[1] if len(arr) > 1 else ""
                symbol = key[2:]
                offset = 1 if (len(arr) > 3 and _looks_like_code(arr[2], symbol)) else 0
                price = float(arr[2 + offset] or 0) if len(arr) > 2 + offset else 0.0
                prev_close = float(arr[3 + offset] or 0) if len(arr) > 3 + offset else 0.0
                vol = float(arr[7 + offset] or 0) if len(arr) > 7 + offset else 0.0
                amt = float(arr[8 + offset] or 0) if len(arr) > 8 + offset else 0.0
                pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0

                market = "sh" if key.startswith("sh") else "sz"

                parsed.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "close": price,
                        "pct_chg": round(pct, 2),
                        "volume": vol,
                        "amount": amt,
                        "market": market,
                    }
                )
            except Exception:
                continue
        return parsed

    def _fetch_chunk(chunk: list[str]) -> list[dict]:
        session = _build_session(use_proxy, proxy)
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        try:
            r = session.get(url, timeout=timeout)
        except Exception:
            r = session.get("http://qt.gtimg.cn/q=" + ",".join(chunk), timeout=timeout)
        r.encoding = "gbk"
        return _parse_text(r.text)

    chunks = [mcodes[i : i + chunk_size] for i in range(0, len(mcodes), chunk_size)]
    rows: list[dict] = []
    if max_workers > 1 and len(chunks) > 1:
        workers = min(max_workers, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for parsed in ex.map(_fetch_chunk, chunks):
                if parsed:
                    rows.extend(parsed)
    else:
        for chunk in chunks:
            rows.extend(_fetch_chunk(chunk))

    df = pd.DataFrame(rows)
    if df.empty:
        if allow_direct_fallback and (use_proxy or proxy):
            return _fetch_tencent_realtime_quotes(symbols, timeout=timeout, use_proxy=False, proxy="", allow_direct_fallback=False)
        return df

    df["symbol"] = df["symbol"].astype(str)
    for c in ["close", "pct_chg", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if allow_direct_fallback and (use_proxy or proxy):
        valid = df[pd.to_numeric(df["close"], errors="coerce") > 0]
        if valid.empty:
            return _fetch_tencent_realtime_quotes(symbols, timeout=timeout, use_proxy=False, proxy="", allow_direct_fallback=False)
    return df


def _fetch_sina_realtime_quotes(
    symbols: list[str],
    timeout: int = 10,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
) -> pd.DataFrame:
    try:
        from .sina_fetcher import fetch_realtime_batch
    except Exception:
        return pd.DataFrame()
    return fetch_realtime_batch(
        symbols,
        timeout=timeout,
        use_proxy=use_proxy,
        proxy=proxy,
        allow_direct_fallback=allow_direct_fallback,
    )


def _fetch_netease_realtime_quotes(
    symbols: list[str],
    timeout: int = 10,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
) -> pd.DataFrame:
    try:
        from .netease_fetcher import fetch_realtime_batch
    except Exception:
        return pd.DataFrame()
    return fetch_realtime_batch(
        symbols,
        timeout=timeout,
        use_proxy=use_proxy,
        proxy=proxy,
        allow_direct_fallback=allow_direct_fallback,
    )


def _fetch_biying_realtime_quotes(
    symbols: list[str],
    timeout: int = 10,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
    *,
    licence: str = "",
    base_url: str = "http://api.biyingapi.com",
    chunk_size: int = 20,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Fetch realtime quotes via 必盈 API.

    Endpoint (docs): /hsrl/ssjy_more/{licence}?stock_codes=000001,600000
    Notes:
    - max 20 symbols per request
    - response fields can include dm/p/pc/v/cje
    """

    def _norm_symbol(sym: str) -> str:
        s = str(sym).strip().lower()
        if s.startswith(("sh", "sz")):
            s = s[2:]
        return s.zfill(6)

    def _to_market(sym: str) -> str:
        return "sh" if str(sym).startswith(("5", "6", "9")) else "sz"

    def _to_float(v: object) -> float:
        if v is None:
            return float("nan")
        if isinstance(v, (int, float)):
            try:
                return float(v)
            except Exception:
                return float("nan")
        s = str(v).strip()
        if not s:
            return float("nan")
        if s.endswith("%"):
            s = s[:-1].strip()
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return float("nan")

    def _sum_level_volume(item: dict, side: str) -> float:
        # side in {"buy", "sell"}, best-effort over common field variants.
        total = 0.0
        found = False
        aliases = [
            f"{side}",
            f"{side}_v",
            f"{side}_vol",
            f"{side}v",
            f"{side}vol",
            f"wt_{side}",
            f"entrust_{side}",
            f"delegate_{side}",
            f"w_{side}",
            f"w{side}",
        ]
        for k in aliases:
            val = _to_float(item.get(k))
            if val == val and val >= 0:
                total += float(val)
                found = True
                break

        for i in range(1, 6):
            lvl_keys = [
                f"{side}{i}_v",
                f"{side}{i}_vol",
                f"{side}{i}v",
                f"b{i}_v" if side == "buy" else f"a{i}_v",
                f"b{i}v" if side == "buy" else f"a{i}v",
                f"buy{i}_v" if side == "buy" else f"sell{i}_v",
                f"buy{i}v" if side == "buy" else f"sell{i}v",
                f"buy{i}_vol" if side == "buy" else f"sell{i}_vol",
            ]
            for k in lvl_keys:
                val = _to_float(item.get(k))
                if val == val and val >= 0:
                    total += float(val)
                    found = True
                    break
        return total if found else float("nan")

    if not licence:
        return pd.DataFrame()

    syms = [_norm_symbol(s) for s in symbols if str(s).strip()]
    if not syms:
        return pd.DataFrame()

    b = str(base_url or "http://api.biyingapi.com").rstrip("/")
    url = f"{b}/hsrl/ssjy_more/{licence}"
    # 必盈接口单次最多 20 个 code
    chunk_size = max(1, min(int(chunk_size or 20), 20))
    max_workers = max(1, int(max_workers or 1))

    def _extract_rows(payload: object) -> list[dict]:
        rows: list[dict] = []
        data = payload
        if isinstance(payload, dict):
            # Common wrapper keys
            for k in ("data", "result", "results", "list", "rows"):
                v = payload.get(k)
                if isinstance(v, list):
                    data = v
                    break
            else:
                # single-row dict fallback
                if any(k in payload for k in ("dm", "stock_code", "code", "p", "price", "close")):
                    data = [payload]
                else:
                    data = []
        if not isinstance(data, list):
            return rows

        for item in data:
            if not isinstance(item, dict):
                continue
            symbol = str(
                item.get("dm")
                or item.get("stock_code")
                or item.get("code")
                or item.get("symbol")
                or ""
            ).strip()
            symbol = _norm_symbol(symbol)
            if not symbol or not symbol.isdigit():
                continue
            close = pd.to_numeric(
                item.get("p", item.get("price", item.get("close", None))),
                errors="coerce",
            )
            pct = pd.to_numeric(
                item.get("pc", item.get("pct_chg", item.get("change_percent", None))),
                errors="coerce",
            )
            volume = pd.to_numeric(item.get("v", item.get("volume", None)), errors="coerce")
            amount = pd.to_numeric(item.get("cje", item.get("amount", None)), errors="coerce")
            name = str(item.get("n") or item.get("name") or item.get("mc") or "").strip()
            weibi = _to_float(
                item.get("wb")
                or item.get("weibi")
                or item.get("commission_ratio")
                or item.get("entrust_ratio")
                or item.get("delegate_ratio")
            )
            if not (weibi == weibi):
                buy_total = _sum_level_volume(item, "buy")
                sell_total = _sum_level_volume(item, "sell")
                if buy_total == buy_total and sell_total == sell_total and (buy_total + sell_total) > 0:
                    weibi = (buy_total - sell_total) / (buy_total + sell_total) * 100.0
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "close": float(close) if pd.notna(close) else float("nan"),
                    "pct_chg": float(pct) if pd.notna(pct) else float("nan"),
                    "volume": float(volume) if pd.notna(volume) else float("nan"),
                    "amount": float(amount) if pd.notna(amount) else float("nan"),
                    "weibi": float(weibi) if weibi == weibi else float("nan"),
                    "market": _to_market(symbol),
                }
            )
        return rows

    chunks = [syms[i : i + chunk_size] for i in range(0, len(syms), chunk_size)]

    def _fetch_chunk(chunk: list[str]) -> list[dict]:
        try:
            session = _build_session(use_proxy, proxy)
            r = session.get(url, params={"stock_codes": ",".join(chunk)}, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            return _extract_rows(payload)
        except Exception:
            return []

    all_rows: list[dict] = []
    if max_workers > 1 and len(chunks) > 1:
        workers = min(max_workers, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rows in ex.map(_fetch_chunk, chunks):
                if rows:
                    all_rows.extend(rows)
    else:
        for chunk in chunks:
            rows = _fetch_chunk(chunk)
            if rows:
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        if allow_direct_fallback and (use_proxy or proxy):
            return _fetch_biying_realtime_quotes(
                symbols,
                timeout=timeout,
                use_proxy=False,
                proxy="",
                allow_direct_fallback=False,
                licence=licence,
                base_url=base_url,
                chunk_size=chunk_size,
                max_workers=max_workers,
            )
        return df

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    for c in ["close", "pct_chg", "volume", "amount", "weibi"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    return df


def fetch_a_share_daily_panel(
    trade_date: str = "",
    signals_cfg: dict | None = None,
    no_cache: bool = False,
    use_proxy: bool = False,
    proxy: str = "",
    allow_eastmoney_fallback: bool = True,
) -> pd.DataFrame:
    """Unified daily panel fetcher for the pipeline.

    Preference order:
    1) If signals_cfg provides a symbol list, use configured realtime provider quotes.
    2) Otherwise, fall back to the existing Eastmoney full-market panel (legacy).

    Notes:
    - `trade_date` is kept for compatibility with the CLI.
    - When using realtime quotes, the data is realtime (T-day).
    """
    cfg = signals_cfg or {}
    provider = str(cfg.get("realtime_provider", "auto")).lower()
    auto_fallback = bool(cfg.get("realtime_auto_fallback", False))
    provider_order_cfg = cfg.get("realtime_provider_order")
    symbols = cfg.get("symbols") or cfg.get("watchlist") or cfg.get("universe")
    if isinstance(symbols, str):
        # allow comma-separated string
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    if isinstance(symbols, list) and symbols:
        allowed = ["tencent", "netease", "sina", "biying", "eastmoney"]
        if isinstance(provider_order_cfg, (list, tuple)) and provider_order_cfg:
            provider_order = [str(p).lower() for p in provider_order_cfg if str(p).lower() in allowed]
        else:
            # Default to single-provider mode to avoid long multi-provider retries
            # on networks where fallback providers are blocked/unreachable.
            if provider == "tencent":
                provider_order = ["tencent"]
            elif provider == "netease":
                provider_order = ["netease"]
            elif provider == "sina":
                provider_order = ["sina"]
            elif provider == "biying":
                provider_order = ["biying"]
            elif provider == "eastmoney":
                provider_order = ["eastmoney"]
            else:
                provider_order = ["tencent"]
                if auto_fallback:
                    provider_order = ["tencent", "netease", "sina", "biying"]
        if allow_eastmoney_fallback and "eastmoney" not in provider_order:
            provider_order.append("eastmoney")

        chunk_size = int(cfg.get("realtime_chunk_size", 200) or 200)
        max_workers = int(cfg.get("realtime_max_workers", 4) or 4)

        def _fetch_provider(p: str, syms: list[str]) -> pd.DataFrame:
            if p == "tencent":
                return _fetch_tencent_realtime_quotes(
                    syms,
                    use_proxy=use_proxy,
                    proxy=proxy,
                    allow_direct_fallback=True,
                    chunk_size=chunk_size,
                    max_workers=max_workers,
                )
            if p == "netease":
                return _fetch_netease_realtime_quotes(syms, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=True)
            if p == "sina":
                return _fetch_sina_realtime_quotes(syms, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=True)
            if p == "biying":
                biying_cfg = cfg.get("biying") or {}
                licence = str(
                    cfg.get("realtime_biying_licence")
                    or cfg.get("biying_licence")
                    or biying_cfg.get("licence")
                    or ""
                ).strip()
                base_url = str(
                    cfg.get("realtime_biying_base_url")
                    or biying_cfg.get("base_url")
                    or "http://api.biyingapi.com"
                ).strip()
                timeout_biying = int(cfg.get("realtime_biying_timeout", 10) or 10)
                if not licence:
                    raise RuntimeError("biying provider requires licence")
                return _fetch_biying_realtime_quotes(
                    syms,
                    timeout=timeout_biying,
                    use_proxy=use_proxy,
                    proxy=proxy,
                    allow_direct_fallback=True,
                    licence=licence,
                    base_url=base_url,
                    chunk_size=int(cfg.get("realtime_biying_chunk_size", 20) or 20),
                    max_workers=max_workers,
                )
            if p == "eastmoney":
                if not allow_eastmoney_fallback:
                    return pd.DataFrame()
                df_em = fetch_spot_a_share(no_cache=no_cache, use_proxy=use_proxy, proxy=proxy)
                if df_em.empty:
                    return df_em
                if syms:
                    df_em = df_em[df_em["symbol"].isin(set(syms))].copy()
                return df_em
            return pd.DataFrame()

        df_valid = pd.DataFrame()
        missing = set(symbols)
        source_used = ""
        for p in provider_order:
            if not missing:
                break
            try:
                df_p = _fetch_provider(p, list(missing))
            except Exception as e:
                print(f"[warn] {p} realtime fetch failed: {e}")
                df_p = pd.DataFrame()
            if df_p.empty:
                continue
            if "close" in df_p.columns:
                df_p = df_p[pd.to_numeric(df_p["close"], errors="coerce") > 0].copy()
            if df_p.empty:
                continue
            if not source_used:
                source_used = p
            if df_valid.empty:
                df_valid = df_p.copy()
            else:
                df_valid = pd.concat([df_valid, df_p], ignore_index=True)
                df_valid = df_valid.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
            missing = set(symbols) - set(df_valid.get("symbol", []))

        if df_valid.empty:
            if not allow_eastmoney_fallback:
                # Create placeholders so factor builder can still use historical data.
                if symbols:
                    return pd.DataFrame(
                        [{"symbol": s, "name": "", "close": 0.0, "pct_chg": float("nan"), "amount": 0.0} for s in symbols]
                    )
                return df_valid
            print("[warn] Realtime quotes empty; falling back to Eastmoney full-market panel.")
            df_em = fetch_spot_a_share(no_cache=no_cache, use_proxy=use_proxy, proxy=proxy)
            if symbols:
                df_em = df_em[df_em["symbol"].isin(set(symbols))].copy()
            df_em.attrs["spot_source"] = "eastmoney"
            return df_em
        if missing:
            placeholders = pd.DataFrame(
                [{"symbol": s, "name": "", "close": 0.0, "pct_chg": float("nan"), "amount": 0.0} for s in missing]
            )
            df_valid = pd.concat([df_valid, placeholders], ignore_index=True)
        df_valid.attrs["spot_source"] = source_used or "mixed"
        return df_valid

    # Legacy fallback (full market). If you want zero Eastmoney, pass `signals.symbols` in config.
    if not allow_eastmoney_fallback:
        raise RuntimeError("signals.symbols not provided and Eastmoney fallback disabled.")
    print("[warn] signals.symbols not provided; falling back to Eastmoney full-market spot panel.")
    df_em = fetch_spot_a_share(no_cache=no_cache, use_proxy=use_proxy, proxy=proxy)
    df_em.attrs["spot_source"] = "eastmoney"
    return df_em


def _to_tencent_code(symbol: str) -> str:
    """Convert plain 6-digit symbol to Tencent market code like sh600000/sz000001."""
    s = str(symbol).strip()
    if s.startswith(("sh", "sz")):
        return s
    if s.startswith(("6", "5", "9")):
        return f"sh{s}"
    return f"sz{s}"


def fetch_hist(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "",
    end_date: str = "",
    period: str = "day",
    adj: str = "qfq",
    hist_days: int = 0,
    timeout: int = 10,
    no_cache: bool = False,
    use_proxy: bool = False,
    proxy: str = "",
    cache_dir: str = "./cache/hist",
    cache_ttl_sec: int = 6 * 3600,
    use_cache: bool = True,
    allow_akshare_fallback: bool = True,
    manual_hist_dir: str | None = "./data/manual_hist",
) -> pd.DataFrame:
    """Fetch historical K-line data via Tencent (no Eastmoney).

    Parameters
    ----------
    symbol: 6-digit code, e.g. '600343', '000547', '588200'
    start/end: 'YYYY-MM-DD'
    end_date: alias for end
    hist_days: if provided, override start based on end (with a buffer for non-trading days)
    period: 'day' (can be extended later)
    adj: 'qfq' (forward adjusted) or '' (no adjustment)

    Returns
    -------
    DataFrame with columns: date, open, close, high, low, volume, amount
    """
    if end_date and not end:
        end = end_date

    if hist_days:
        if not end:
            end = dt.date.today().isoformat()
        try:
            end_dt = dt.date.fromisoformat(end)
        except ValueError:
            end_dt = dt.date.today()
            end = end_dt.isoformat()

        # buffer ~2x to cover non-trading days
        lookback_days = max(int(hist_days * 2), hist_days + 5)
        start = (end_dt - dt.timedelta(days=lookback_days)).isoformat()

    mem_ttl = 0
    if use_cache and not no_cache:
        mem_ttl = min(int(cache_ttl_sec), 600) if cache_ttl_sec > 0 else 0
    mem_key = f"{str(symbol).zfill(6)}|{start}|{end}|{period}|{adj}|{manual_hist_dir or ''}"
    df_mem = _hist_mem_get(mem_key, mem_ttl)
    if df_mem is not None and not df_mem.empty:
        return df_mem

    # Prefer manual history if provided.
    if manual_hist_dir:
        manual_path = Path(manual_hist_dir) / f"{str(symbol).zfill(6)}.csv"
        if manual_path.exists() and manual_path.stat().st_size > 0:
            try:
                dfm = pd.read_csv(manual_path)
                col_map = {
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "date": "date",
                    "open": "open",
                    "close": "close",
                    "high": "high",
                    "low": "low",
                    "volume": "volume",
                    "amount": "amount",
                }
                dfm = dfm.rename(columns=col_map)
                if "date" in dfm.columns:
                    dfm["date"] = dfm["date"].astype(str).str.slice(0, 10)
                    if start:
                        dfm = dfm[dfm["date"] >= start]
                    if end:
                        dfm = dfm[dfm["date"] <= end]
                keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount"] if c in dfm.columns]
                if "date" in keep:
                    out = dfm[keep].copy()
                    for c in ["open", "close", "high", "low", "volume", "amount"]:
                        if c in out.columns:
                            out[c] = pd.to_numeric(out[c], errors="coerce")
                    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
                    if not out.empty:
                        _hist_mem_set(mem_key, out)
                        return out
            except Exception:
                pass

    cache_path = None
    if use_cache and not no_cache:
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{symbol}_{start}_{end}_{period}_{adj}")
        cache_path = Path(cache_dir) / f"{safe}.csv"
        try:
            if cache_path.exists():
                age = time.time() - cache_path.stat().st_mtime
                if age <= cache_ttl_sec:
                    dfc = pd.read_csv(cache_path)
                    if dfc is not None and not dfc.empty:
                        _hist_mem_set(mem_key, dfc)
                        return dfc
        except Exception:
            pass

    mcode = _to_tencent_code(symbol)
    per = period
    adj_key_prefix = adj if adj else ""

    # Tencent endpoint
    # Example:
    # https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600343,day,2020-01-01,2026-02-03,640,qfq
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    # 640 is the max count for this interface in a single call (commonly used)
    count = 640
    param = f"{mcode},{per},{start},{end},{count},{adj_key_prefix}"

    session = _build_session(use_proxy, proxy)
    try:
        r = session.get(url, params={"param": param}, timeout=timeout)
        r.raise_for_status()
        # Some responses may include leading junk; be defensive.
        text = r.text.strip()
        try:
            j = r.json()
        except Exception:
            j = json.loads(text)
    except Exception as e:
        if not allow_akshare_fallback:
            print(f"[warn] Tencent hist request failed and AkShare fallback disabled: {e}")
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume", "amount"])
        # Network error: fall back to AkShare if possible
        try:
            with _with_proxy_env(proxy if proxy else (None if not use_proxy else os.environ.get("http_proxy") or os.environ.get("https_proxy"))):
                df = _fetch_hist_akshare(symbol, start=start, end=end, period=period, adj=adj)
                if cache_path is not None and df is not None and not df.empty:
                    try:
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                        df.to_csv(cache_path, index=False, encoding="utf-8")
                    except Exception:
                        pass
                if df is not None and not df.empty:
                    _hist_mem_set(mem_key, df)
                return df
        except Exception as e2:
            print(f"[warn] Tencent hist request failed and AkShare fallback failed: {e2}")
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume", "amount"])

    data = (j.get("data") or {}).get(mcode) or {}

    # Possible fields: 'qfqday', 'day', 'qfqweek', etc.
    k_key = None
    prefer = []
    if adj_key_prefix:
        prefer.append(f"{adj_key_prefix}{per}")
    prefer.append(per)

    for kk in prefer:
        if kk in data:
            k_key = kk
            break

    if k_key is None:
        # Try any key that ends with period
        for kk in data.keys():
            if kk.endswith(per):
                k_key = kk
                break

    rows = data.get(k_key) if k_key else None
    if not rows:
        if not allow_akshare_fallback:
            print("[warn] Tencent hist empty and AkShare fallback disabled.")
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume", "amount"])
        # Fallback to AkShare history if available
        try:
            with _with_proxy_env(proxy if proxy else (None if not use_proxy else os.environ.get("http_proxy") or os.environ.get("https_proxy"))):
                df = _fetch_hist_akshare(symbol, start=start, end=end, period=period, adj=adj)
                if cache_path is not None and df is not None and not df.empty:
                    try:
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                        df.to_csv(cache_path, index=False, encoding="utf-8")
                    except Exception:
                        pass
                if df is not None and not df.empty:
                    _hist_mem_set(mem_key, df)
                return df
        except Exception as e:
            print(f"[warn] Tencent hist empty and AkShare fallback failed: {e}")
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume", "amount"])

    # Each row typically: [date, open, close, high, low, volume, amount]
    out = []
    for row in rows:
        if not row or len(row) < 6:
            continue
        date_str = row[0]
        o = row[1]
        c = row[2]
        h = row[3]
        l = row[4]
        v = row[5]
        amt = row[6] if len(row) > 6 else 0
        out.append([date_str, o, c, h, l, v, amt])

    df = pd.DataFrame(out, columns=["date", "open", "close", "high", "low", "volume", "amount"])

    for c in ["open", "close", "high", "low", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if cache_path is not None and df is not None and not df.empty:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False, encoding="utf-8")
        except Exception:
            pass
    if df is not None and not df.empty:
        _hist_mem_set(mem_key, df)
    return df


def _fetch_hist_akshare(symbol: str, start: str, end: str, period: str = "day", adj: str = "qfq") -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError("AkShare not available for fallback.") from e

    ak_period = "daily" if period in ("day", "daily") else period
    start_s = start.replace("-", "") if start else ""
    end_s = end.replace("-", "") if end else ""
    adjust = adj if adj else ""

    if not hasattr(ak, "stock_zh_a_hist"):
        raise RuntimeError("AkShare missing stock_zh_a_hist for fallback.")

    df = ak.stock_zh_a_hist(
        symbol=str(symbol),
        period=ak_period,
        start_date=start_s,
        end_date=end_s,
        adjust=adjust,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume", "amount"])

    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
    }
    df = df.rename(columns=col_map)

    keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount"] if c in df.columns]
    if "date" not in keep:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume", "amount"])

    out = df[keep].copy()
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out
