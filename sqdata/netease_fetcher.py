from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import requests

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0"})


@dataclass
class RealtimeQuote:
    code: str
    name: str
    price: float
    prev_close: float
    open: float
    high: float
    low: float
    volume: float
    amount: float
    pct: float
    ts: int


def _build_session(use_proxy: bool, proxy: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
        session.trust_env = False
    else:
        session.trust_env = use_proxy
    return session


def _to_netease_code(code: str) -> str:
    s = str(code).strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        s = s[2:]
    if len(s) == 7 and s[0] in "012":
        return s
    if s.startswith(("6", "5", "9")):
        return "0" + s
    if s.startswith(("8", "4")):
        return "2" + s
    return "1" + s


def _normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if s.startswith(("sh", "sz", "bj")):
        s = s[2:]
    if len(s) >= 6:
        return s[-6:]
    return s.zfill(6)


def _parse_jsonp(text: str) -> dict:
    raw = text.strip()
    if not raw:
        return {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"unexpected response: {raw[:120]}")
    payload = raw[start : end + 1]
    return json.loads(payload)


def _pct_from_percent(percent: float, price: float, prev_close: float) -> float:
    pct = 0.0
    try:
        p = float(percent)
        if abs(p) <= 1.5:
            pct = p * 100
        else:
            pct = p
    except Exception:
        pct = 0.0
    if (not pct) and prev_close > 0 and price > 0:
        pct = (price - prev_close) / prev_close * 100
    return pct


def fetch_realtime_batch(
    symbols: List[str],
    timeout: int = 10,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
) -> pd.DataFrame:
    codes = [_to_netease_code(s) for s in symbols]
    if not codes:
        return pd.DataFrame()

    chunk_size = 200
    session = _build_session(use_proxy, proxy)
    direct_session = _build_session(False, "")
    rows = []

    for i in range(0, len(codes), chunk_size):
        chunk = codes[i : i + chunk_size]
        base = ",".join(chunk) + ",money.api"
        url_https = f"https://api.money.126.net/data/feed/{base}?callback=_ntes_quote_callback"
        url_http = f"http://api.money.126.net/data/feed/{base}?callback=_ntes_quote_callback"

        r = None
        try:
            r = session.get(url_https, timeout=timeout)
        except Exception:
            try:
                r = session.get(url_http, timeout=timeout)
            except Exception:
                if allow_direct_fallback and (use_proxy or proxy):
                    try:
                        r = direct_session.get(url_https, timeout=timeout)
                    except Exception:
                        r = direct_session.get(url_http, timeout=timeout)
                else:
                    raise

        if r is None:
            continue
        data = _parse_jsonp(r.text)
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            symbol = _normalize_symbol(val.get("symbol") or key)
            name = str(val.get("name") or "").strip()
            price = float(val.get("price") or val.get("last") or 0)
            prev_close = float(val.get("yestclose") or val.get("preclose") or 0)
            open_ = float(val.get("open") or 0)
            high = float(val.get("high") or 0)
            low = float(val.get("low") or 0)
            volume = float(val.get("volume") or 0)
            amount = float(val.get("turnover") or val.get("amount") or 0)
            pct = _pct_from_percent(val.get("percent") or 0, price, prev_close)
            market = str(val.get("type") or "").lower().strip()
            if market not in ("sh", "sz", "bj"):
                if symbol.startswith(("6", "5", "9")):
                    market = "sh"
                elif symbol.startswith(("8", "4")):
                    market = "bj"
                else:
                    market = "sz"

            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "close": price,
                    "prev_close": prev_close,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "pct_chg": round(float(pct), 2),
                    "volume": volume,
                    "amount": amount,
                    "market": market,
                }
            )

        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if df.empty:
        if allow_direct_fallback and (use_proxy or proxy):
            return fetch_realtime_batch(symbols, timeout=timeout, use_proxy=False, proxy="", allow_direct_fallback=False)
        return df

    df["symbol"] = df["symbol"].astype(str)
    for c in ["close", "pct_chg", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if allow_direct_fallback and (use_proxy or proxy):
        valid = df[pd.to_numeric(df["close"], errors="coerce") > 0]
        if valid.empty:
            return fetch_realtime_batch(symbols, timeout=timeout, use_proxy=False, proxy="", allow_direct_fallback=False)

    return df


def fetch_realtime(
    code: str,
    timeout: float = 3.0,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
) -> RealtimeQuote:
    df = fetch_realtime_batch([code], timeout=int(max(3, timeout)), use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=allow_direct_fallback)
    if df.empty:
        raise RuntimeError("netease realtime empty")
    row = df.iloc[0]
    price = float(row.get("close") or 0)
    prev_close = float(row.get("prev_close") or 0)
    pct = float(row.get("pct_chg") or 0)
    return RealtimeQuote(
        code=str(row.get("symbol") or code),
        name=str(row.get("name") or ""),
        price=price,
        prev_close=prev_close,
        open=float(row.get("open") or 0),
        high=float(row.get("high") or 0),
        low=float(row.get("low") or 0),
        volume=float(row.get("volume") or 0),
        amount=float(row.get("amount") or 0),
        pct=float(pct),
        ts=int(time.time()),
    )


_CACHE: Dict[str, RealtimeQuote] = {}
_CACHE_TS: Dict[str, float] = {}


def fetch_realtime_cached(
    code: str,
    ttl: float = 1.0,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
) -> RealtimeQuote:
    now = time.time()
    if code in _CACHE and (now - _CACHE_TS.get(code, 0)) < ttl:
        return _CACHE[code]
    q = fetch_realtime(code, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=allow_direct_fallback)
    _CACHE[code] = q
    _CACHE_TS[code] = now
    return q
