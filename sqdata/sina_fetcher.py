from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import requests


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


def _to_market_code(code: str) -> str:
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _build_session(use_proxy: bool, proxy: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
        session.trust_env = False
    else:
        session.trust_env = use_proxy
    return session


def _fetch_realtime_once(code: str, timeout: float, use_proxy: bool, proxy: str, scheme: str = "https") -> RealtimeQuote:
    mcode = _to_market_code(code)
    url = f"{scheme}://hq.sinajs.cn/list={mcode}"

    session = _build_session(use_proxy, proxy)
    r = session.get(url, timeout=timeout)
    r.encoding = "gbk"
    text = r.text.strip()

    if '="' not in text:
        raise RuntimeError(f"unexpected response: {text[:120]}")
    payload = text.split('="', 1)[1].rsplit('";', 1)[0]
    arr = payload.split(",")

    # name, open, prev_close, price, high, low, buy, sell, vol, amount, ...
    name = arr[0] if len(arr) > 0 else ""
    open_ = float(arr[1] or 0) if len(arr) > 1 else 0.0
    prev_close = float(arr[2] or 0) if len(arr) > 2 else 0.0
    price = float(arr[3] or 0) if len(arr) > 3 else 0.0
    high = float(arr[4] or 0) if len(arr) > 4 else 0.0
    low = float(arr[5] or 0) if len(arr) > 5 else 0.0
    volume = float(arr[8] or 0) if len(arr) > 8 else 0.0
    amount = float(arr[9] or 0) if len(arr) > 9 else 0.0

    pct = 0.0
    if prev_close > 0:
        pct = (price - prev_close) / prev_close * 100

    return RealtimeQuote(
        code=code,
        name=name,
        price=price,
        prev_close=prev_close,
        open=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        pct=round(pct, 2),
        ts=int(time.time()),
    )

def fetch_realtime(
    code: str,
    timeout: float = 3.0,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
) -> RealtimeQuote:
    try:
        q = _fetch_realtime_once(code, timeout, use_proxy, proxy, scheme="https")
        if (q.price > 0 and q.name) or not (use_proxy or proxy):
            return q
    except Exception:
        pass

    try:
        q = _fetch_realtime_once(code, timeout, use_proxy, proxy, scheme="http")
        if (q.price > 0 and q.name) or not (use_proxy or proxy):
            return q
    except Exception:
        if not allow_direct_fallback or not (use_proxy or proxy):
            raise

    try:
        return _fetch_realtime_once(code, timeout, False, "", scheme="https")
    except Exception:
        return _fetch_realtime_once(code, timeout, False, "", scheme="http")


def fetch_realtime_batch(
    symbols: List[str],
    timeout: int = 10,
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
) -> pd.DataFrame:
    def to_mcode(sym: str) -> str:
        s = str(sym).strip()
        if s.startswith(("sh", "sz")):
            return s
        if s.startswith(("6", "5", "9")):
            return f"sh{s}"
        return f"sz{s}"

    mcodes = [to_mcode(s) for s in symbols]
    chunk_size = 200
    session = _build_session(use_proxy, proxy)

    rows = []

    def _f(arr: list[str], idx: int) -> float:
        try:
            return float(arr[idx] or 0.0)
        except Exception:
            return 0.0

    for i in range(0, len(mcodes), chunk_size):
        chunk = mcodes[i : i + chunk_size]
        url = "https://hq.sinajs.cn/list=" + ",".join(chunk)
        try:
            r = session.get(url, timeout=timeout)
        except Exception:
            r = session.get("http://hq.sinajs.cn/list=" + ",".join(chunk), timeout=timeout)
        r.encoding = "gbk"
        text = r.text

        for line in text.splitlines():
            line = line.strip()
            if '="' not in line:
                continue
            left, payload = line.split('="', 1)
            payload = payload.rstrip('";')
            key = left.split("hq_str_", 1)[1]
            arr = payload.split(",")

            name = arr[0] if len(arr) > 0 else ""
            open_ = float(arr[1] or 0) if len(arr) > 1 else 0.0
            prev_close = float(arr[2] or 0) if len(arr) > 2 else 0.0
            price = float(arr[3] or 0) if len(arr) > 3 else 0.0
            high = float(arr[4] or 0) if len(arr) > 4 else 0.0
            low = float(arr[5] or 0) if len(arr) > 5 else 0.0
            volume = float(arr[8] or 0) if len(arr) > 8 else 0.0
            amount = float(arr[9] or 0) if len(arr) > 9 else 0.0
            pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0

            # 新浪字段里包含买1~买5/卖1~卖5委托量，可用于委比。
            bid_sum = _f(arr, 10) + _f(arr, 12) + _f(arr, 14) + _f(arr, 16) + _f(arr, 18)
            ask_sum = _f(arr, 20) + _f(arr, 22) + _f(arr, 24) + _f(arr, 26) + _f(arr, 28)
            weibi = float("nan")
            if (bid_sum + ask_sum) > 0:
                weibi = (bid_sum - ask_sum) / (bid_sum + ask_sum) * 100.0

            market = "sh" if key.startswith("sh") else "sz"
            symbol = key[2:]

            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "close": price,
                    "pct_chg": round(pct, 2),
                    "volume": volume,
                    "amount": amount,
                    "weibi": weibi,
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
    for c in ["close", "pct_chg", "volume", "amount", "weibi"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If all close are invalid, retry direct once
    if allow_direct_fallback and (use_proxy or proxy):
        valid = df[pd.to_numeric(df["close"], errors="coerce") > 0]
        if valid.empty:
            return fetch_realtime_batch(symbols, timeout=timeout, use_proxy=False, proxy="", allow_direct_fallback=False)

    return df


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
