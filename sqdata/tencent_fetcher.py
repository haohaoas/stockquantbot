# sqdata/tencent_fetcher.py
from __future__ import annotations
import time
import requests
from dataclasses import dataclass
from typing import Optional, Dict

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0"})


def _build_session(use_proxy: bool, proxy: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
        session.trust_env = False
    else:
        session.trust_env = use_proxy
    return session

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
    # 6/5/9 -> sh, 0/3/1 -> sz (ETF: 51/58/56 多数是sh，15/16 多数sz)
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"

def _looks_like_code(val: str, symbol: str) -> bool:
    s = str(val).strip()
    if not s.isdigit():
        return False
    return s.zfill(6) == str(symbol).zfill(6)


def _fetch_realtime_once(code: str, timeout: float, use_proxy: bool, proxy: str, scheme: str = "https") -> RealtimeQuote:
    mcode = _to_market_code(code)
    url = f"{scheme}://qt.gtimg.cn/q={mcode}"

    session = _build_session(use_proxy, proxy) if (use_proxy or proxy) else _session
    r = session.get(url, timeout=timeout)
    r.encoding = "gbk"
    text = r.text.strip()

    # v_sh600343="1~航天动力~37.25~36.51~37.60~38.51~34.00~..."
    if '="' not in text:
        raise RuntimeError(f"unexpected response: {text[:120]}")
    payload = text.split('="', 1)[1].rsplit('";', 1)[0]
    arr = payload.split("~")

    name = arr[1]
    offset = 1 if (len(arr) > 3 and _looks_like_code(arr[2], code)) else 0
    price = float(arr[2 + offset] or 0)
    prev_close = float(arr[3 + offset] or 0)
    open_ = float(arr[4 + offset] or 0)
    high = float(arr[5 + offset] or 0)
    low = float(arr[6 + offset] or 0)

    def _f(i: int) -> float:
        try:
            return float(arr[i] or 0)
        except Exception:
            return 0.0

    volume = _f(7 + offset)
    amount = _f(8 + offset)

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

    # Try HTTP scheme as fallback (some networks block HTTPS CONNECT)
    try:
        q = _fetch_realtime_once(code, timeout, use_proxy, proxy, scheme="http")
        if (q.price > 0 and q.name) or not (use_proxy or proxy):
            return q
    except Exception:
        if not allow_direct_fallback or not (use_proxy or proxy):
            raise

    # Fallback to direct
    try:
        return _fetch_realtime_once(code, timeout, False, "", scheme="https")
    except Exception:
        return _fetch_realtime_once(code, timeout, False, "", scheme="http")

# 简单缓存（避免你 CLI 连续刷）
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
