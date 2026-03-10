# sqdata/fetcher.py
from __future__ import annotations
from typing import Literal

from .tencent_fetcher import fetch_realtime_cached as fetch_realtime_tencent_cached, RealtimeQuote as TencentRealtimeQuote
from .sina_fetcher import fetch_realtime_cached as fetch_realtime_sina_cached, RealtimeQuote as SinaRealtimeQuote
from .netease_fetcher import fetch_realtime_cached as fetch_realtime_netease_cached, RealtimeQuote as NeteaseRealtimeQuote

from . import akshare_fetcher as _ak
import inspect
from typing import Any, Callable


def _resolve_akshare_kline_func() -> Callable[..., Any]:
    """Resolve a kline/history function from akshare_fetcher.py.

    Because this repo may name the function differently, we try common candidates and then
    fall back to any callable containing 'kline' in its name.
    """
    candidates = [
        "fetch_kline",
        "get_kline",
        "kline",
        "fetch_hist_kline",
        "fetch_daily_kline",
        "get_daily_kline",
        "fetch_ohlcv",
        "get_ohlcv",
        "fetch_history",
        "get_history",
    ]
    for name in candidates:
        fn = getattr(_ak, name, None)
        if callable(fn):
            return fn

    for name in dir(_ak):
        if "kline" in name.lower():
            fn = getattr(_ak, name, None)
            if callable(fn):
                return fn

    callables = [
        n for n in dir(_ak)
        if callable(getattr(_ak, n, None)) and not n.startswith("_")
    ]
    raise RuntimeError(
        "No kline/history function found in sqdata/akshare_fetcher.py. "
        f"Available callables: {callables}"
    )


def _call_kline(fn: Callable[..., Any], code: str, start: str, end: str) -> Any:
    """Call resolved kline function with best-effort argument mapping."""
    sig = inspect.signature(fn)
    params = sig.parameters

    # Common parameter name mapping
    kwargs: dict[str, Any] = {}

    # code / symbol
    if "code" in params:
        kwargs["code"] = code
    elif "symbol" in params:
        kwargs["symbol"] = code
    elif "ts_code" in params:
        kwargs["ts_code"] = code
    else:
        # If the function is positional-only for code, we'll pass it positionally.
        pass

    # start
    if "start" in params:
        kwargs["start"] = start
    elif "start_date" in params:
        kwargs["start_date"] = start
    elif "begin" in params:
        kwargs["begin"] = start
    elif "beg" in params:
        kwargs["beg"] = start

    # end
    if "end" in params:
        kwargs["end"] = end
    elif "end_date" in params:
        kwargs["end_date"] = end
    elif "finish" in params:
        kwargs["finish"] = end

    # Try kwargs first; if code wasn't mapped, pass it positionally.
    try:
        if any(k in kwargs for k in ("code", "symbol", "ts_code")):
            return fn(**kwargs)
        return fn(code, **kwargs)
    except TypeError:
        # Last resort: try simplest positional call patterns
        try:
            return fn(code, start, end)
        except Exception:
            return fn(code)

Provider = Literal["tencent", "sina", "netease", "auto"]

def get_realtime(
    code: str,
    provider: Provider = "auto",
    use_proxy: bool = False,
    proxy: str = "",
    allow_direct_fallback: bool = True,
) -> TencentRealtimeQuote | SinaRealtimeQuote | NeteaseRealtimeQuote:
    if provider == "tencent":
        return fetch_realtime_tencent_cached(code, ttl=1.0, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=allow_direct_fallback)
    if provider == "sina":
        return fetch_realtime_sina_cached(code, ttl=1.0, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=allow_direct_fallback)
    if provider == "netease":
        return fetch_realtime_netease_cached(code, ttl=1.0, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=allow_direct_fallback)

    # auto: tencent -> netease -> sina
    try:
        return fetch_realtime_tencent_cached(code, ttl=1.0, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=allow_direct_fallback)
    except Exception:
        try:
            return fetch_realtime_netease_cached(code, ttl=1.0, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=allow_direct_fallback)
        except Exception:
            return fetch_realtime_sina_cached(code, ttl=1.0, use_proxy=use_proxy, proxy=proxy, allow_direct_fallback=allow_direct_fallback)

def get_kline(code: str, start: str, end: str):
    # 历史/日线继续用 akshare（最省事），但具体函数名可能不同
    fn = _resolve_akshare_kline_func()
    return _call_kline(fn, code=code, start=start, end=end)
