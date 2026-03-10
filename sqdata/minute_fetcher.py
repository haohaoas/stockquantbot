from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import requests


def _to_tencent_code(symbol: str) -> str:
    s = str(symbol).strip()
    if s.startswith(("sh", "sz")):
        return s
    if s.startswith(("6", "5", "9")):
        return f"sh{s}"
    return f"sz{s}"


def _parse_mkline_payload(items: list[Any]) -> pd.DataFrame:
    rows = []
    for item in items:
        if isinstance(item, list):
            parts = item
        else:
            parts = str(item).split(",")
        if len(parts) < 6:
            continue
        ts = str(parts[0]).strip()
        dt_val = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y%m%d%H%M"):
            try:
                dt_val = dt.datetime.strptime(ts, fmt)
                break
            except Exception:
                continue
        try:
            open_ = float(parts[1])
            close = float(parts[2])
            high = float(parts[3])
            low = float(parts[4])
        except Exception:
            continue
        vol = None
        amt = None
        try:
            if len(parts) > 5 and str(parts[5]) != "":
                vol = float(parts[5])
            if len(parts) > 7 and str(parts[7]) != "":
                amt = float(parts[7])
            elif len(parts) > 6 and str(parts[6]) != "" and not isinstance(parts[6], dict):
                amt = float(parts[6])
        except Exception:
            pass
        rows.append(
            {
                "datetime": dt_val or ts,
                "date": dt_val.strftime("%Y-%m-%d") if dt_val else ts[:10],
                "open": open_,
                "close": close,
                "high": high,
                "low": low,
                "volume": vol,
                "amount": amt,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def _build_session(use_proxy: bool, proxy: str) -> requests.Session:
    session = requests.Session()
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
        session.trust_env = False
    else:
        session.trust_env = bool(use_proxy)
    return session


def _to_biying_code(symbol: str) -> str:
    s = str(symbol).strip().lower()
    if s.startswith(("sh", "sz")):
        s = s[2:]
    s = s.zfill(6)
    market = "sh" if s.startswith(("5", "6", "9")) else "sz"
    return f"{s}.{market}"


def _biying_interval(interval: str) -> str:
    m = {
        "m1": "1",
        "m5": "5",
        "m15": "15",
        "m30": "30",
        "m60": "60",
        "1": "1",
        "5": "5",
        "15": "15",
        "30": "30",
        "60": "60",
    }
    v = m.get(str(interval).strip().lower())
    if not v:
        raise ValueError(f"unsupported interval for biying: {interval}")
    return v


def _parse_biying_time(val: Any) -> dt.datetime | None:
    if isinstance(val, (int, float)):
        try:
            iv = int(val)
            # second or millisecond epoch
            if iv > 10_000_000_000:
                return dt.datetime.fromtimestamp(iv / 1000.0)
            if iv > 1_000_000_000:
                return dt.datetime.fromtimestamp(iv)
        except Exception:
            pass
    s = str(val or "").strip()
    if not s:
        return None
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d%H:%M",
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y-%m-%d",
        "%Y%m%d",
    )
    for fmt in fmts:
        try:
            return dt.datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _normalize_biying_rows(payload: Any) -> pd.DataFrame:
    data = payload
    if isinstance(payload, dict):
        for k in ("data", "result", "results", "rows", "list"):
            v = payload.get(k)
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []
    for item in data:
        t = None
        o = c = h = l = v = a = float("nan")

        if isinstance(item, dict):
            t = _parse_biying_time(item.get("t") or item.get("time") or item.get("datetime") or item.get("dt"))
            o = pd.to_numeric(item.get("o", item.get("open")), errors="coerce")
            c = pd.to_numeric(item.get("c", item.get("close")), errors="coerce")
            h = pd.to_numeric(item.get("h", item.get("high")), errors="coerce")
            l = pd.to_numeric(item.get("l", item.get("low")), errors="coerce")
            v = pd.to_numeric(item.get("v", item.get("volume")), errors="coerce")
            a = pd.to_numeric(item.get("a", item.get("amount")), errors="coerce")
        elif isinstance(item, (list, tuple)):
            parts = list(item)
            if len(parts) >= 5:
                t = _parse_biying_time(parts[0])
                o = pd.to_numeric(parts[1], errors="coerce")
                h = pd.to_numeric(parts[2], errors="coerce")
                l = pd.to_numeric(parts[3], errors="coerce")
                c = pd.to_numeric(parts[4], errors="coerce")
                if len(parts) > 5:
                    v = pd.to_numeric(parts[5], errors="coerce")
                if len(parts) > 6:
                    a = pd.to_numeric(parts[6], errors="coerce")
        else:
            continue

        if t is None:
            continue
        if not (pd.notna(o) and pd.notna(c) and pd.notna(h) and pd.notna(l)):
            continue

        rows.append(
            {
                "datetime": t,
                "date": t.strftime("%Y-%m-%d"),
                "open": float(o),
                "close": float(c),
                "high": float(h),
                "low": float(l),
                "volume": float(v) if pd.notna(v) else float("nan"),
                "amount": float(a) if pd.notna(a) else float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    return out


def fetch_biying_minute_history(
    symbol: str,
    interval: str = "m5",
    *,
    licence: str,
    start: str = "",
    end: str = "",
    limit: int = 0,
    base_url: str = "http://api.biyingapi.com",
    use_proxy: bool = False,
    proxy: str = "",
    timeout: int = 12,
) -> pd.DataFrame:
    """Fetch historical minute K-line from 必盈.

    API pattern:
    /hsstock/history/{code.market}/{level}/n/{licence}?st=YYYYMMDD&et=YYYYMMDD&lt=1000
    """
    if not licence:
        return pd.DataFrame()

    code = _to_biying_code(symbol)
    level = _biying_interval(interval)
    base = str(base_url or "http://api.biyingapi.com").rstrip("/")
    url = f"{base}/hsstock/history/{code}/{level}/n/{licence}"
    params: dict[str, Any] = {}
    if start:
        params["st"] = str(start)
    if end:
        params["et"] = str(end)
    if int(limit or 0) > 0:
        params["lt"] = int(limit)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
    }

    def _request(_use_proxy: bool, _proxy: str) -> pd.DataFrame:
        session = _build_session(_use_proxy, _proxy)
        r = session.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        return _normalize_biying_rows(payload)

    try:
        df = _request(use_proxy, proxy)
        if not df.empty:
            return df
    except Exception:
        pass

    if use_proxy or proxy:
        try:
            return _request(False, "")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def fetch_tencent_minute(
    symbol: str,
    interval: str = "m5",
    limit: int = 320,
    *,
    use_proxy: bool = False,
    proxy: str = "",
    timeout: int = 10,
) -> pd.DataFrame:
    """Fetch recent minute K-line from Tencent mkline endpoint.

    interval: m1/m5/m15/m30/m60
    limit: max bars returned by endpoint (typically 320)
    """
    mcode = _to_tencent_code(symbol)
    params = {"param": f"{mcode},{interval},,{int(limit)}"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://quote.gtimg.cn/",
    }

    def _request(url: str, *, use_proxy: bool, proxy: str) -> pd.DataFrame:
        session = _build_session(use_proxy, proxy)
        r = session.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        payload = (((data.get("data") or {}).get(mcode) or {}).get(interval)) or []
        if not payload:
            return pd.DataFrame()
        return _parse_mkline_payload(payload)

    urls = [
        "https://ifzq.gtimg.cn/appstock/app/kline/mkline",
        "https://web.ifzq.gtimg.cn/appstock/app/kline/mkline",
        "http://ifzq.gtimg.cn/appstock/app/kline/mkline",
    ]

    last_df = pd.DataFrame()
    for u in urls:
        try:
            df = _request(u, use_proxy=use_proxy, proxy=proxy)
            if not df.empty:
                return df
            last_df = df
        except Exception:
            continue

    # If proxy used and no data, try direct once.
    if (use_proxy or proxy):
        for u in urls:
            try:
                df = _request(u, use_proxy=False, proxy="")
                if not df.empty:
                    return df
                last_df = df
            except Exception:
                continue

    return last_df
