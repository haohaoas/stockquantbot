from __future__ import annotations

import os
import time
import random
from pathlib import Path

import pandas as pd


def _apply_proxy_from_config(config_path: str = "config/default.yaml") -> None:
    try:
        import yaml
    except Exception:
        return

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    signals = cfg.get("signals", {}) or {}
    use_proxy = bool(signals.get("use_proxy", False))
    proxy = str(signals.get("proxy", "") or "")

    if use_proxy and proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy


def _fetch_trade_calendar(max_retries: int = 5, base_sleep: float = 0.8) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError("AkShare not installed. Run `pip install akshare`.") from e

    candidates = [
        "tool_trade_date_hist_sina",
        "tool_trade_date_hist_sina",
    ]

    last_err: Exception | None = None
    for i in range(max_retries):
        for name in candidates:
            fn = getattr(ak, name, None)
            if not callable(fn):
                continue
            try:
                df = fn()
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                last_err = e

        sleep_s = base_sleep * (1.6 ** i) + random.uniform(0.0, 0.6)
        time.sleep(min(sleep_s, 6.0))

    raise RuntimeError(f"AkShare trade calendar fetch failed after {max_retries} retries: {last_err}")


def update_trade_calendar(out_path: str = "./data/trade_calendar.csv") -> int:
    df = _fetch_trade_calendar()
    col_map = {"trade_date": "trade_date", "日期": "trade_date", "date": "trade_date"}
    df = df.rename(columns=col_map)
    if "trade_date" not in df.columns:
        raise RuntimeError(f"Unexpected columns: {list(df.columns)}")

    df["trade_date"] = df["trade_date"].astype(str).str.slice(0, 10)
    out = df[["trade_date"]].drop_duplicates().sort_values("trade_date")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return int(len(out))


def main() -> None:
    _apply_proxy_from_config()
    try:
        n = update_trade_calendar()
        print(f"[ok] trade calendar saved: ./data/trade_calendar.csv (n={n})")
    except Exception as e:
        p = Path("./data/trade_calendar.csv")
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
                print(f"[warn] update failed, keep existing calendar (n={len(df)}): {e}")
            except Exception:
                print(f"[warn] update failed, keep existing calendar: {e}")
        else:
            raise


if __name__ == "__main__":
    main()
