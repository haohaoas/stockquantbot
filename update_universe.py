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


def _fetch_with_retries(max_retries: int = 5, base_sleep: float = 0.8):
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError("AkShare not installed. Run `pip install akshare`.") from e

    if not hasattr(ak, "stock_info_a_code_name"):
        raise RuntimeError("AkShare missing stock_info_a_code_name.")

    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                return df
            last_err = RuntimeError("AkShare returned empty universe list.")
        except Exception as e:
            last_err = e

        # backoff
        sleep_s = base_sleep * (1.6 ** i) + random.uniform(0.0, 0.6)
        time.sleep(min(sleep_s, 6.0))

    raise RuntimeError(f"AkShare universe fetch failed after {max_retries} retries: {last_err}")


def update_universe(out_path: str = "./data/universe.csv") -> int:
    existing = Path(out_path)
    df = _fetch_with_retries()

    col_map = {"代码": "symbol", "名称": "name", "code": "symbol", "name": "name"}
    df = df.rename(columns=col_map)
    if "symbol" not in df.columns:
        raise RuntimeError(f"Unexpected columns: {list(df.columns)}")

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    if "name" not in df.columns:
        df["name"] = ""

    out = df[["symbol", "name"]].drop_duplicates().sort_values("symbol")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return int(len(out))


def main() -> None:
    _apply_proxy_from_config()
    try:
        n = update_universe()
        print(f"[ok] universe saved: ./data/universe.csv (n={n})")
    except Exception as e:
        # If existing file is present, keep it and warn.
        p = Path("./data/universe.csv")
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
                print(f"[warn] update failed, keep existing universe (n={len(df)}): {e}")
            except Exception:
                print(f"[warn] update failed, keep existing universe: {e}")
        else:
            raise


if __name__ == "__main__":
    main()
