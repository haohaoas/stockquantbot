from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

_SECTOR_CACHE: Dict[str, str] | None = None
_SECTOR_CACHE_PATH: str | None = None


def load_sector_map(path: str) -> Dict[str, str]:
    global _SECTOR_CACHE, _SECTOR_CACHE_PATH
    if _SECTOR_CACHE is not None and _SECTOR_CACHE_PATH == path:
        return _SECTOR_CACHE
    p = Path(path)
    if not p.exists():
        _SECTOR_CACHE = {}
        _SECTOR_CACHE_PATH = path
        return _SECTOR_CACHE
    try:
        df = pd.read_csv(p)
    except Exception:
        _SECTOR_CACHE = {}
        _SECTOR_CACHE_PATH = path
        return _SECTOR_CACHE
    if "symbol" not in df.columns or "sector" not in df.columns:
        _SECTOR_CACHE = {}
        _SECTOR_CACHE_PATH = path
        return _SECTOR_CACHE
    mapping: Dict[str, str] = {}
    for _, row in df[["symbol", "sector"]].iterrows():
        sym = str(row["symbol"]).strip().zfill(6)
        sec = str(row["sector"]).strip()
        if not sym or not sec:
            continue
        if sym not in mapping:
            mapping[sym] = sec
    _SECTOR_CACHE = mapping
    _SECTOR_CACHE_PATH = path
    return mapping


def apply_sector_map(df: pd.DataFrame, path: str) -> pd.DataFrame:
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    mapping = load_sector_map(path)
    if not mapping:
        return df
    out = df.copy()
    out["sector"] = out["symbol"].astype(str).str.zfill(6).map(mapping).fillna("")
    return out
