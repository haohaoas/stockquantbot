from __future__ import annotations

import re
from datetime import date
from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache" / "hist"
OUT_DIR = BASE_DIR / "data" / "manual_hist"


def _parse_end_date(name: str) -> date | None:
    # filename pattern: 000001_2025-02-06_2026-02-06_day_qfq.csv
    m = re.match(r"^([0-9]{6})_([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{4}-[0-9]{2}-[0-9]{2})_.*[.]csv$", name)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(3))
    except Exception:
        return None


def export_from_cache() -> int:
    if not CACHE_DIR.exists():
        raise RuntimeError(f"cache dir not found: {CACHE_DIR}")

    latest_by_symbol: dict[str, tuple[Path, date | None, float]] = {}
    for p in CACHE_DIR.glob("*.csv"):
        name = p.name
        m = re.match(r"^([0-9]{6})_.*[.]csv$", name)
        if not m:
            continue
        sym = m.group(1)
        end_dt = _parse_end_date(name)
        mtime = p.stat().st_mtime
        cur = latest_by_symbol.get(sym)
        if cur is None:
            latest_by_symbol[sym] = (p, end_dt, mtime)
            continue
        cur_p, cur_end, cur_mtime = cur
        if end_dt and (cur_end is None or end_dt > cur_end):
            latest_by_symbol[sym] = (p, end_dt, mtime)
        elif end_dt == cur_end and mtime > cur_mtime:
            latest_by_symbol[sym] = (p, end_dt, mtime)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for sym, (p, _, _) in latest_by_symbol.items():
        out = OUT_DIR / f"{sym}.csv"
        shutil.copyfile(p, out)
        n += 1
    return n


def main() -> None:
    n = export_from_cache()
    print(f"[ok] exported {n} symbols to {OUT_DIR}")


if __name__ == "__main__":
    main()
