from __future__ import annotations

import csv
import re
from pathlib import Path


def _normalize_symbol(s: str) -> str:
    t = str(s or "").strip()
    if not t:
        return ""
    if t.startswith(("sh", "sz")):
        t = t[2:]
    if "." in t:
        t = t.split(".", 1)[0]
    if re.fullmatch(r"\d{6}", t):
        return t
    return ""


def load_universe_symbols(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []

    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    symbols: list[str] = []

    # Try CSV with headers first.
    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fields = [h.lower() for h in reader.fieldnames]
                keys = []
                for k in ("symbol", "code", "ts_code", "ticker"):
                    if k in fields:
                        keys.append(reader.fieldnames[fields.index(k)])
                if keys:
                    for row in reader:
                        for k in keys:
                            sym = _normalize_symbol(row.get(k, ""))
                            if sym:
                                symbols.append(sym)
                    return sorted(set(symbols))
    except Exception:
        pass

    # Fallback: one symbol per line
    for line in text.splitlines():
        sym = _normalize_symbol(line)
        if sym:
            symbols.append(sym)

    return sorted(set(symbols))


def filter_symbols_by_market(symbols: list[str], scope: list[str] | None) -> list[str]:
    if not symbols:
        return symbols
    if not scope:
        return symbols

    scope_set = {s.lower() for s in scope}
    out: list[str] = []
    for sym in symbols:
        s = _normalize_symbol(sym)
        if not s:
            continue
        if s.startswith(("6", "5", "9")):
            market = "sh"
        elif s.startswith(("0", "3")):
            market = "sz"
        elif s.startswith(("8", "4")):
            market = "bj"
        else:
            market = ""

        if market in scope_set:
            out.append(s)

    return sorted(set(out))


def filter_symbols_by_board(
    symbols: list[str],
    *,
    exclude_star: bool = False,
    exclude_chi_next: bool = False,
    mainboard_only: bool = False,
) -> list[str]:
    if not symbols:
        return symbols

    out: list[str] = []
    for sym in symbols:
        s = _normalize_symbol(sym)
        if not s:
            continue
        if mainboard_only:
            if s.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
                out.append(s)
            continue
        if exclude_star and (s.startswith("688") or s.startswith("689")):
            continue
        if exclude_chi_next and s.startswith("30"):
            continue
        out.append(s)

    return sorted(set(out))
