from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from sqdata.universe import filter_symbols_by_market, filter_symbols_by_board


def load_config(path: str) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from e

    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def load_universe_info(path: str) -> tuple[list[str], dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return [], {}
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        # fallback: try code column
        if "code" in df.columns:
            df = df.rename(columns={"code": "symbol"})
        else:
            return [], {}
    if "name" not in df.columns:
        df["name"] = ""
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    symbols = df["symbol"].dropna().astype(str).tolist()
    name_map = {str(r["symbol"]).zfill(6): str(r.get("name", "")) for _, r in df.iterrows()}
    return symbols, name_map


def _read_hist(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
    }
    df = df.rename(columns=col_map)
    if "date" not in df.columns:
        return pd.DataFrame()
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def build_trade_pool(config_path: str = "config/default.yaml") -> int:
    cfg = load_config(config_path)
    universe_file = str(cfg.get("universe_file", "./data/universe.csv"))
    universe_cfg = cfg.get("universe", {}) or {}

    market_scope = cfg.get("market_scope", ["sh", "sz"])
    exclude_star = bool(cfg.get("exclude_star", True))
    exclude_chi_next = bool(cfg.get("exclude_chi_next", False))
    mainboard_only = bool(cfg.get("mainboard_only", True))

    min_price = float(universe_cfg.get("min_price", 5.0))
    max_price = float(universe_cfg.get("max_price", 0.0))
    min_avg_amount_20 = float(universe_cfg.get("min_avg_amount_20", 1e8))
    exclude_st = bool(universe_cfg.get("exclude_st", True))

    manual_hist_dir = Path(str(cfg.get("manual_hist_dir", "./data/manual_hist")))
    if not manual_hist_dir.exists():
        raise RuntimeError(f"manual_hist_dir not found: {manual_hist_dir}")

    symbols, name_map = load_universe_info(universe_file)
    symbols = filter_symbols_by_market(symbols, market_scope)
    symbols = filter_symbols_by_board(
        symbols,
        exclude_star=exclude_star,
        exclude_chi_next=exclude_chi_next,
        mainboard_only=mainboard_only,
    )

    out_rows: list[dict[str, str]] = []
    missing = 0
    filtered = 0

    for sym in symbols:
        p = manual_hist_dir / f"{sym}.csv"
        if not p.exists():
            missing += 1
            continue
        df = _read_hist(p)
        if df.empty or "close" not in df.columns:
            missing += 1
            continue

        last_close = float(df["close"].iloc[-1]) if pd.notna(df["close"].iloc[-1]) else 0.0
        if last_close < min_price:
            filtered += 1
            continue
        if max_price > 0 and last_close > max_price:
            filtered += 1
            continue

        amt = df.get("amount")
        if amt is None or amt.dropna().empty or float(amt.fillna(0).abs().sum()) <= 0.0:
            # fallback: volume * close
            if "volume" in df.columns:
                # Tencent volume is in "hands" for A-shares; convert to amount by *100 shares.
                amt = df["volume"] * df["close"] * 100.0
            else:
                amt = pd.Series([0.0] * len(df))

        avg_amount_20 = float(amt.tail(20).mean()) if len(amt) >= 1 else 0.0
        if min_avg_amount_20 > 0 and avg_amount_20 < min_avg_amount_20:
            filtered += 1
            continue

        name = str(name_map.get(sym, ""))
        if exclude_st and name and re.search(r"ST|\\*ST|退", name):
            filtered += 1
            continue

        out_rows.append({"symbol": sym, "name": name})

    out_path = Path("./data/trade_pool.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"[ok] trade pool saved: {out_path} (n={len(out_rows)})")
    print(f"[info] missing_hist={missing} filtered_out={filtered}")
    return int(len(out_rows))


def main() -> None:
    build_trade_pool()


if __name__ == "__main__":
    main()
