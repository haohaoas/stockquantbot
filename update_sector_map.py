from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import pandas as pd


def _set_proxy(proxy: str) -> None:
    if not proxy:
        return
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


def _load_secret_token() -> str:
    secret_path = Path(__file__).resolve().parent / "config" / "secret.json"
    if not secret_path.exists():
        return ""
    try:
        data = json.loads(secret_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(data.get("tushare_token", "") or "").strip()


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fetch_by_akshare() -> pd.DataFrame:
    import akshare as ak  # type: ignore

    # Eastmoney industry board names
    boards = ak.stock_board_industry_name_em()
    if boards is None or boards.empty:
        raise RuntimeError("akshare industry board list empty")

    name_col = _pick_col(boards, ["板块名称", "板块", "行业名称", "name"])
    if not name_col:
        raise RuntimeError("unable to locate industry name column")

    mapping = []
    for _, row in boards.iterrows():
        board = str(row.get(name_col, "")).strip()
        if not board:
            continue
        try:
            cons = ak.stock_board_industry_cons_em(symbol=board)
        except Exception:
            continue
        if cons is None or cons.empty:
            continue
        code_col = _pick_col(cons, ["代码", "股票代码", "symbol", "code"])
        if not code_col:
            continue
        for code in cons[code_col].dropna().astype(str).tolist():
            sym = code.strip().zfill(6)
            if sym:
                mapping.append({"symbol": sym, "sector": board})
    if not mapping:
        raise RuntimeError("industry mapping empty")
    return pd.DataFrame(mapping).drop_duplicates(subset=["symbol"])


def fetch_by_tushare(token: str) -> pd.DataFrame:
    if not token:
        raise RuntimeError("tushare token is empty")
    try:
        import tushare as ts  # type: ignore
    except Exception as e:
        raise RuntimeError("tushare not installed. pip install tushare") from e

    pro = ts.pro_api(token)
    df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry,name")
    if df is None or df.empty:
        raise RuntimeError("tushare stock_basic empty")

    mapping = []
    for _, row in df.iterrows():
        ts_code = str(row.get("ts_code", "")).strip()
        industry = str(row.get("industry", "")).strip()
        if not ts_code or not industry:
            continue
        symbol = ts_code.split(".")[0].zfill(6)
        mapping.append({"symbol": symbol, "sector": industry})
    if not mapping:
        raise RuntimeError("tushare mapping empty")
    return pd.DataFrame(mapping).drop_duplicates(subset=["symbol"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto fetch industry mapping to data/sector_map.csv")
    parser.add_argument("--provider", choices=["auto", "akshare", "tushare"], default="auto")
    parser.add_argument("--proxy", default="")
    parser.add_argument("--token", default="")
    parser.add_argument("--out", default="data/sector_map.csv")
    args = parser.parse_args()

    if args.proxy:
        _set_proxy(args.proxy)

    df = pd.DataFrame()
    errors: list[str] = []

    if args.provider in ("auto", "akshare"):
        try:
            df = fetch_by_akshare()
        except Exception as e:
            errors.append(f"akshare failed: {e}")

    if df.empty and args.provider in ("auto", "tushare"):
        token = args.token or os.environ.get("TUSHARE_TOKEN", "") or _load_secret_token()
        if token:
            try:
                df = fetch_by_tushare(token)
            except Exception as e:
                errors.append(f"tushare failed: {e}")
        else:
            errors.append("tushare failed: missing token")

    if df.empty:
        msg = " | ".join(errors) if errors else "no provider available"
        raise RuntimeError(f"sector map fetch failed: {msg}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[ok] sector map saved: {out} (n={len(df)})")


if __name__ == "__main__":
    main()
