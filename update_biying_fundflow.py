from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yaml


_REQ_LOCK = threading.Lock()


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _load_trade_days(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    if "trade_date" not in df.columns:
        return []
    vals = (
        df["trade_date"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.slice(0, 10)
        .tolist()
    )
    return [v for v in vals if v]


def _resolve_end_date(trade_days: list[str], end_date: str | None) -> str:
    if end_date:
        return str(end_date)
    today = dt.datetime.now().date()
    if not trade_days:
        return today.isoformat()
    allowed = [d for d in trade_days if d <= today.isoformat()]
    if not allowed:
        return today.isoformat()
    now = dt.datetime.now()
    # 必盈资金流向文档说明：每日 21:30 更新。当天 21:30 前，取上一交易日更稳。
    if allowed[-1] == today.isoformat() and (now.hour < 21 or (now.hour == 21 and now.minute < 30)):
        if len(allowed) >= 2:
            return allowed[-2]
    return allowed[-1]


def _read_universe_symbols(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise RuntimeError("universe file missing symbol column")
    out = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .dropna()
        .astype(str)
        .str.zfill(6)
        .tolist()
    )
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _parse_symbols_arg(raw: str) -> list[str]:
    vals = []
    for item in str(raw or "").replace("\n", ",").split(","):
        item = item.strip()
        if not item:
            continue
        item = "".join(ch for ch in item if ch.isdigit()).zfill(6)
        if item and item.isdigit():
            vals.append(item)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in vals:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _header_columns(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
    except Exception:
        return []
    return [x.strip().strip('"').strip("'") for x in first.split(",") if x.strip()]


def _read_last_date(path: Path) -> str:
    if not path.exists() or path.stat().st_size <= 0:
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            pos = max(0, size - 4096)
            f.seek(pos)
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        for line in reversed(lines):
            head = line.split(",", 1)[0].strip().strip('"').strip("'")
            if len(head) >= 10:
                cand = head[:10]
                try:
                    dt.date.fromisoformat(cand)
                    return cand
                except Exception:
                    pass
    except Exception:
        return ""
    return ""


def _normalize_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty or "t" not in df.columns:
        return pd.DataFrame()

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([0.0] * len(df), index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    out = pd.DataFrame()
    out["date"] = df["t"].astype(str).str.slice(0, 10)
    out["main_buy_amt"] = _num("zmbtdcjzl") + _num("zmbddcjzl") + _num("zmbzdcjzl") + _num("zmbxdcjzl")
    out["main_sell_amt"] = _num("zmstdcjzl") + _num("zmsddcjzl") + _num("zmszdcjzl") + _num("zmsxdcjzl")
    out["main_net_inflow"] = out["main_buy_amt"] - out["main_sell_amt"]
    out["large_main_net_inflow"] = (_num("zmbtdcjzl") + _num("zmbddcjzl")) - (_num("zmstdcjzl") + _num("zmsddcjzl"))
    out["big_order_direction"] = _num("dddx")
    out["price_driver"] = _num("zddy")
    out["big_order_diff"] = _num("ddcf")
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _fetch_one(
    symbol: str,
    *,
    licence: str,
    base_url: str,
    timeout: int,
    sleep_sec: float,
    retries: int,
    start_date: str,
    end_date: str,
    latest: int,
) -> pd.DataFrame:
    url = f"{base_url.rstrip('/')}/hsstock/history/transaction/{symbol}/{licence}"
    params = {}
    if start_date:
        params["st"] = start_date.replace("-", "")
    if end_date:
        params["et"] = end_date.replace("-", "")
    if latest > 0:
        params["lt"] = str(int(latest))
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with _REQ_LOCK:
                resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            rows = resp.json()
            if isinstance(rows, dict):
                for key in ("data", "result", "results", "list", "rows"):
                    if isinstance(rows.get(key), list):
                        rows = rows.get(key)
                        break
            if not isinstance(rows, list):
                return pd.DataFrame()
            df = _normalize_rows(rows)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            return df
        except Exception as e:
            last_err = e
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            if attempt < retries:
                time.sleep(min(3.0, 0.6 * (attempt + 1)))
    if last_err:
        raise last_err
    return pd.DataFrame()


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    out = pd.concat([old, new], ignore_index=True)
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def _write_failed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 必盈历史资金流向并落盘到 data/biying_fundflow")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--universe-file", default="data/universe.csv")
    parser.add_argument("--trade-calendar-file", default="data/trade_calendar.csv")
    parser.add_argument("--out-dir", default="data/biying_fundflow")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--latest", type=int, default=0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--only-stale", action="store_true")
    parser.add_argument("--require-columns", default="")
    parser.add_argument("--checkpoint-file", default="cache/update_biying_fundflow_checkpoint.json")
    parser.add_argument("--failed-log-file", default="cache/update_biying_fundflow_failed.jsonl")
    parser.add_argument("--reset-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    signals = cfg.get("signals") or {}
    licence = str(signals.get("realtime_biying_licence") or "").strip()
    base_url = str(signals.get("realtime_biying_base_url") or "http://api.biyingapi.com").strip()
    if not licence:
        raise RuntimeError("missing signals.realtime_biying_licence in config")

    trade_days = _load_trade_days(args.trade_calendar_file)
    end_date = _resolve_end_date(trade_days, args.end_date or None)
    if args.start_date:
        start_date = str(args.start_date)
    elif trade_days:
        # 先抓近 3 年，避免第一轮量过大；后面要更长可继续调。
        end_dt = dt.date.fromisoformat(end_date)
        start_date = (end_dt - dt.timedelta(days=365 * 3)).isoformat()
    else:
        start_date = ""
    print(f"[info] auto end-date adjusted to last closed trading day: {end_date}")

    require_columns = [x.strip() for x in str(args.require_columns or "").split(",") if x.strip()]

    if args.symbols:
        symbols = _parse_symbols_arg(args.symbols)
    else:
        symbols = _read_universe_symbols(args.universe_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_file)
    failed_log_path = Path(args.failed_log_file)

    done_map: dict[str, str] = {}
    if checkpoint_path.exists() and not args.reset_checkpoint:
        try:
            done_map = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not isinstance(done_map, dict):
                done_map = {}
        except Exception:
            done_map = {}

    if done_map:
        symbols = [s for s in symbols if done_map.get(s) not in {"ok", "skip", "empty"}]
        print(f"[info] resume checkpoint loaded: skip_done={len(done_map)} remaining={len(symbols)}")

    total = len(symbols)
    stats = {"ok": 0, "skip": 0, "empty": 0, "error": 0}
    lock = threading.Lock()

    def _save_checkpoint(symbol: str, status: str) -> None:
        with lock:
            done_map[symbol] = status
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(done_map, ensure_ascii=False), encoding="utf-8")

    def _task(symbol: str) -> tuple[str, str]:
        out_path = out_dir / f"{symbol}.csv"
        if args.only_stale and out_path.exists():
            cols = _header_columns(out_path)
            last_date = _read_last_date(out_path)
            if (not require_columns or all(c in cols for c in require_columns)) and last_date >= end_date:
                return symbol, "skip"

        try:
            old = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
        except Exception:
            old = pd.DataFrame()
        try:
            new = _fetch_one(
                symbol,
                licence=licence,
                base_url=base_url,
                timeout=args.timeout,
                sleep_sec=args.sleep,
                retries=args.retries,
                start_date=start_date,
                end_date=end_date,
                latest=args.latest,
            )
            if new is None or new.empty:
                return symbol, "empty"
            merged = _merge(old, new)
            merged.to_csv(out_path, index=False, encoding="utf-8")
            return symbol, "ok"
        except Exception as e:
            _write_failed(
                failed_log_path,
                {
                    "symbol": symbol,
                    "status": "error",
                    "error": repr(e),
                    "start_date": start_date,
                    "end_date": end_date,
                    "ts": time.time(),
                },
            )
            return symbol, "error"

    if args.workers <= 1:
        for i, sym in enumerate(symbols, start=1):
            symbol, status = _task(sym)
            stats[status] += 1
            _save_checkpoint(symbol, status)
            if i % 20 == 0 or i == total:
                print(
                    f"[info] {i}/{total} ok={stats['ok']} skip={stats['skip']} "
                    f"empty={stats['empty']} error={stats['error']}"
                )
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futures = {ex.submit(_task, sym): sym for sym in symbols}
            done = 0
            for fut in as_completed(futures):
                symbol, status = fut.result()
                done += 1
                stats[status] += 1
                _save_checkpoint(symbol, status)
                if done % 20 == 0 or done == total:
                    print(
                        f"[info] {done}/{total} ok={stats['ok']} skip={stats['skip']} "
                        f"empty={stats['empty']} error={stats['error']}"
                    )

    print(
        f"[ok] done: ok={stats['ok']} skip={stats['skip']} empty={stats['empty']} "
        f"error={stats['error']} range={start_date or 'all'}~{end_date}"
    )


if __name__ == "__main__":
    main()
