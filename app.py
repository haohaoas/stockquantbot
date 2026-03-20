from __future__ import annotations

import csv
import datetime as dt
import os
import json
import re
import socket
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from features.factors import build_factors
from ml.ref_model import load_ref_model, predict_ref_score, build_ref_features, build_minute_ref_features
from sqdata.akshare_fetcher import fetch_a_share_daily_panel, fetch_hist
from sqdata.calendar import resolve_trade_date
from sqdata.universe import load_universe_symbols, filter_symbols_by_market, filter_symbols_by_board
from strategy.scorer import score_and_rank
from sqdata.sector_map import apply_sector_map, load_sector_map
from sqdata.news_sentiment import get_market_news_sentiment
from strategy.universe import filter_universe
from strategy.decision import apply_short_term_decision, apply_env_overrides
from strategy.market_regime import market_regime_ok, detect_market_env, compute_volatility_ratio, evaluate_market_filter
from sqdata.intraday_v2 import apply_intraday_v2
from sqdata.fetcher import get_realtime


def load_config(path: str) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from e

    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def load_secret(path: str = "config/secret.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_deepseek_key() -> str:
    secret = load_secret("config/secret.json")
    key = str(secret.get("deepseek_api_key", "") or "")
    if key:
        return key
    return str(os.environ.get("DEEPSEEK_API_KEY") or "")


def _normalize_sector_name(val: str) -> str:
    return (
        str(val or "")
        .replace(" ", "")
        .replace("，", "")
        .replace(",", "")
        .replace("、", "")
        .replace("板块", "")
        .replace("概念股", "")
        .replace("概念", "")
        .replace("主题", "")
        .replace("产业", "")
        .strip()
    )


def _is_hot_sector(sector: str, hot_set: set[str]) -> bool:
    sec = _normalize_sector_name(sector)
    if not sec:
        return False
    if sec in hot_set:
        return True
    for hot in hot_set:
        if not hot:
            continue
        if sec in hot or hot in sec:
            return True
    return False


def _parse_sector_keywords(text: str) -> set[str]:
    raw = str(text or "").strip()
    if not raw:
        return set()
    parts = re.split(r"[\s,，、;；|/]+", raw)
    out: set[str] = set()
    for p in parts:
        n = _normalize_sector_name(p)
        if n:
            out.add(n)
    return out


def _expand_sector_keywords(keywords: set[str], alias_map: dict | None = None) -> set[str]:
    out = set(keywords or set())
    if not out:
        return out

    # Common user aliases -> sector_map terms.
    builtins = {
        "CPO": ["光模块", "光通信", "通信设备", "通信"],
        "光模块": ["光通信", "通信设备", "CPO"],
        "光通信": ["光模块", "通信设备", "CPO"],
        "算力": ["服务器", "通信设备", "光模块", "计算机设备"],
        "AI应用": ["软件服务", "互联网", "IT设备", "计算机", "AIGC", "人工智能"],
        "AIGC": ["软件服务", "互联网", "IT设备", "AI应用", "人工智能"],
        "人工智能": ["软件服务", "互联网", "IT设备", "AI应用", "AIGC"],
    }

    def _match(a: str, b: str) -> bool:
        return a == b or a in b or b in a

    for k in list(out):
        for key, vals in builtins.items():
            key_n = _normalize_sector_name(key)
            if not key_n or not _match(k, key_n):
                continue
            for v in vals:
                v_n = _normalize_sector_name(v)
                if v_n:
                    out.add(v_n)

    if isinstance(alias_map, dict):
        for k in list(out):
            for key, vals in alias_map.items():
                key_n = _normalize_sector_name(key)
                if not key_n:
                    continue
                if isinstance(vals, str):
                    vals = [x.strip() for x in vals.split("、") if x.strip()]
                vals_n = [_normalize_sector_name(x) for x in (vals or []) if _normalize_sector_name(x)]
                if _match(k, key_n):
                    out.add(key_n)
                    out.update(vals_n)
    return out


def _is_target_sector(sector: str, keywords: set[str]) -> bool:
    if not keywords:
        return False
    sec = _normalize_sector_name(sector)
    if not sec:
        return False
    for kw in keywords:
        if not kw:
            continue
        if sec == kw or sec in kw or kw in sec:
            return True
    return False


def _expand_hot_sectors(hot_list: list[str], alias_map: dict) -> set[str]:
    base = {_normalize_sector_name(x) for x in hot_list if _normalize_sector_name(x)}
    expanded = set(base)
    if not isinstance(alias_map, dict):
        return expanded
    for key, vals in alias_map.items():
        key_norm = _normalize_sector_name(key)
        if not key_norm or key_norm not in base:
            continue
        if isinstance(vals, str):
            vals = [v.strip() for v in vals.split("、") if v.strip()]
        for v in vals or []:
            v_norm = _normalize_sector_name(v)
            if v_norm:
                expanded.add(v_norm)
    return expanded


_HOT_SECTOR_CACHE: dict[str, object] = {"ts": 0.0, "sectors": []}

_MODEL_SCORE_CACHE: dict[str, object] = {"key": "", "ts": 0.0, "scores": None}
_UNIVERSE_NAME_CACHE: dict[str, dict[str, str]] = {}
_MANUAL_UNIVERSE_CACHE: dict[str, list[str]] = {}
_FUND_FLOW_CACHE: dict[str, pd.DataFrame] = {}


def clear_runtime_caches() -> None:
    _MODEL_SCORE_CACHE["key"] = ""
    _MODEL_SCORE_CACHE["ts"] = 0.0
    _MODEL_SCORE_CACHE["scores"] = None
    _UNIVERSE_NAME_CACHE.clear()
    _MANUAL_UNIVERSE_CACHE.clear()
    _FUND_FLOW_CACHE.clear()
    _HOT_SECTOR_CACHE["ts"] = 0.0
    _HOT_SECTOR_CACHE["sectors"] = []


def _normalize_model_score_series(scores: pd.Series | None) -> pd.Series | None:
    if scores is None:
        return None
    ser = pd.Series(scores).copy()
    if ser.empty:
        return ser
    ser.index = ser.index.astype(str).str.zfill(6)
    ser = pd.to_numeric(ser, errors="coerce")
    ser = ser[ser.notna()]
    if ser.empty:
        return ser
    if ser.index.has_duplicates:
        ser = ser.groupby(level=0, sort=False).last()
    return ser


def _model_cache_get(key: str, ttl_sec: int) -> pd.Series | None:
    if not key:
        return None
    ts = float(_MODEL_SCORE_CACHE.get("ts", 0.0) or 0.0)
    if _MODEL_SCORE_CACHE.get("key") != key:
        return None
    if ttl_sec > 0 and (time.time() - ts) > ttl_sec:
        return None
    scores = _MODEL_SCORE_CACHE.get("scores")
    if isinstance(scores, pd.Series) and not scores.empty:
        return _normalize_model_score_series(scores)
    return None


def _model_cache_set(key: str, scores: pd.Series) -> None:
    if not key or scores is None or scores.empty:
        return
    scores = _normalize_model_score_series(scores)
    if scores is None or scores.empty:
        return
    _MODEL_SCORE_CACHE["key"] = key
    _MODEL_SCORE_CACHE["ts"] = time.time()
    _MODEL_SCORE_CACHE["scores"] = scores.copy()


def _load_universe_name_map(universe_file: str) -> dict[str, str]:
    cache_key = str(Path(universe_file).resolve())
    cached = _UNIVERSE_NAME_CACHE.get(cache_key)
    if cached is not None:
        return cached
    mapping: dict[str, str] = {}
    try:
        df = pd.read_csv(universe_file, dtype={"symbol": str}, usecols=["symbol", "name"])
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        mapping = dict(zip(df["symbol"], df["name"].astype(str)))
    except Exception:
        mapping = {}
    _UNIVERSE_NAME_CACHE[cache_key] = mapping
    return mapping


def _list_manual_hist_symbols(manual_hist_dir: str) -> list[str]:
    cache_key = str(Path(manual_hist_dir).resolve())
    cached = _MANUAL_UNIVERSE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    out: list[str] = []
    p = Path(manual_hist_dir)
    if p.exists():
        for fp in p.glob("*.csv"):
            s = fp.stem.strip()
            if s.isdigit() and len(s) == 6:
                out.append(s)
    out = sorted(set(out))
    _MANUAL_UNIVERSE_CACHE[cache_key] = out
    return out


def _load_manual_last_quote(manual_hist_dir: str, symbol: str) -> tuple[float, float]:
    p = Path(manual_hist_dir) / f"{str(symbol).zfill(6)}.csv"
    if not p.exists():
        return float("nan"), float("nan")
    try:
        df = pd.read_csv(p, usecols=["close"])
    except Exception:
        return float("nan"), float("nan")
    if df.empty or "close" not in df.columns:
        return float("nan"), float("nan")
    c = pd.to_numeric(df["close"], errors="coerce").dropna()
    if c.empty:
        return float("nan"), float("nan")
    close = float(c.iloc[-1])
    if len(c) >= 2 and c.iloc[-2] > 0:
        pct = float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)
    else:
        pct = float("nan")
    return close, pct


def _load_manual_hist_tail(manual_hist_dir: str, symbol: str, tail_rows: int = 80) -> pd.DataFrame:
    p = Path(manual_hist_dir) / f"{str(symbol).zfill(6)}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    keep = [c for c in ["date", "open", "close", "volume", "turnover_rate"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    out = df[keep].copy()
    for c in ("open", "close", "volume", "turnover_rate"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if tail_rows > 0 and len(out) > tail_rows:
        out = out.tail(tail_rows).copy()
    return out.reset_index(drop=True)


def _load_fundflow_tail(fundflow_dir: str, symbol: str, tail_rows: int = 5) -> pd.DataFrame:
    p = Path(fundflow_dir) / f"{str(symbol).zfill(6)}.csv"
    cache_key = str(p.resolve())
    cached = _FUND_FLOW_CACHE.get(cache_key)
    if cached is None:
        if not p.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        keep = [c for c in ["date", "main_buy_amt", "main_sell_amt", "main_net_inflow", "large_main_net_inflow"] if c in df.columns]
        if "date" not in keep:
            return pd.DataFrame()
        out = df[keep].copy()
        out["date"] = out["date"].astype(str).str.slice(0, 10)
        for c in ("main_buy_amt", "main_sell_amt", "main_net_inflow", "large_main_net_inflow"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        _FUND_FLOW_CACHE[cache_key] = out
        cached = out
    if cached is None or cached.empty:
        return pd.DataFrame()
    if tail_rows > 0 and len(cached) > tail_rows:
        return cached.tail(tail_rows).copy().reset_index(drop=True)
    return cached.copy().reset_index(drop=True)


def _fundflow_tag_from_hist(hist: pd.DataFrame, streak_days: int = 3) -> tuple[str, str, int]:
    if hist is None or hist.empty or "main_net_inflow" not in hist.columns:
        return "", "", 0
    series = pd.to_numeric(hist["main_net_inflow"], errors="coerce").dropna()
    if len(series) < streak_days:
        return "", "", 0
    tail = series.tail(streak_days)
    if (tail > 0).all():
        return f"主力{streak_days}连流入", "in", int(streak_days)
    if (tail < 0).all():
        return f"主力{streak_days}连流出", "out", -int(streak_days)
    return "", "", 0


def _attach_fundflow_tags(df: pd.DataFrame, *, fundflow_dir: str, streak_days: int = 3) -> pd.DataFrame:
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    out = df.copy()
    tags: list[str] = []
    tag_types: list[str] = []
    streaks: list[int] = []
    for sym in out["symbol"].astype(str).str.zfill(6):
        hist = _load_fundflow_tail(fundflow_dir, sym, tail_rows=max(streak_days, 5))
        tag, tag_type, streak = _fundflow_tag_from_hist(hist, streak_days=streak_days)
        tags.append(tag)
        tag_types.append(tag_type)
        streaks.append(streak)
    out["fundflow_tag"] = tags
    out["fundflow_tag_type"] = tag_types
    out["main_net_inflow_streak"] = streaks
    return out


def _compute_model_risk_features(
    symbols: list[str],
    *,
    manual_hist_dir: str,
    factors_rule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    fr = (
        factors_rule.set_index("symbol")
        if factors_rule is not None and not factors_rule.empty and "symbol" in factors_rule.columns
        else pd.DataFrame()
    )
    for sym in symbols:
        sym6 = str(sym).zfill(6)
        turnover_ratio = float("nan")
        high_turnover_risk = float("nan")
        volume_ratio_5 = float("nan")

        if not fr.empty and sym6 in fr.index:
            rr = fr.loc[sym6]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            volume_ratio_5 = float(pd.to_numeric(rr.get("volume_ratio_5", float("nan")), errors="coerce"))

        hist = _load_manual_hist_tail(manual_hist_dir, sym6, tail_rows=90)
        if not hist.empty:
            if "volume" in hist.columns:
                vol = pd.to_numeric(hist["volume"], errors="coerce")
                vr5 = vol / vol.rolling(5, min_periods=5).mean().shift(1)
                latest_vr5 = pd.to_numeric(vr5.iloc[-1], errors="coerce")
                if pd.notna(latest_vr5):
                    volume_ratio_5 = float(latest_vr5)
            if "turnover_rate" in hist.columns:
                turn = pd.to_numeric(hist["turnover_rate"], errors="coerce")
                tr20 = turn.rolling(20, min_periods=20).mean().shift(1)
                q90_60 = turn.rolling(60, min_periods=40).quantile(0.9).shift(1)
                latest_turn = pd.to_numeric(turn.iloc[-1], errors="coerce")
                latest_tr20 = pd.to_numeric(tr20.iloc[-1], errors="coerce")
                latest_q90 = pd.to_numeric(q90_60.iloc[-1], errors="coerce")
                if pd.notna(latest_turn) and pd.notna(latest_tr20) and float(latest_tr20) > 0:
                    turnover_ratio = float(latest_turn / latest_tr20)
                if pd.notna(latest_turn) and pd.notna(latest_q90):
                    high_turnover_risk = float(latest_turn > latest_q90)

        rows.append(
            {
                "symbol": sym6,
                "turnover_ratio": turnover_ratio,
                "high_turnover_risk": high_turnover_risk,
                "volume_ratio_5": volume_ratio_5,
            }
        )
    return pd.DataFrame(rows)


def _apply_model_risk_penalty(
    score: pd.Series | None,
    *,
    top_n: int,
    candidate_n: int,
    manual_hist_dir: str,
    factors_rule: pd.DataFrame | None,
    penalty_cfg: dict,
    notes: list[str],
) -> pd.Series | None:
    if score is None or score.empty:
        return score
    if not bool(penalty_cfg.get("enabled", False)):
        return score

    pool_n = max(int(candidate_n or top_n), int(top_n))
    pool = score.sort_values(ascending=False).head(pool_n)
    risk_df = _compute_model_risk_features(pool.index.astype(str).tolist(), manual_hist_dir=manual_hist_dir, factors_rule=factors_rule)
    if risk_df.empty:
        notes.append("模型风险惩罚: 风险因子为空，已跳过")
        return pool

    risk_df = risk_df.set_index("symbol")
    out = pd.DataFrame({"model_score": pool.astype(float)})
    out = out.join(risk_df, how="left")

    tr_rank = pd.to_numeric(out["turnover_ratio"], errors="coerce").rank(pct=True)
    vr_rank = pd.to_numeric(out["volume_ratio_5"], errors="coerce").rank(pct=True)
    hot = pd.to_numeric(out["high_turnover_risk"], errors="coerce")

    out["turnover_ratio_rank"] = tr_rank.fillna(0.0)
    out["volume_ratio_5_rank"] = vr_rank.fillna(0.0)
    out["high_turnover_risk"] = hot.fillna(0.0).clip(lower=0.0, upper=1.0)

    w_turn = float(penalty_cfg.get("turnover_ratio_weight", 0.45) or 0.45)
    w_hot = float(penalty_cfg.get("high_turnover_weight", 0.35) or 0.35)
    w_vol = float(penalty_cfg.get("volume_ratio_weight", 0.20) or 0.20)
    penalty_factor = float(penalty_cfg.get("penalty_factor", 0.08) or 0.08)
    hard_filter_high = bool(penalty_cfg.get("hard_filter_high_turnover", False))

    out["risk_score"] = (
        out["turnover_ratio_rank"] * w_turn
        + out["high_turnover_risk"] * w_hot
        + out["volume_ratio_5_rank"] * w_vol
    )
    out["final_score"] = out["model_score"] - penalty_factor * out["risk_score"]

    if hard_filter_high:
        before = len(out)
        out = out[out["high_turnover_risk"] < 0.5].copy()
        notes.append(f"模型风险硬过滤: high_turnover_risk 过滤 {before-len(out)}")

    notes.append(
        "模型风险惩罚: "
        f"pool={len(pool)}, penalty={penalty_factor:.3f}, "
        f"w(turnover_ratio/high_turnover/volume_ratio_5)="
        f"{w_turn:.2f}/{w_hot:.2f}/{w_vol:.2f}"
    )
    return out["final_score"].sort_values(ascending=False)


def _to_symbol_list(val) -> list[str]:
    if isinstance(val, str):
        return [s.strip().zfill(6) for s in val.split(",") if str(s).strip()]
    if isinstance(val, list):
        return [str(s).strip().zfill(6) for s in val if str(s).strip()]
    return []


def _build_model_top_rows(
    score: pd.Series,
    *,
    top_n: int,
    universe_file: str,
    manual_hist_dir: str,
    fundflow_dir: str = "./data/biying_fundflow",
    fundflow_streak_days: int = 3,
    sector_mapping: dict[str, str] | None = None,
    factors_rule: pd.DataFrame | None = None,
    candidate_n: int | None = None,
    risk_penalty_cfg: dict | None = None,
    notes: list[str] | None = None,
) -> pd.DataFrame:
    if score is None or score.empty:
        return pd.DataFrame()
    local_notes = notes if notes is not None else []
    effective_score = _apply_model_risk_penalty(
        score,
        top_n=top_n,
        candidate_n=int(candidate_n or max(top_n, 100)),
        manual_hist_dir=manual_hist_dir,
        factors_rule=factors_rule,
        penalty_cfg=dict(risk_penalty_cfg or {}),
        notes=local_notes,
    )
    top_scores = (effective_score if isinstance(effective_score, pd.Series) and not effective_score.empty else score).sort_values(ascending=False).head(top_n)
    name_map = _load_universe_name_map(universe_file)
    fr = (
        factors_rule.set_index("symbol")
        if factors_rule is not None and not factors_rule.empty and "symbol" in factors_rule.columns
        else pd.DataFrame()
    )
    model_rows = []
    for sym, sc in top_scores.items():
        sym = str(sym).zfill(6)
        name = name_map.get(sym, "")
        close = float("nan")
        pct = float("nan")
        pct_source = "hist"
        if not fr.empty and sym in fr.index:
            rr = fr.loc[sym]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            name = str(rr.get("name", name) or name)
            close = float(pd.to_numeric(rr.get("close", float("nan")), errors="coerce"))
            pct = float(pd.to_numeric(rr.get("pct_chg", float("nan")), errors="coerce"))
            pct_source = str(rr.get("pct_source", "spot") or "spot")
        if not (close == close and close > 0):
            close, pct_hist = _load_manual_last_quote(manual_hist_dir, sym)
            if not (pct == pct):
                pct = pct_hist
            pct_source = "hist"
        sec = sector_mapping.get(sym, "") if sector_mapping else ""
        model_rows.append(
                {
                    "symbol": sym,
                    "name": name,
                    "sector": sec,
                    "model_score": float(pd.to_numeric(score.get(sym, sc), errors="coerce")),
                    "final_score": float(sc),
                    "close": close,
                    "pct_chg": pct,
                    "pct_source": pct_source,
                }
        )
    out = pd.DataFrame(model_rows)
    return _attach_fundflow_tags(out, fundflow_dir=fundflow_dir, streak_days=int(fundflow_streak_days or 3))


def _compute_model_top_fast(
    *,
    trade_date: str,
    params: dict,
    signals_cfg: dict,
    notes: list[str],
    stats: dict,
    idx_hist: pd.DataFrame | None,
    universe_file: str,
    manual_hist_dir: str,
    model_ref_cfg: dict,
    model_ref_path: str,
    model_candidate_mode: str,
    model_candidate_max_symbols: int,
    model_sector_filter: str,
    news_cfg: dict,
    sector_cfg: dict,
    is_watchlist: bool,
    model_independent: bool,
    preselect_cached_symbols: list[str] | None = None,
    factors_rule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    t_model = time.time()
    m = load_ref_model(model_ref_path)
    if not m:
        notes.append("模型参考分不可用（模型文件缺失或加载失败）。")
        stats["time_model_ms"] = round((time.time() - t_model) * 1000, 2)
        return pd.DataFrame()

    model_task = str((m.get("meta") or {}).get("task", "")).strip().lower()
    is_minute_model = model_task == "minute_cls"
    model_risk_penalty_cfg = dict(model_ref_cfg.get("risk_penalty") or {})
    sector_map_file = str(sector_cfg.get("map_file", "./data/sector_map.csv"))
    sector_mapping = load_sector_map(sector_map_file)

    symbols_for_model: list[str] = []
    if is_watchlist:
        symbols_for_model = sorted(set(_to_symbol_list(signals_cfg.get("symbols"))))
        notes.append(f"模型榜单快路径(自选): {len(symbols_for_model)}")
    elif model_independent:
        symbols_for_model = _list_manual_hist_symbols(manual_hist_dir)
        before_candidate = len(symbols_for_model)
        market_scope = params.get("market_scope", ["sh", "sz"])
        symbols_for_model = filter_symbols_by_market(symbols_for_model, market_scope)
        symbols_for_model = filter_symbols_by_board(
            symbols_for_model,
            exclude_star=bool(params.get("exclude_star", False)),
            exclude_chi_next=bool(params.get("exclude_chi_next", False)),
            mainboard_only=bool(params.get("mainboard_only", False)),
        )
        st_removed = 0
        universe_cfg = params.get("universe", {}) or {}
        if bool(universe_cfg.get("exclude_st", True)):
            name_map = _load_universe_name_map(universe_file)
            filtered = []
            for s in symbols_for_model:
                nm = str(name_map.get(str(s).zfill(6), "") or "")
                if re.search(r"ST|\*ST|退", nm):
                    st_removed += 1
                    continue
                filtered.append(s)
            symbols_for_model = filtered
        if model_candidate_max_symbols > 0:
            symbols_for_model = symbols_for_model[:model_candidate_max_symbols]
        notes.append(
            f"模型榜单快路径(独立同池): {len(symbols_for_model)} "
            f"(raw={before_candidate}, st_removed={st_removed})"
        )
    elif preselect_cached_symbols is not None:
        symbols_for_model = sorted(set(str(s).zfill(6) for s in preselect_cached_symbols if str(s).strip()))
        notes.append(f"模型榜单快路径(预选缓存): {len(symbols_for_model)}")
    elif model_candidate_mode == "amount_topn":
        notes.append("模型榜单快路径不支持 amount_topn，已跳过。")
        stats["time_model_ms"] = round((time.time() - t_model) * 1000, 2)
        return pd.DataFrame()
    else:
        symbols_for_model = _list_manual_hist_symbols(manual_hist_dir)
        before_candidate = len(symbols_for_model)
        market_scope = params.get("market_scope", ["sh", "sz"])
        symbols_for_model = filter_symbols_by_market(symbols_for_model, market_scope)
        symbols_for_model = filter_symbols_by_board(
            symbols_for_model,
            exclude_star=bool(params.get("exclude_star", False)),
            exclude_chi_next=bool(params.get("exclude_chi_next", False)),
            mainboard_only=bool(params.get("mainboard_only", False)),
        )
        st_removed = 0
        universe_cfg = params.get("universe", {}) or {}
        if bool(universe_cfg.get("exclude_st", True)):
            name_map = _load_universe_name_map(universe_file)
            filtered = []
            for s in symbols_for_model:
                nm = str(name_map.get(str(s).zfill(6), "") or "")
                if re.search(r"ST|\*ST|退", nm):
                    st_removed += 1
                    continue
                filtered.append(s)
            symbols_for_model = filtered
        if model_candidate_max_symbols > 0:
            symbols_for_model = symbols_for_model[:model_candidate_max_symbols]
        notes.append(
            f"模型榜单快路径(回测同池): {len(symbols_for_model)} "
            f"(raw={before_candidate}, st_removed={st_removed})"
        )

    minute_model_max_symbols = int(model_ref_cfg.get("minute_max_symbols", 0) or 0)
    if is_minute_model and minute_model_max_symbols > 0 and len(symbols_for_model) > minute_model_max_symbols:
        symbols_for_model = symbols_for_model[:minute_model_max_symbols]
        notes.append(f"分钟模型候选限流: {len(symbols_for_model)} (max={minute_model_max_symbols})")

    if model_sector_filter and symbols_for_model:
        kws = _parse_sector_keywords(model_sector_filter)
        kws = _expand_sector_keywords(kws, news_cfg.get("hot_sector_aliases", {}) or {})
        if kws and sector_mapping:
            before = len(symbols_for_model)
            filtered_syms = []
            for sym in symbols_for_model:
                sym6 = str(sym).zfill(6)
                if _is_target_sector(sector_mapping.get(sym6, ""), kws):
                    filtered_syms.append(sym6)
            symbols_for_model = filtered_syms
            notes.append(f"模型板块过滤: {len(symbols_for_model)}/{before}")
        elif kws and not sector_mapping:
            notes.append("模型板块过滤: 板块映射缺失，已跳过。")

    if not symbols_for_model:
        notes.append("模型榜单候选为空。")
        stats["time_model_ms"] = round((time.time() - t_model) * 1000, 2)
        return pd.DataFrame()

    model_cache_enabled = bool(model_ref_cfg.get("cache_enabled", True))
    model_cache_ttl_sec = int(model_ref_cfg.get("cache_ttl_sec", 0))
    model_cache_dir = str(model_ref_cfg.get("cache_dir", "./cache/model_scores"))
    cache_key = f"{Path(model_ref_path).stem}_{trade_date}"
    cache_path = str(Path(model_cache_dir) / f"{cache_key}.csv")
    try:
        minute_model_cache_ttl_sec = int(signals_cfg.get("minute_model_cache_ttl_sec", 90))
    except Exception:
        minute_model_cache_ttl_sec = 90
    effective_model_cache_ttl = minute_model_cache_ttl_sec if is_minute_model else model_cache_ttl_sec

    cached_scores = None
    if model_cache_enabled and effective_model_cache_ttl > 0:
        cached_scores = _model_cache_get(cache_key, effective_model_cache_ttl)
        if cached_scores is None and not is_minute_model:
            cached_scores = _load_model_score_cache(cache_path, effective_model_cache_ttl)
            if cached_scores is not None:
                _model_cache_set(cache_key, cached_scores)
        if is_minute_model:
            notes.append(f"分钟模型缓存: {effective_model_cache_ttl}s")
    elif is_minute_model:
        notes.append("分钟模型缓存: 关闭")

    missing_syms = []
    if cached_scores is not None and symbols_for_model:
        missing_syms = [s for s in symbols_for_model if s not in cached_scores.index]
        if missing_syms:
            notes.append(f"模型分缓存命中: {len(symbols_for_model) - len(missing_syms)}/{len(symbols_for_model)}，补算 {len(missing_syms)}")
        else:
            notes.append(f"模型分缓存命中: {len(symbols_for_model)}")

    score = None
    if cached_scores is None or missing_syms:
        model_signals_cfg = dict(signals_cfg)
        target_syms = missing_syms if missing_syms else symbols_for_model
        fast_workers = int(model_ref_cfg.get("fast_max_workers", 24) or 24)
        current_workers = int(model_signals_cfg.get("max_workers", 8) or 8)
        if current_workers < fast_workers:
            model_signals_cfg["max_workers"] = fast_workers
        if m:
            if model_task == "minute_cls":
                if len(target_syms) > 120 and not bool(model_signals_cfg.get("minute_live_only_missing_today", False)):
                    model_signals_cfg["minute_live_only_missing_today"] = True
                    notes.append("分钟模型提速: 仅补齐缺失当日分钟线")
                live_cap = int(model_signals_cfg.get("minute_live_max_symbols", 0) or 0)
                if len(target_syms) > 120 and (live_cap <= 0 or live_cap > 200):
                    model_signals_cfg["minute_live_max_symbols"] = 200
                    notes.append("分钟模型提速: 实时拉取上限=200")
                model_feats = build_minute_ref_features(
                    target_syms,
                    m,
                    model_signals_cfg,
                    trade_date=trade_date,
                )
            else:
                model_feature_set = str((m.get("meta") or {}).get("feature_set") or "").strip().lower()
                if model_feature_set:
                    model_signals_cfg["feature_set"] = model_feature_set
                model_feats = build_ref_features(
                    target_syms,
                    trade_date,
                    model_signals_cfg,
                    market_df=idx_hist,
                    market_symbol=str(params.get("index_symbol", "000300")),
                )
            if not model_feats.empty:
                score = predict_ref_score(model_feats, m)
        if score is None:
            notes.append("模型参考分不可用（特征缺失或异常）。")

    if cached_scores is not None:
        score_series = cached_scores.copy()
        if isinstance(score, pd.Series) and not score.empty:
            score_series = score_series.combine_first(score)
            if model_cache_enabled and effective_model_cache_ttl > 0:
                _model_cache_set(cache_key, score_series)
                if not is_minute_model:
                    _save_model_score_cache(cache_path, score_series)
        score = score_series.reindex(symbols_for_model) if symbols_for_model else score_series
    elif isinstance(score, pd.Series):
        if model_cache_enabled and effective_model_cache_ttl > 0:
            _model_cache_set(cache_key, score)
            if not is_minute_model:
                _save_model_score_cache(cache_path, score)

    if isinstance(score, pd.Series) and not score.empty:
        score.index = score.index.astype(str).str.zfill(6)
        notes.append("模型参考分已生成（仅作参考，不参与评分）。")
        out = _build_model_top_rows(
            score,
            top_n=int(params.get("top_n", 20)),
            universe_file=universe_file,
            manual_hist_dir=manual_hist_dir,
            fundflow_dir=fundflow_dir,
            fundflow_streak_days=fundflow_streak_days,
            sector_mapping=sector_mapping,
            factors_rule=factors_rule,
            candidate_n=int(model_risk_penalty_cfg.get("candidate_pool_size", 100) or 100),
            risk_penalty_cfg=model_risk_penalty_cfg,
            notes=notes,
        )
        stats["time_model_ms"] = round((time.time() - t_model) * 1000, 2)
        return out

    stats["time_model_ms"] = round((time.time() - t_model) * 1000, 2)
    return pd.DataFrame()


def _load_model_score_cache(path: str, ttl_sec: int) -> pd.Series | None:
    p = Path(path)
    if not p.exists():
        return None
    if ttl_sec > 0:
        try:
            age = time.time() - p.stat().st_mtime
            if age > ttl_sec:
                return None
        except Exception:
            return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if "symbol" not in df.columns or "model_score" not in df.columns:
        return None
    ser = df.set_index("symbol")["model_score"]
    return _normalize_model_score_series(ser)


def _save_model_score_cache(path: str, scores: pd.Series) -> None:
    if scores is None or scores.empty:
        return
    scores = _normalize_model_score_series(scores)
    if scores is None or scores.empty:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = scores.reset_index()
    df.columns = ["symbol", "model_score"]
    df.to_csv(p, index=False, encoding="utf-8")


def _resolve_hot_sectors(cfg: dict, *, use_proxy: bool, proxy: str) -> list[str]:
    news_cfg = cfg.get("news", {}) or {}
    ttl_sec = int(news_cfg.get("summary_cache_sec", 0))
    now = time.time()
    cached = _HOT_SECTOR_CACHE.get("sectors") or []
    ts = float(_HOT_SECTOR_CACHE.get("ts", 0.0) or 0.0)
    if ttl_sec > 0 and cached and now - ts < ttl_sec:
        return list(cached)

    deepseek_cfg = cfg.get("deepseek", {}) or {}
    api_key = _load_deepseek_key() if deepseek_cfg.get("enabled", True) else ""
    base_url = str(deepseek_cfg.get("base_url", "") or "")
    model = str(deepseek_cfg.get("model", "deepseek-chat") or "deepseek-chat")

    summary = get_market_news_sentiment(
        limit=30,
        use_proxy=use_proxy,
        proxy=proxy,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    hot = summary.get("hot_sectors") or []
    if isinstance(hot, str):
        hot = [h.strip() for h in hot.split("、") if h.strip()]
    hot = [str(h).strip() for h in hot if str(h).strip()]
    _HOT_SECTOR_CACHE["ts"] = now
    _HOT_SECTOR_CACHE["sectors"] = hot
    return hot


def _normalize_symbol(sym: str) -> str:
    s = str(sym or "").strip()
    if not s:
        return ""
    if s.startswith(("sh", "sz")):
        s = s[2:]
    if "." in s:
        s = s.split(".", 1)[0]
    if len(s) == 6 and s.isdigit():
        return s
    return ""


def _apply_profile_filters(factors: pd.DataFrame, decision_cfg: dict, notes: list[str], is_watchlist: bool = False) -> pd.DataFrame:
    cfg = (decision_cfg or {}).get("profile_filter") or {}
    if not bool(cfg.get("enabled", False)):
        return factors
    if factors is None or factors.empty:
        return factors
    if is_watchlist:
        notes.append("自选模式跳过轮廓过滤。")
        return factors

    out = factors.copy()
    before = len(out)

    close_min = float(cfg.get("close_min", 4.0))
    close_max = float(cfg.get("close_max", 40.0))
    pct_min_raw = cfg.get("pct_min", None)
    pct_max_raw = cfg.get("pct_max", 6.0)
    pct_min = float(pct_min_raw) if pct_min_raw not in (None, "") else None
    pct_max = float(pct_max_raw) if pct_max_raw not in (None, "") else None
    require_upper_wick_gt_body = bool(cfg.get("upper_wick_gt_body", True))
    require_shrink_up = bool(cfg.get("require_shrink_up", True))
    shrink_vol_ratio_max = float(cfg.get("shrink_vol_ratio_max", 1.0))
    shrink_require_20d = bool(cfg.get("shrink_require_20d", False))
    cap_min = float(cfg.get("mkt_cap_min", 3e9))
    cap_max = float(cfg.get("mkt_cap_max", 1.5e10))
    turnover_min = float(cfg.get("turnover_pct_min", 3.0))
    turnover_max = float(cfg.get("turnover_pct_max", 15.0))
    fallback_if_empty = bool(cfg.get("fallback_if_empty", True))

    mask = pd.Series(True, index=out.index)
    if "close" in out.columns:
        c = pd.to_numeric(out["close"], errors="coerce")
        mask &= (c >= close_min) & (c <= close_max)
    if "pct_chg" in out.columns:
        p = pd.to_numeric(out["pct_chg"], errors="coerce")
        if pct_min is not None:
            mask &= p >= pct_min
        if pct_max is not None:
            mask &= p <= pct_max
    if require_upper_wick_gt_body and "upper_wick" in out.columns and "body" in out.columns:
        uw = pd.to_numeric(out["upper_wick"], errors="coerce")
        bd = pd.to_numeric(out["body"], errors="coerce").abs()
        mask &= uw > bd
    if require_shrink_up:
        p = pd.to_numeric(out.get("pct_chg", 0), errors="coerce")
        shrink = pd.to_numeric(out.get("volume_shrink_20d", float("nan")), errors="coerce")
        vr = pd.to_numeric(out.get("vol_ratio", float("nan")), errors="coerce")
        cond_shrink = (shrink >= 1.0)
        cond_ratio = (vr == vr) & (vr <= shrink_vol_ratio_max)
        cond = (p > 0) & ((cond_shrink & cond_ratio) if shrink_require_20d else (cond_shrink | cond_ratio))
        mask &= cond

    # Cap/turnover filters are applied only when data exists; otherwise skipped.
    cap_applied = False
    if "mkt_cap" in out.columns:
        cap = pd.to_numeric(out["mkt_cap"], errors="coerce")
        if cap.notna().any():
            cap_applied = True
            mask &= (cap >= cap_min) & (cap <= cap_max)
    if not cap_applied:
        notes.append("轮廓过滤: 市值数据不可用，已跳过市值区间过滤。")

    turnover_applied = False
    if "turnover_pct_est" in out.columns:
        to = pd.to_numeric(out["turnover_pct_est"], errors="coerce")
        if to.notna().any():
            turnover_applied = True
            mask &= (to >= turnover_min) & (to <= turnover_max)
    if not turnover_applied:
        notes.append("轮廓过滤: 换手率数据不可用，已跳过换手区间过滤。")

    filtered = out[mask].reset_index(drop=True)
    if filtered.empty and fallback_if_empty:
        notes.append("轮廓过滤命中为0，已回退到未过滤列表。")
        return factors

    notes.append(f"轮廓过滤: {len(filtered)}/{before}")
    return filtered


def _sanitize_factor_frame(factors: pd.DataFrame, notes: list[str], *, stage: str) -> pd.DataFrame:
    if factors is None or factors.empty:
        return factors

    out = factors.copy()

    dup_cols = out.columns[out.columns.duplicated()].tolist()
    if dup_cols:
        out = out.loc[:, ~out.columns.duplicated(keep="last")].copy()
        joined = ",".join(dict.fromkeys(str(x) for x in dup_cols))
        notes.append(f"{stage}: 已移除重复列 {joined}")

    if out.index.has_duplicates:
        dup_count = int(out.index.duplicated().sum())
        out = out.reset_index(drop=True)
        notes.append(f"{stage}: 已重置重复索引 {dup_count}")

    if "symbol" in out.columns:
        sym = out["symbol"].astype(str).str.zfill(6)
        dup_mask = sym.duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            out = out.loc[~dup_mask].copy()
            notes.append(f"{stage}: 已去重重复股票 {dup_count}")
        out["symbol"] = sym

    return out.reset_index(drop=True)


def _preselect_by_amount(spot: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if spot is None or spot.empty:
        return spot
    n = max(1, int(top_n or 1))
    if "amount" in spot.columns:
        tmp = spot.copy()
        tmp["_amt"] = pd.to_numeric(tmp["amount"], errors="coerce").fillna(-1)
        out = tmp.sort_values("_amt", ascending=False).head(n).drop(columns=["_amt"], errors="ignore")
        return out.reset_index(drop=True)
    return spot.head(n).reset_index(drop=True)


def _preselect_spot(spot: pd.DataFrame, preselect_n: int, signals_cfg: dict, notes: list[str]) -> pd.DataFrame:
    if spot is None or spot.empty:
        return spot

    n = max(1, int(preselect_n or 1))

    if not bool(signals_cfg.get("preselect_layered", False)):
        return _preselect_by_amount(spot, n)

    cap_col = str(signals_cfg.get("preselect_layer_col", "mkt_cap") or "mkt_cap")
    bins_raw = signals_cfg.get("preselect_layer_bins", [0, 3e10, 1e11, 1e18]) or []
    weights_raw = signals_cfg.get("preselect_layer_weights", [0.4, 0.4, 0.2]) or []

    try:
        bins = [float(x) for x in bins_raw]
        weights = [float(x) for x in weights_raw]
    except Exception:
        notes.append("分层预选参数非法，已回退成交额TopN。")
        return _preselect_by_amount(spot, n)

    if len(bins) < 2 or len(weights) != len(bins) - 1:
        notes.append("分层预选参数长度不匹配，已回退成交额TopN。")
        return _preselect_by_amount(spot, n)

    if cap_col not in spot.columns:
        notes.append(f"分层预选列缺失({cap_col})，已回退成交额TopN。")
        return _preselect_by_amount(spot, n)

    tmp = spot.copy()
    tmp["_cap"] = pd.to_numeric(tmp[cap_col], errors="coerce")
    tmp["_amt"] = pd.to_numeric(tmp.get("amount", float("nan")), errors="coerce").fillna(-1)
    if not tmp["_cap"].notna().any():
        notes.append(f"分层预选列无有效值({cap_col})，已回退成交额TopN。")
        return _preselect_by_amount(tmp.drop(columns=["_cap", "_amt"], errors="ignore"), n)

    picks = []
    used_idx: set[int] = set()
    bucket_hits: list[int] = []
    remain = n
    for i, w in enumerate(weights):
        if remain <= 0:
            bucket_hits.append(0)
            continue
        low, high = bins[i], bins[i + 1]
        k = int(round(n * w))
        if i == len(weights) - 1:
            k = max(k, remain)
        k = max(0, min(k, remain))
        if k == 0:
            bucket_hits.append(0)
            continue
        bucket = tmp[(tmp["_cap"] >= low) & (tmp["_cap"] < high) & (~tmp.index.isin(used_idx))]
        part = bucket.sort_values("_amt", ascending=False).head(k)
        picks.append(part)
        used_idx.update(part.index.tolist())
        got = int(len(part))
        bucket_hits.append(got)
        remain -= got

    selected = pd.concat(picks, axis=0) if picks else tmp.iloc[0:0]
    if len(selected) < n:
        fill = tmp[~tmp.index.isin(selected.index)].sort_values("_amt", ascending=False).head(n - len(selected))
        selected = pd.concat([selected, fill], axis=0)

    selected = selected.head(n).drop(columns=["_cap", "_amt"], errors="ignore").reset_index(drop=True)
    notes.append(f"分层预选: {len(selected)}/{len(spot)} (桶命中: {bucket_hits})")
    return selected


def _exclude_preselect_sectors(
    spot: pd.DataFrame,
    *,
    signals_cfg: dict,
    sector_cfg: dict,
    notes: list[str],
) -> pd.DataFrame:
    if spot is None or spot.empty:
        return spot

    enabled = bool(signals_cfg.get("preselect_exclude_bank", False))
    if not enabled:
        return spot

    raw = signals_cfg.get("preselect_exclude_sector_keywords", ["银行"])
    if isinstance(raw, str):
        keys = {k for k in _parse_sector_keywords(raw) if k}
    elif isinstance(raw, (list, tuple, set)):
        keys = {_normalize_sector_name(str(k)) for k in raw if _normalize_sector_name(str(k))}
    else:
        keys = {"银行"}
    if not keys:
        keys = {"银行"}

    out = spot.copy()
    rm = pd.Series(False, index=out.index)
    removed_by_sector = 0
    removed_by_name = 0

    if "sector" in out.columns:
        sec_hit = out["sector"].astype(str).apply(lambda s: _is_target_sector(s, keys))
        removed_by_sector += int((sec_hit & ~rm).sum())
        rm |= sec_hit

    map_file = str((sector_cfg or {}).get("map_file", "./data/sector_map.csv"))
    mapping = load_sector_map(map_file)
    if mapping and "symbol" in out.columns:
        mapped = out["symbol"].astype(str).str.zfill(6).map(mapping).fillna("")
        map_hit = mapped.apply(lambda s: _is_target_sector(s, keys))
        removed_by_sector += int((map_hit & ~rm).sum())
        rm |= map_hit

    # Fallback by name to catch bank stocks when sector map is missing/stale.
    if "name" in out.columns:
        name_norm = out["name"].astype(str).apply(_normalize_sector_name)
        name_hit = name_norm.apply(lambda s: any(k and (k in s) for k in keys))
        removed_by_name += int((name_hit & ~rm).sum())
        rm |= name_hit

    filtered = out[~rm].reset_index(drop=True)
    keys_text = "、".join(sorted(keys))
    notes.append(
        f"预筛板块排除({keys_text}): {len(filtered)}/{len(out)} "
        f"(sector={removed_by_sector}, name={removed_by_name})"
    )
    return filtered


def _exclude_preselect_symbols_from_cache(
    symbols: list[str],
    *,
    signals_cfg: dict,
    sector_cfg: dict,
    universe_file: str,
) -> tuple[list[str], int]:
    if not symbols:
        return [], 0
    if not bool(signals_cfg.get("preselect_exclude_bank", False)):
        cleaned = [str(s).zfill(6) for s in symbols if str(s).strip()]
        return cleaned, 0

    raw = signals_cfg.get("preselect_exclude_sector_keywords", ["银行"])
    if isinstance(raw, str):
        keys = {k for k in _parse_sector_keywords(raw) if k}
    elif isinstance(raw, (list, tuple, set)):
        keys = {_normalize_sector_name(str(k)) for k in raw if _normalize_sector_name(str(k))}
    else:
        keys = {"银行"}
    if not keys:
        keys = {"银行"}

    map_file = str((sector_cfg or {}).get("map_file", "./data/sector_map.csv"))
    mapping = load_sector_map(map_file)
    name_map = _load_universe_name_map(universe_file)

    kept: list[str] = []
    removed = 0
    seen: set[str] = set()
    for s in symbols:
        sym = str(s).strip().zfill(6)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        sec = str(mapping.get(sym, "") or "")
        if _is_target_sector(sec, keys):
            removed += 1
            continue
        nm = _normalize_sector_name(name_map.get(sym, ""))
        if nm and any(k and (k in nm) for k in keys):
            removed += 1
            continue
        kept.append(sym)
    return kept, removed


def _repair_spot_from_manual_hist(
    spot: pd.DataFrame,
    *,
    symbols: list[str],
    universe_file: str,
    manual_hist_dir: str,
) -> tuple[pd.DataFrame, int, int]:
    if spot is None or spot.empty or "symbol" not in spot.columns:
        return spot, 0, 0

    out = spot.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.zfill(6)
    filled_price = 0
    filled_name = 0

    # Fill blank names from universe map so model_top/列表能显示中文名。
    if "name" in out.columns:
        name_blank = out["name"].astype(str).str.strip().isin(["", "nan", "None"])
        if name_blank.any():
            try:
                udf = pd.read_csv(universe_file, dtype={"symbol": str}, usecols=["symbol", "name"])
                udf["symbol"] = udf["symbol"].astype(str).str.zfill(6)
                nmap = dict(zip(udf["symbol"], udf["name"].astype(str)))
                before = int(name_blank.sum())
                out.loc[name_blank, "name"] = out.loc[name_blank, "symbol"].map(nmap).fillna(out.loc[name_blank, "name"])
                after_blank = out["name"].astype(str).str.strip().isin(["", "nan", "None"]).sum()
                filled_name = max(0, before - int(after_blank))
            except Exception:
                pass

    close_series = pd.to_numeric(out.get("close", float("nan")), errors="coerce")
    need_symbols = set(symbols) if symbols else set(out["symbol"].tolist())
    need_mask = out["symbol"].isin(need_symbols) & ((close_series.isna()) | (close_series <= 0))
    if not need_mask.any():
        return out, filled_price, filled_name

    hist_dir = Path(manual_hist_dir)
    if not hist_dir.exists():
        return out, filled_price, filled_name

    for idx in out.index[need_mask]:
        sym = str(out.at[idx, "symbol"]).zfill(6)
        p = hist_dir / f"{sym}.csv"
        if not p.exists():
            continue
        try:
            h = pd.read_csv(p)
            if h.empty or "close" not in h.columns:
                continue
            c = pd.to_numeric(h["close"], errors="coerce").dropna()
            if c.empty:
                continue
            last_close = float(c.iloc[-1])
            if not (last_close > 0):
                continue
            out.at[idx, "close"] = last_close
            if "pct_chg" in out.columns:
                if len(c) >= 2 and c.iloc[-2] > 0:
                    out.at[idx, "pct_chg"] = float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)
                elif pd.isna(out.at[idx, "pct_chg"]):
                    out.at[idx, "pct_chg"] = 0.0
            if "amount" in out.columns and "amount" in h.columns:
                a = pd.to_numeric(h["amount"], errors="coerce").dropna()
                if not a.empty and (pd.isna(out.at[idx, "amount"]) or float(out.at[idx, "amount"]) <= 0):
                    out.at[idx, "amount"] = float(a.iloc[-1])
            filled_price += 1
        except Exception:
            continue

    return out, filled_price, filled_name


def load_watchlist(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    symbols: list[str] = []
    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fields = [h.lower() for h in reader.fieldnames]
                key = None
                for k in ("symbol", "code", "ts_code", "ticker"):
                    if k in fields:
                        key = reader.fieldnames[fields.index(k)]
                        break
                if key:
                    for row in reader:
                        sym = _normalize_symbol(row.get(key, ""))
                        if sym:
                            symbols.append(sym)
                    return sorted(set(symbols))
    except Exception:
        pass

    for line in text.splitlines():
        sym = _normalize_symbol(line)
        if sym:
            symbols.append(sym)

    return sorted(set(symbols))


def save_watchlist(path: str, symbols: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol"])
        for s in sorted(set(symbols)):
            writer.writerow([s])


def _load_trade_calendar(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return set()

    dates: set[str] = set()
    lines = text.splitlines()
    header = lines[0].lower() if lines else ""

    if "date" in header:
        for line in lines[1:]:
            parts = [x.strip() for x in line.split(",")]
            if parts and len(parts[0]) == 10:
                dates.add(parts[0])
    else:
        for line in lines:
            s = line.strip()
            if len(s) == 10:
                dates.add(s)
    return dates


def _apply_sector_heat(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "sector" not in df.columns:
        return df
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return df

    out = df.copy()
    top_k = int(cfg.get("top_k", 5))
    weights = cfg.get("weights", {}) or {}
    w_price = float(weights.get("price", 0.5))
    w_volume = float(weights.get("volume", 0.3))
    w_amount = float(weights.get("amount", 0.2))
    hot_bonus = float(cfg.get("hot_bonus", 0.0))

    price_rank = pd.to_numeric(out.get("pct_chg", 0), errors="coerce").rank(pct=True)
    vol_rank = pd.to_numeric(out.get("vol_ratio", 0), errors="coerce").rank(pct=True)
    amt_rank = pd.to_numeric(out.get("amount", 0), errors="coerce").rank(pct=True)
    row_heat = (price_rank.fillna(0) * w_price + vol_rank.fillna(0) * w_volume + amt_rank.fillna(0) * w_amount)
    out["_row_heat"] = row_heat.fillna(0.0)

    def _sector_score(s: pd.Series) -> float:
        if s.empty:
            return 0.0
        if top_k > 0:
            return float(s.nlargest(min(top_k, len(s))).mean())
        return float(s.mean())

    sector_heat = out.groupby("sector")["_row_heat"].apply(_sector_score)
    out["sector_heat"] = out["sector"].map(sector_heat).fillna(0.0) * 100.0
    if hot_bonus > 0:
        hot_mask = out["sector_heat"] > 0
        out.loc[hot_mask, "sector_heat"] = (out.loc[hot_mask, "sector_heat"] + hot_bonus).clip(0, 100)
    out.drop(columns=["_row_heat"], inplace=True, errors="ignore")
    return out


def _load_preselect_cache(
    path: str, trade_date: str, ttl_sec: int, hot_filter: bool | None = None
) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    if str(data.get("trade_date", "")) != str(trade_date):
        return []
    if hot_filter is not None:
        cached_hot = data.get("hot_filter", None)
        if cached_hot is None or bool(cached_hot) != bool(hot_filter):
            return []
    ts = float(data.get("ts", 0) or 0)
    if ttl_sec > 0 and (time.time() - ts) > ttl_sec:
        return []
    symbols = data.get("symbols") or []
    if not isinstance(symbols, list):
        return []
    cleaned = []
    for s in symbols:
        sym = _normalize_symbol(str(s))
        if sym:
            cleaned.append(sym)
    return sorted(set(cleaned))


def _save_preselect_cache(
    path: str, trade_date: str, symbols: list[str], hot_filter: bool | None = None
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trade_date": str(trade_date),
        "ts": time.time(),
        "symbols": [str(s).zfill(6) for s in symbols if str(s).strip()],
    }
    if hot_filter is not None:
        payload["hot_filter"] = bool(hot_filter)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _socks_available() -> bool:
    try:
        import socks  # noqa: F401
        return True
    except Exception:
        return False


def _test_proxy(proxy: str, timeout: float = 1.5) -> bool:
    import requests

    try:
        r = requests.get(
            "https://qt.gtimg.cn/q=sh600000",
            proxies={"http": proxy, "https": proxy},
            timeout=timeout,
        )
        if r.status_code:
            return True
    except Exception:
        pass

    try:
        r = requests.get(
            "https://82.push2.eastmoney.com/api/qt/clist/get",
            params={
                "pn": "1",
                "pz": "1",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f12",
                "fs": "m:0 t:6",
                "fields": "f12,f14,f2,f3,f5,f6,f20,f21,f13",
            },
            proxies={"http": proxy, "https": proxy},
            timeout=timeout,
        )
        return bool(r.status_code)
    except Exception:
        return False


def _detect_proxy(timeout: float = 1.5) -> str:
    candidates = [7890, 7891, 7892, 7893, 7897, 7898, 7899, 33331, 8888, 8889, 1080, 1081]
    socks_ok = _socks_available()

    for port in candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                s.close()
                continue
            s.close()
        except Exception:
            continue

        http_proxy = f"http://127.0.0.1:{port}"
        if _test_proxy(http_proxy, timeout=timeout):
            return http_proxy

        if socks_ok:
            socks_proxy = f"socks5h://127.0.0.1:{port}"
            if _test_proxy(socks_proxy, timeout=timeout):
                return socks_proxy

    return ""


def _get_env_proxy() -> str:
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        val = os.environ.get(key, "")
        if val:
            return val.strip()
    return ""


def _ensure_no_proxy_defaults() -> None:
    defaults = [
        "localhost",
        "127.0.0.1",
        "eastmoney.com",
        "push2.eastmoney.com",
        "82.push2.eastmoney.com",
        "82.push2delay.eastmoney.com",
        "gtimg.cn",
        "qt.gtimg.cn",
        "ifzq.gtimg.cn",
        "web.ifzq.gtimg.cn",
        "sinaimg.cn",
        "api.tushare.pro",
        "baostock.com",
        "api.finance.ifeng.com",
        "quotes.money.163.com",
    ]
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    current_list = [x.strip() for x in current.split(",") if x.strip()]
    merged = current_list[:]
    for host in defaults:
        if host not in merged:
            merged.append(host)
    value = ",".join(merged)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _set_env_proxy(proxy: str) -> None:
    if not proxy:
        return
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    _ensure_no_proxy_defaults()


def _clear_env_proxy() -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if k in os.environ:
            os.environ.pop(k, None)


def _alt_proxy(proxy: str) -> str:
    if proxy.startswith("http://") or proxy.startswith("https://"):
        return "socks5h://" + proxy.split("://", 1)[1]
    if proxy.startswith("socks5h://") or proxy.startswith("socks5://"):
        return "http://" + proxy.split("://", 1)[1]
    return ""


def _proxy_candidates(use_proxy: bool, proxy: str) -> list[tuple[bool, str]]:
    cands: list[tuple[bool, str]] = []
    if proxy:
        cands.append((True, proxy))
        alt = _alt_proxy(proxy)
        if alt:
            cands.append((True, alt))
        cands.append((False, ""))
    elif use_proxy:
        cands.append((True, ""))
        cands.append((False, ""))
    else:
        cands.append((False, ""))
    return cands


def _is_market_open_cn(now: dt.datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    # Include opening call auction (09:15-09:30) so auto refresh starts before continuous trading.
    return (555 <= t <= 690) or (780 <= t <= 900)


def _is_trading_day_cn(date: dt.date, calendar_dates: set[str] | None) -> bool:
    if calendar_dates:
        return date.isoformat() in calendar_dates
    return date.weekday() < 5


def compute_market(params: dict) -> dict:
    notes: list[str] = []
    stats: dict = {}
    model_top = pd.DataFrame()
    t_start = time.time()
    pause_signals = False
    model_top_only = bool(params.get("model_top_only", False))
    model_independent = bool(params.get("model_independent", False))

    def _finalize_timings() -> None:
        stats["time_total_ms"] = round((time.time() - t_start) * 1000, 2)

    trade_date = resolve_trade_date(str(params.get("trade_date", "")).strip())
    universe_cfg = params.get("universe", {}) or {}
    signals_cfg = params.get("signals", {}) or {}
    feature_set = str((params.get("features") or {}).get("set", "legacy") or "legacy")
    if "feature_set" not in signals_cfg and feature_set:
        signals_cfg = dict(signals_cfg)
        signals_cfg["feature_set"] = feature_set
    weights = params.get("weights", {}) or {}
    sector_cfg = params.get("sector_boost", {}) or {}
    decision_cfg = params.get("decision", {}) or {}
    market_filter_v2_cfg = decision_cfg.get("market_filter_v2") or {}
    market_filter_v2_enabled = bool(market_filter_v2_cfg.get("enabled", False))
    news_cfg = params.get("news", {}) or {}
    deepseek_cfg = params.get("deepseek", {}) or {}
    model_sector_filter = str(params.get("model_sector", "") or "").strip()
    universe_file = str(params.get("universe_file", "./data/universe.csv"))
    use_universe_file = bool(params.get("use_universe_file", True))
    allow_eastmoney_fallback = bool(params.get("allow_eastmoney_fallback", False))
    exclude_limit_up = bool(params.get("exclude_limit_up", True))
    limit_up_pct = float(params.get("limit_up_pct", 9.8))
    exclude_non_realtime_pct = bool(params.get("exclude_non_realtime_pct", True))
    is_watchlist = bool(params.get("is_watchlist", False))

    hot_filter_enabled = bool(sector_cfg.get("hot_filter", False))
    hot_filter_min = int(sector_cfg.get("hot_filter_min", 1))
    hot_filter_strict = bool(sector_cfg.get("hot_filter_strict", True))

    preselect_n = int(signals_cfg.get("preselect_n", 300))
    dynamic_preselect_expand = bool(signals_cfg.get("dynamic_preselect_expand", False))
    dynamic_preselect_buy_min = max(0, int(signals_cfg.get("dynamic_preselect_buy_min", 1) or 0))
    dynamic_preselect_steps_raw = signals_cfg.get("dynamic_preselect_steps", [preselect_n, 600, 1000]) or [preselect_n]
    dynamic_preselect_steps: list[int] = []
    for step in dynamic_preselect_steps_raw:
        try:
            step_n = int(step or 0)
        except Exception:
            continue
        if step_n <= 0:
            continue
        dynamic_preselect_steps.append(step_n)
    if preselect_n not in dynamic_preselect_steps:
        dynamic_preselect_steps.insert(0, preselect_n)
    dynamic_preselect_steps = sorted({max(preselect_n, x) for x in dynamic_preselect_steps})
    preselect_cache_enabled = bool(signals_cfg.get("preselect_cache", False))
    preselect_cache_ttl_sec = int(signals_cfg.get("preselect_cache_ttl_sec", 0))
    preselect_cache_file = str(signals_cfg.get("preselect_cache_file", "./cache/preselect_symbols.json"))
    model_ref_cfg = signals_cfg.get("model_ref") or {}
    model_ref_enabled = bool(model_ref_cfg.get("enabled", False))
    model_ref_path = str(model_ref_cfg.get("path", "") or "").strip()
    model_ref_filter_enabled = bool(model_ref_cfg.get("filter_enabled", False))
    model_ref_min_score = float(model_ref_cfg.get("min_score", 0.0))
    model_ref_show_min = float(model_ref_cfg.get("show_min", 0.7))
    model_cache_enabled = bool(model_ref_cfg.get("cache_enabled", True))
    model_cache_ttl_sec = int(model_ref_cfg.get("cache_ttl_sec", 0))
    model_cache_dir = str(model_ref_cfg.get("cache_dir", "./cache/model_scores"))
    model_candidate_mode = str(model_ref_cfg.get("candidate_mode", "backtest_like") or "backtest_like").lower()
    model_candidate_max_symbols = int(model_ref_cfg.get("candidate_max_symbols", 0) or 0)
    model_risk_penalty_cfg = dict(model_ref_cfg.get("risk_penalty") or {})
    fundflow_cfg = signals_cfg.get("fundflow") or {}
    fundflow_dir = str(fundflow_cfg.get("dir", "./data/biying_fundflow") or "./data/biying_fundflow")
    fundflow_streak_days = max(2, int(fundflow_cfg.get("streak_days", 3) or 3))
    regime_filter_enabled = bool(params.get("regime_filter", True))
    index_symbol = str(params.get("index_symbol", "000300"))
    index_ma_window = int(params.get("index_ma_window", 200))
    top_n = int(params.get("top_n", 50))
    use_proxy = bool(signals_cfg.get("use_proxy", False))
    proxy = str(signals_cfg.get("proxy", "") or "")
    chosen_use_proxy = use_proxy
    chosen_proxy = proxy
    env_proxy = _get_env_proxy()
    idx_hist: pd.DataFrame | None = None
    manual_hist_dir = str(signals_cfg.get("manual_hist_dir", "./data/manual_hist"))
    realtime_fix_limit = int(signals_cfg.get("realtime_fix_limit", 200))
    realtime_cap = max(top_n * 2, 60)
    if realtime_fix_limit > realtime_cap:
        signals_cfg = dict(signals_cfg)
        signals_cfg["realtime_fix_limit"] = realtime_cap
        notes.append(f"实时补价上限: {realtime_cap}")

    if proxy:
        use_proxy = True
    elif env_proxy:
        if _test_proxy(env_proxy, timeout=1.2):
            use_proxy = True
            proxy = ""
            _ensure_no_proxy_defaults()
            notes.append("检测到环境代理，已启用 NO_PROXY 白名单。")
        else:
            use_proxy = False
            proxy = ""
            _clear_env_proxy()
            notes.append(f"检测到环境代理但不可用，已忽略: {env_proxy}")
    elif use_proxy and not proxy:
        detected = _detect_proxy()
        if detected:
            _set_env_proxy(detected)
            proxy = ""
            notes.append(f"自动检测到本地代理(ENV): {detected}")
    elif (not use_proxy) and not proxy:
        detected = _detect_proxy()
        if detected:
            use_proxy = True
            _set_env_proxy(detected)
            proxy = ""
            notes.append(f"自动检测到本地代理(ENV): {detected}")

    if regime_filter_enabled and not market_filter_v2_enabled and not is_watchlist and not model_top_only:
        try:
            idx_hist = fetch_hist(
                index_symbol,
                end_date=trade_date,
                hist_days=max(index_ma_window + 50, 260),
                use_proxy=use_proxy,
                proxy=proxy,
                allow_akshare_fallback=allow_eastmoney_fallback,
            )
        except Exception as e:
            idx_hist = pd.DataFrame()
            notes.append(f"指数行情获取失败: {e}")
        if idx_hist is None or idx_hist.empty:
            notes.append("指数数据为空，已跳过市场状态过滤。")
        elif not market_regime_ok(idx_hist, ma_window=index_ma_window):
            notes.append(f"市场状态偏弱（{index_symbol} < MA{index_ma_window}），已暂停信号。")
            pause_signals = True

    if not model_top_only:
        market_env = str(decision_cfg.get("market_env", "") or "").strip()
        if not market_env or market_env.lower() == "auto":
            if idx_hist is None:
                try:
                    idx_hist = fetch_hist(
                        index_symbol,
                        end_date=trade_date,
                        hist_days=max(index_ma_window + 50, 260),
                        use_proxy=use_proxy,
                        proxy=proxy,
                        allow_akshare_fallback=allow_eastmoney_fallback,
                    )
                except Exception as e:
                    notes.append(f"市场环境识别失败: {e}")
                    idx_hist = pd.DataFrame()
            market_env = detect_market_env(idx_hist) if idx_hist is not None else "震荡市场"
        if market_env:
            decision_cfg = apply_env_overrides(decision_cfg, market_env)
            notes.append(f"市场环境: {market_env}")

        dyn_cfg = decision_cfg.get("dynamic_volatility") or {}
        if isinstance(dyn_cfg, dict) and dyn_cfg.get("enabled"):
            window = int(dyn_cfg.get("window", 20))
            history = int(dyn_cfg.get("history", 20))
            vol_ratio_info = compute_volatility_ratio(idx_hist, window=window, history=history) if idx_hist is not None else None
            if vol_ratio_info:
                current_vol, avg_vol, vol_ratio = vol_ratio_info
                high_ratio = float(dyn_cfg.get("high_ratio", 1.2))
                low_ratio = float(dyn_cfg.get("low_ratio", 0.8))
                base_vol_min = float(decision_cfg.get("breakout_min_volume_multiple", 1.2))
                base_breakout_th = float(decision_cfg.get("breakout_threshold", 0.015))
                if vol_ratio > high_ratio:
                    hi = dyn_cfg.get("high") or {}
                    vol_mult = float(hi.get("vol_min_mult", 1.2))
                    th_mult = float(hi.get("breakout_threshold_mult", 1.5))
                    decision_cfg["breakout_min_volume_multiple"] = base_vol_min * vol_mult
                    decision_cfg["breakout_threshold"] = base_breakout_th * th_mult
                    if "require_volume_persistence" in hi:
                        decision_cfg["require_volume_persistence"] = bool(hi.get("require_volume_persistence"))
                    notes.append(
                        f"动态阈值: 高波动({vol_ratio:.2f}x) vol_min={decision_cfg['breakout_min_volume_multiple']:.2f}, "
                        f"breakout_th={decision_cfg['breakout_threshold']:.3f}"
                    )
                elif vol_ratio < low_ratio:
                    lo = dyn_cfg.get("low") or {}
                    vol_mult = float(lo.get("vol_min_mult", 0.8))
                    th_mult = float(lo.get("breakout_threshold_mult", 0.7))
                    decision_cfg["breakout_min_volume_multiple"] = base_vol_min * vol_mult
                    decision_cfg["breakout_threshold"] = base_breakout_th * th_mult
                    if "require_volume_persistence" in lo:
                        decision_cfg["require_volume_persistence"] = bool(lo.get("require_volume_persistence"))
                    notes.append(
                        f"动态阈值: 低波动({vol_ratio:.2f}x) vol_min={decision_cfg['breakout_min_volume_multiple']:.2f}, "
                        f"breakout_th={decision_cfg['breakout_threshold']:.3f}"
                    )
                else:
                    notes.append(f"动态阈值: 正常波动({vol_ratio:.2f}x)")

    symbols_cfg = signals_cfg.get("symbols") or signals_cfg.get("watchlist") or signals_cfg.get("universe")
    cache_hit = False
    preselect_cached_symbols: list[str] = []
    if (not symbols_cfg) and preselect_cache_enabled and not is_watchlist:
        cached_syms = _load_preselect_cache(
            preselect_cache_file,
            trade_date,
            preselect_cache_ttl_sec,
            hot_filter=hot_filter_enabled,
        )
        if cached_syms:
            cached_for_use = cached_syms
            if bool(signals_cfg.get("preselect_exclude_bank", False)):
                cached_for_use, removed = _exclude_preselect_symbols_from_cache(
                    cached_syms,
                    signals_cfg=signals_cfg,
                    sector_cfg=sector_cfg,
                    universe_file=universe_file,
                )
                if removed > 0:
                    notes.append(f"预选缓存板块排除: -{removed}")
                if len(cached_for_use) < preselect_n:
                    notes.append(
                        f"预选缓存不足({len(cached_for_use)}/{preselect_n})，已重建预选池。"
                    )
                    cached_for_use = []
            if cached_for_use:
                cached_for_use = cached_for_use[:preselect_n]
                signals_cfg = dict(signals_cfg)
                signals_cfg["symbols"] = cached_for_use
                preselect_cached_symbols = cached_for_use[:]
                notes.append(f"预选缓存命中: {len(cached_for_use)}")
                cache_hit = True
                symbols_cfg = cached_for_use

    if model_top_only and model_ref_enabled and model_ref_path:
        fast_preselect_symbols = None
        if is_watchlist:
            fast_preselect_symbols = None
        elif model_independent:
            fast_preselect_symbols = None
            notes.append("模型独立: 跳过共享预选快路径。")
        elif model_candidate_mode in ("rule_shared", "shared", "rule"):
            if preselect_cached_symbols:
                fast_preselect_symbols = preselect_cached_symbols
            else:
                notes.append("模型榜单快路径未命中预选缓存，回退完整链路。")
        elif model_candidate_mode == "amount_topn":
            notes.append("模型榜单快路径不支持 amount_topn，回退完整链路。")
        else:
            fast_preselect_symbols = None

        if is_watchlist or model_candidate_mode not in ("rule_shared", "shared", "rule", "amount_topn") or fast_preselect_symbols is not None:
            model_top = _compute_model_top_fast(
                trade_date=trade_date,
                params=params,
                signals_cfg=signals_cfg,
                notes=notes,
                stats=stats,
                idx_hist=idx_hist,
                universe_file=universe_file,
                manual_hist_dir=manual_hist_dir,
                model_ref_cfg=model_ref_cfg,
                model_ref_path=model_ref_path,
                model_candidate_mode=model_candidate_mode,
                model_candidate_max_symbols=model_candidate_max_symbols,
                model_sector_filter=model_sector_filter,
                news_cfg=news_cfg,
                sector_cfg=sector_cfg,
                is_watchlist=is_watchlist,
                model_independent=model_independent,
                preselect_cached_symbols=fast_preselect_symbols,
                factors_rule=None,
            )
            notes.append("模型榜单快路径: 已跳过规则列表计算")
            _finalize_timings()
            return {
                "df": pd.DataFrame(),
                "notes": notes,
                "stats": stats,
                "proxy_used": chosen_proxy,
                "use_proxy": chosen_use_proxy,
                "model_top": model_top,
            }
    if (not symbols_cfg) and use_universe_file:
        symbols_list = load_universe_symbols(universe_file)
        if symbols_list:
            market_scope = params.get("market_scope", ["sh", "sz"])
            symbols_list = filter_symbols_by_market(symbols_list, market_scope)
            symbols_list = filter_symbols_by_board(
                symbols_list,
                exclude_star=bool(params.get("exclude_star", False)),
                exclude_chi_next=bool(params.get("exclude_chi_next", False)),
                mainboard_only=bool(params.get("mainboard_only", False)),
            )
            signals_cfg = dict(signals_cfg)
            signals_cfg["symbols"] = symbols_list
            notes.append(f"使用本地股票清单: {universe_file} (n={len(symbols_list)})")
        else:
            notes.append(f"本地股票清单为空: {universe_file}")
            notes.append("请运行: python update_universe.py 生成全市场清单。")
            if not allow_eastmoney_fallback:
                _finalize_timings()
                return {"df": pd.DataFrame(), "notes": notes, "stats": stats}

    # hot_filter_* already resolved above to keep cache consistent
    hot_sectors: list[str] = []
    hot_set: set[str] = set()
    if hot_filter_enabled and not is_watchlist and not model_top_only:
        try:
            cfg_stub = {"news": news_cfg, "deepseek": deepseek_cfg}
            hot_sectors = _resolve_hot_sectors(cfg_stub, use_proxy=use_proxy, proxy=proxy)
        except Exception:
            hot_sectors = []
        hot_sectors = [h for h in hot_sectors if str(h).strip()]
        hot_set = _expand_hot_sectors(hot_sectors, news_cfg.get("hot_sector_aliases", {}) or {})
        if not hot_sectors or len(hot_sectors) < hot_filter_min:
            notes.append("热点板块: 未获取到有效热点（可能为空或数量不足）。")

    # Hot-sector prefilter: reduce universe before fetching spot.
    if hot_filter_enabled and not is_watchlist and hot_set and not model_top_only:
        symbols_cfg = signals_cfg.get("symbols") or signals_cfg.get("watchlist") or signals_cfg.get("universe")
        symbols_list = []
        if isinstance(symbols_cfg, str):
            symbols_list = [s.strip() for s in symbols_cfg.split(",") if s.strip()]
        elif isinstance(symbols_cfg, list):
            symbols_list = [str(s).strip() for s in symbols_cfg if str(s).strip()]
        if symbols_list:
            map_file = str(sector_cfg.get("map_file", "./data/sector_map.csv"))
            mapping = load_sector_map(map_file)
            if mapping:
                before = len(symbols_list)
                filtered_syms = []
                for s in symbols_list:
                    norm = _normalize_symbol(s) or str(s).zfill(6)
                    sector = mapping.get(norm, "")
                    if _is_hot_sector(sector, hot_set):
                        filtered_syms.append(norm)
                if filtered_syms:
                    signals_cfg = dict(signals_cfg)
                    signals_cfg["symbols"] = sorted(set(filtered_syms))
                    notes.append(f"热点板块预筛: 命中 {len(filtered_syms)}/{before}")
                else:
                    if hot_filter_strict:
                        notes.append("热点板块预筛命中为0（严格模式），已返回空列表。")
                        _finalize_timings()
                        return {"df": pd.DataFrame(), "notes": notes, "stats": stats}
                    notes.append("热点板块预筛命中为0，已回退全市场。")
            else:
                notes.append("热点板块预筛: 板块映射缺失，已跳过。")

    spot = None
    last_err = None
    t_spot = time.time()
    for use_proxy_i, proxy_i in _proxy_candidates(use_proxy, proxy):
        try:
            spot = fetch_a_share_daily_panel(
                trade_date=trade_date,
                signals_cfg=signals_cfg,
                use_proxy=use_proxy_i,
                proxy=proxy_i,
                allow_eastmoney_fallback=allow_eastmoney_fallback,
            )
            chosen_use_proxy = use_proxy_i
            chosen_proxy = proxy_i
            break
        except Exception as e:
            last_err = e
            continue
    stats["time_spot_ms"] = round((time.time() - t_spot) * 1000, 2)

    if spot is None:
        notes.append(f"行情面板获取失败: {last_err}")
        _finalize_timings()
        return {"df": pd.DataFrame(), "notes": notes, "stats": stats, "proxy_used": chosen_proxy, "use_proxy": chosen_use_proxy}

    if chosen_use_proxy and chosen_proxy:
        notes.append(f"当前代理: {chosen_proxy}")
    elif chosen_use_proxy and not chosen_proxy:
        notes.append("当前代理: 环境变量")
    else:
        notes.append("当前代理: 直连")

    spot_source = ""
    try:
        spot_source = str(getattr(spot, "attrs", {}).get("spot_source", "") or "")
    except Exception:
        spot_source = ""
    if spot_source:
        notes.append(f"行情来源: {spot_source}")
        stats["spot_source"] = spot_source

    if market_filter_v2_enabled and not is_watchlist and not model_top_only:
        mf = evaluate_market_filter(
            idx_hist,
            spot,
            ma_fast=int(market_filter_v2_cfg.get("ma_fast", 20) or 20),
            ma_slow=int(market_filter_v2_cfg.get("ma_slow", 60) or 60),
            vol_window=int(market_filter_v2_cfg.get("vol_window", 20) or 20),
            vol_history=int(market_filter_v2_cfg.get("vol_history", 20) or 20),
            vol_ratio_max=float(market_filter_v2_cfg.get("vol_ratio_max", 1.5) or 1.5),
            breadth_up_ratio_min=float(market_filter_v2_cfg.get("breadth_up_ratio_min", 0.40) or 0.40),
        )
        trend_ok = mf.get("trend_ok")
        vol_ok = mf.get("vol_ok")
        breadth_ok = mf.get("breadth_ok")
        breadth_ratio = mf.get("breadth_up_ratio")
        vol_ratio = mf.get("vol_ratio")
        market_filter_parts = [f"trend={trend_ok}", f"vol={vol_ok}", f"breadth={breadth_ok}"]
        if breadth_ratio is not None:
            market_filter_parts.append(f"up_ratio={float(breadth_ratio) * 100:.1f}%")
        if vol_ratio is not None:
            market_filter_parts.append(f"vol_ratio={float(vol_ratio):.2f}")
        notes.append("市场过滤V2: " + ", ".join(market_filter_parts))
        stats["market_filter_v2"] = mf
        if not bool(mf.get("passed", False)):
            block_on_fail = bool(market_filter_v2_cfg.get("block_on_fail", True))
            if block_on_fail:
                pause_signals = True
                notes.append("市场过滤V2未通过，已暂停规则信号。")

    signals_cfg = dict(signals_cfg)
    signals_cfg["use_proxy"] = chosen_use_proxy
    signals_cfg["proxy"] = chosen_proxy
    signals_cfg["allow_akshare_fallback"] = allow_eastmoney_fallback

    if spot.empty:
        notes.append("行情面板为空，可能是数据源不可用或网络受限。")
        _finalize_timings()
        return {"df": pd.DataFrame(), "notes": notes, "stats": stats, "proxy_used": chosen_proxy, "use_proxy": chosen_use_proxy}

    stats["spot_rows"] = int(len(spot))

    symbols_cfg = signals_cfg.get("symbols") or signals_cfg.get("watchlist") or signals_cfg.get("universe")
    symbols_list = []
    if isinstance(symbols_cfg, str):
        symbols_list = [s.strip() for s in symbols_cfg.split(",") if s.strip()]
    elif isinstance(symbols_cfg, list):
        symbols_list = [str(s).strip() for s in symbols_cfg if str(s).strip()]

    if symbols_list:
        universe_cfg = dict(universe_cfg)
        universe_cfg["min_avg_amount_20"] = 0
        notes.append("已检测到自选 symbol 列表，已关闭成交额过滤。")
        spot, filled_price, filled_name = _repair_spot_from_manual_hist(
            spot,
            symbols=symbols_list,
            universe_file=universe_file,
            manual_hist_dir=manual_hist_dir,
        )
        if filled_price > 0:
            notes.append(f"实时失败回填(日线): {filled_price}")
        if filled_name > 0:
            notes.append(f"名称回填: {filled_name}")

    if is_watchlist:
        notes.append("自选模式跳过价格/成交额/名称过滤。")
    else:
        spot = filter_universe(spot, universe_cfg)
        stats["after_universe"] = int(len(spot))
        spot = _exclude_preselect_sectors(
            spot,
            signals_cfg=signals_cfg,
            sector_cfg=sector_cfg,
            notes=notes,
        )
        stats["after_preselect_sector_exclude"] = int(len(spot))

    spot_base = spot.copy()
    spot_rule = _preselect_spot(spot_base, preselect_n, signals_cfg, notes)
    if len(spot_rule) < preselect_n:
        notes.append(f"预筛后候选不足: {len(spot_rule)}/{preselect_n}（可能由基础过滤或板块排除导致）")
    stats["after_preselect"] = int(len(spot_rule))
    if preselect_cache_enabled and not is_watchlist and not cache_hit:
        try:
            _save_preselect_cache(
                preselect_cache_file,
                trade_date,
                spot_rule["symbol"].astype(str).tolist(),
                hot_filter=hot_filter_enabled,
            )
            notes.append(f"预选缓存已更新: {len(spot_rule)}")
        except Exception:
            pass

    def _build_model_top_from_score(score: pd.Series | None, factors_rule: pd.DataFrame | None) -> pd.DataFrame:
        sector_map_file = str(sector_cfg.get("map_file", "./data/sector_map.csv"))
        sector_mapping = load_sector_map(sector_map_file)
        return _build_model_top_rows(
            score,
            top_n=top_n,
            universe_file=universe_file,
            manual_hist_dir=manual_hist_dir,
            fundflow_dir=fundflow_dir,
            fundflow_streak_days=fundflow_streak_days,
            sector_mapping=sector_mapping,
            factors_rule=factors_rule,
            candidate_n=int(model_risk_penalty_cfg.get("candidate_pool_size", 100) or 100),
            risk_penalty_cfg=model_risk_penalty_cfg,
            notes=notes,
        )

    def _run_rule_pipeline(current_spot_rule: pd.DataFrame, current_preselect_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.Series | None, list[str]]:
        attempt_notes: list[str] = []
        score = None

        t_factors = time.time()
        rule_syms = set(current_spot_rule["symbol"].astype(str).str.zfill(6).tolist()) if not current_spot_rule.empty else set()
        current_factors_rule = build_factors(current_spot_rule, signals_cfg, trade_date, stats=stats)
        if current_factors_rule is not None and not current_factors_rule.empty:
            current_factors_rule["symbol"] = current_factors_rule["symbol"].astype(str).str.zfill(6)
            current_factors_rule = _sanitize_factor_frame(current_factors_rule, attempt_notes, stage="因子清洗")
        stats["time_factors_ms"] = round(float(stats.get("time_factors_ms", 0.0)) + (time.time() - t_factors) * 1000, 2)

        if model_ref_enabled and model_ref_path:
            t_model = time.time()
            m = load_ref_model(model_ref_path)
            model_task = str((m.get("meta") or {}).get("task", "")).strip().lower() if m else ""
            is_minute_model = model_task == "minute_cls"
            sector_map_file = str(sector_cfg.get("map_file", "./data/sector_map.csv"))
            sector_mapping = load_sector_map(sector_map_file)
            symbols_for_model: list[str] = []
            if is_watchlist:
                symbols_for_model = sorted(rule_syms)
                attempt_notes.append(f"模型榜单候选(自选): {len(symbols_for_model)}")
            elif model_independent:
                symbols_for_model = _list_manual_hist_symbols(manual_hist_dir)
                before_candidate = len(symbols_for_model)
                market_scope = params.get("market_scope", ["sh", "sz"])
                symbols_for_model = filter_symbols_by_market(symbols_for_model, market_scope)
                symbols_for_model = filter_symbols_by_board(
                    symbols_for_model,
                    exclude_star=bool(params.get("exclude_star", False)),
                    exclude_chi_next=bool(params.get("exclude_chi_next", False)),
                    mainboard_only=bool(params.get("mainboard_only", False)),
                )
                st_removed = 0
                if bool(universe_cfg.get("exclude_st", True)):
                    name_map = _load_universe_name_map(universe_file)
                    filtered = []
                    for s in symbols_for_model:
                        nm = str(name_map.get(str(s).zfill(6), "") or "")
                        if re.search(r"ST|\*ST|退", nm):
                            st_removed += 1
                            continue
                        filtered.append(s)
                    symbols_for_model = filtered
                if model_candidate_max_symbols > 0:
                    symbols_for_model = symbols_for_model[:model_candidate_max_symbols]
                attempt_notes.append(
                    f"模型榜单候选(独立同池): {len(symbols_for_model)} "
                    f"(raw={before_candidate}, st_removed={st_removed})"
                )
            elif model_candidate_mode in ("rule_shared", "shared", "rule"):
                symbols_for_model = sorted(rule_syms)
                attempt_notes.append(f"模型榜单候选(规则同池): {len(symbols_for_model)}")
            elif model_candidate_mode == "amount_topn":
                spot_model = _preselect_by_amount(spot_base, current_preselect_n)
                symbols_for_model = sorted(set(spot_model["symbol"].astype(str).str.zfill(6).tolist())) if not spot_model.empty else []
                attempt_notes.append(f"模型榜单候选(原版成交额TopN): {len(symbols_for_model)}")
            else:
                symbols_for_model = _list_manual_hist_symbols(manual_hist_dir)
                before_candidate = len(symbols_for_model)
                market_scope = params.get("market_scope", ["sh", "sz"])
                symbols_for_model = filter_symbols_by_market(symbols_for_model, market_scope)
                symbols_for_model = filter_symbols_by_board(
                    symbols_for_model,
                    exclude_star=bool(params.get("exclude_star", False)),
                    exclude_chi_next=bool(params.get("exclude_chi_next", False)),
                    mainboard_only=bool(params.get("mainboard_only", False)),
                )
                st_removed = 0
                if bool(universe_cfg.get("exclude_st", True)):
                    name_map = _load_universe_name_map(universe_file)
                    filtered = []
                    for s in symbols_for_model:
                        nm = str(name_map.get(str(s).zfill(6), "") or "")
                        if re.search(r"ST|\*ST|退", nm):
                            st_removed += 1
                            continue
                        filtered.append(s)
                    symbols_for_model = filtered
                if model_candidate_max_symbols > 0:
                    symbols_for_model = symbols_for_model[:model_candidate_max_symbols]
                attempt_notes.append(
                    f"模型榜单候选(回测同池): {len(symbols_for_model)} "
                    f"(raw={before_candidate}, st_removed={st_removed})"
                )

            minute_model_max_symbols = int(model_ref_cfg.get("minute_max_symbols", 0) or 0)
            if is_minute_model and minute_model_max_symbols > 0 and len(symbols_for_model) > minute_model_max_symbols:
                symbols_for_model = symbols_for_model[:minute_model_max_symbols]
                attempt_notes.append(f"分钟模型候选限流: {len(symbols_for_model)} (max={minute_model_max_symbols})")

            if model_sector_filter and symbols_for_model:
                kws = _parse_sector_keywords(model_sector_filter)
                kws = _expand_sector_keywords(kws, news_cfg.get("hot_sector_aliases", {}) or {})
                if kws and sector_mapping:
                    before = len(symbols_for_model)
                    filtered_syms = []
                    for sym in symbols_for_model:
                        sym6 = str(sym).zfill(6)
                        if _is_target_sector(sector_mapping.get(sym6, ""), kws):
                            filtered_syms.append(sym6)
                    symbols_for_model = filtered_syms
                    attempt_notes.append(f"模型板块过滤: {len(symbols_for_model)}/{before}")
                elif kws and not sector_mapping:
                    attempt_notes.append("模型板块过滤: 板块映射缺失，已跳过。")

            cache_key = f"{Path(model_ref_path).stem}_{trade_date}"
            cache_path = str(Path(model_cache_dir) / f"{cache_key}.csv")
            cached_scores = None
            try:
                minute_model_cache_ttl_sec = int(signals_cfg.get("minute_model_cache_ttl_sec", 90))
            except Exception:
                minute_model_cache_ttl_sec = 90
            effective_model_cache_ttl = minute_model_cache_ttl_sec if is_minute_model else model_cache_ttl_sec
            if model_cache_enabled and effective_model_cache_ttl > 0:
                cached_scores = _model_cache_get(cache_key, effective_model_cache_ttl)
                if cached_scores is None and not is_minute_model:
                    cached_scores = _load_model_score_cache(cache_path, effective_model_cache_ttl)
                    if cached_scores is not None:
                        _model_cache_set(cache_key, cached_scores)
                if is_minute_model:
                    attempt_notes.append(f"分钟模型缓存: {effective_model_cache_ttl}s")
            elif is_minute_model:
                attempt_notes.append("分钟模型缓存: 关闭")

            missing_syms = []
            if cached_scores is not None and symbols_for_model:
                missing_syms = [s for s in symbols_for_model if s not in cached_scores.index]
                if missing_syms:
                    attempt_notes.append(
                        f"模型分缓存命中: {len(symbols_for_model) - len(missing_syms)}/{len(symbols_for_model)}，补算 {len(missing_syms)}"
                    )
                else:
                    attempt_notes.append(f"模型分缓存命中: {len(symbols_for_model)}")

            if cached_scores is None or missing_syms:
                if m:
                    model_signals_cfg = dict(signals_cfg)
                    target_syms = missing_syms if missing_syms else symbols_for_model
                    if model_task == "minute_cls":
                        if len(target_syms) > 120 and not bool(model_signals_cfg.get("minute_live_only_missing_today", False)):
                            model_signals_cfg["minute_live_only_missing_today"] = True
                            attempt_notes.append("分钟模型提速: 仅补齐缺失当日分钟线")
                        live_cap = int(model_signals_cfg.get("minute_live_max_symbols", 0) or 0)
                        if len(target_syms) > 120 and (live_cap <= 0 or live_cap > 200):
                            model_signals_cfg["minute_live_max_symbols"] = 200
                            attempt_notes.append("分钟模型提速: 实时拉取上限=200")
                        model_feats = build_minute_ref_features(
                            target_syms,
                            m,
                            model_signals_cfg,
                            trade_date=trade_date,
                        )
                    else:
                        model_feature_set = str((m.get("meta") or {}).get("feature_set") or "").strip().lower()
                        if model_feature_set:
                            model_signals_cfg["feature_set"] = model_feature_set
                        model_feats = build_ref_features(
                            target_syms,
                            trade_date,
                            model_signals_cfg,
                            market_df=idx_hist,
                            market_symbol=index_symbol,
                        )
                    if not model_feats.empty:
                        score = predict_ref_score(model_feats, m)
                    if score is None:
                        attempt_notes.append("模型参考分不可用（特征缺失或异常）。")
                else:
                    attempt_notes.append("模型参考分不可用（模型文件缺失或加载失败）。")

            if cached_scores is not None:
                score_series = cached_scores.copy()
                if isinstance(score, pd.Series) and not score.empty:
                    score_series = score_series.combine_first(score)
                    if model_cache_enabled and effective_model_cache_ttl > 0:
                        _model_cache_set(cache_key, score_series)
                        if not is_minute_model:
                            _save_model_score_cache(cache_path, score_series)
                score = score_series.reindex(symbols_for_model) if symbols_for_model else score_series
            elif isinstance(score, pd.Series):
                if model_cache_enabled and effective_model_cache_ttl > 0:
                    _model_cache_set(cache_key, score)
                    if not is_minute_model:
                        _save_model_score_cache(cache_path, score)

            if isinstance(score, pd.Series) and not score.empty:
                score.index = score.index.astype(str).str.zfill(6)
                if current_factors_rule is not None and not current_factors_rule.empty:
                    current_factors_rule = current_factors_rule.merge(
                        score.rename("model_score"),
                        left_on="symbol",
                        right_index=True,
                        how="left",
                    )
                    current_factors_rule = _sanitize_factor_frame(current_factors_rule, attempt_notes, stage="模型合并清洗")
                attempt_notes.append("模型参考分已生成（仅作参考，不参与评分）。")
            stats["time_model_ms"] = round(float(stats.get("time_model_ms", 0.0)) + (time.time() - t_model) * 1000, 2)

        t_score = time.time()
        if pause_signals:
            ranked_full = pd.DataFrame()
        else:
            current_factors_rule = _sanitize_factor_frame(current_factors_rule, attempt_notes, stage="评分前清洗")
            factors_for_rule = _apply_profile_filters(current_factors_rule, decision_cfg, attempt_notes, is_watchlist=is_watchlist)
            if sector_cfg.get("enabled"):
                map_file = str(sector_cfg.get("map_file", "./data/sector_map.csv"))
                factors_for_rule = apply_sector_map(factors_for_rule, map_file)
                if "sector" in factors_for_rule.columns and factors_for_rule["sector"].astype(str).str.len().gt(0).any():
                    attempt_notes.append("板块映射: 已加载")
                else:
                    attempt_notes.append("板块映射: 未命中")
                if hot_set and "sector" in factors_for_rule.columns:
                    factors_for_rule["hot_sector"] = factors_for_rule["sector"].apply(lambda x: 1 if _is_hot_sector(x, hot_set) else 0)
            ranked_full = score_and_rank(
                factors_for_rule,
                weights,
                top_n=current_preselect_n,
                sector_cfg=sector_cfg,
                decision_cfg=decision_cfg,
            )
            ranked_full = apply_short_term_decision(ranked_full, decision_cfg)
            intraday_cfg = decision_cfg.get("intraday_v2") or {}
            if isinstance(intraday_cfg, dict) and intraday_cfg.get("enabled"):
                intraday_cfg = dict(intraday_cfg)
                no_data_after_close = str(intraday_cfg.get("no_data_action_after_close", "") or "").upper()
                if no_data_after_close:
                    now_cn = dt.datetime.now(tz=ZoneInfo("Asia/Shanghai"))
                    if not _is_market_open_cn(now_cn):
                        intraday_cfg["no_data_action"] = no_data_after_close
                        attempt_notes.append(f"分钟V2无数据动作(收盘后): {no_data_after_close}")
                buy_before = int((ranked_full["action"] == "BUY").sum()) if "action" in ranked_full.columns else 0
                ranked_full = apply_intraday_v2(ranked_full, intraday_cfg)
                buy_after = int((ranked_full["action"] == "BUY").sum()) if "action" in ranked_full.columns else 0
                attempt_notes.append(f"分钟V2过滤: BUY {buy_before}->{buy_after}")

        conflict_cfg = decision_cfg.get("model_conflict") or {}
        if conflict_cfg.get("enabled") and "model_score" in ranked_full.columns and "score" in ranked_full.columns:
            ms = pd.to_numeric(ranked_full["model_score"], errors="coerce")
            rs = pd.to_numeric(ranked_full["score"], errors="coerce")
            model_min = float(conflict_cfg.get("model_min", 0.85))
            rule_max = float(conflict_cfg.get("rule_max", decision_cfg.get("min_score", 20.0)))
            action_col = ranked_full.get("action")
            if action_col is not None:
                conflict_mask = (ms >= model_min) & (rs < rule_max)
                if conflict_mask.any():
                    ranked_full.loc[conflict_mask, "action"] = "WATCH"
                    if "reason" in ranked_full.columns:
                        reason_add = f"模型强但规则弱(≥{model_min:.2f}/<{rule_max:.1f})"
                        ranked_full.loc[conflict_mask, "reason"] = (
                            ranked_full.loc[conflict_mask, "reason"].fillna("").astype(str).apply(
                                lambda x: reason_add if (not x or x == "nan") else (x if reason_add in x else f"{x}; {reason_add}")
                            )
                        )
                    attempt_notes.append(
                        f"模型冲突降级: model>={model_min:.2f} 且 score<{rule_max:.1f} -> WATCH ({int(conflict_mask.sum())})"
                    )
        if model_ref_filter_enabled:
            if "model_score" in ranked_full.columns and "action" in ranked_full.columns:
                ms = pd.to_numeric(ranked_full["model_score"], errors="coerce")
                bw_mask = ranked_full["action"].isin(["BUY", "WATCH"])
                ok_mask = (ms >= model_ref_min_score) & bw_mask
                drop_mask = bw_mask & ~ok_mask
                if drop_mask.any():
                    ranked_full.loc[drop_mask, "action"] = "AVOID"
                    if "reason" in ranked_full.columns:
                        reason_add = f"模型<{model_ref_min_score:.2f}"
                        ranked_full.loc[drop_mask, "reason"] = (
                            ranked_full.loc[drop_mask, "reason"].fillna("").astype(str).apply(
                                lambda x: reason_add if (not x or x == "nan") else (x if reason_add in x else f"{x}; {reason_add}")
                            )
                        )
                if bw_mask.any():
                    attempt_notes.append(
                        f"模型过滤: 阈值>= {model_ref_min_score:.2f}, BUY/WATCH保留 {int(ok_mask.sum())}/{int(bw_mask.sum())}"
                    )
            else:
                attempt_notes.append("模型过滤启用但未生成模型分或动作列，已跳过。")
        if exclude_limit_up and "pct_chg" in ranked_full.columns:
            ranked_full = ranked_full[pd.to_numeric(ranked_full["pct_chg"], errors="coerce") < limit_up_pct].reset_index(drop=True)

        if "pct_source" in ranked_full.columns:
            src_counts = ranked_full["pct_source"].value_counts().to_dict()
            attempt_notes.append("涨跌幅来源统计: " + ", ".join(f"{k}={v}" for k, v in src_counts.items()))

            if exclude_non_realtime_pct:
                filtered = ranked_full[ranked_full["pct_source"].isin(["spot", "spot_calc", "spot_fix"])].reset_index(drop=True)
                if filtered.empty and not ranked_full.empty:
                    attempt_notes.append("实时涨跌幅不可用，已保留历史/估算数据。")
                else:
                    ranked_full = filtered
            elif is_watchlist:
                attempt_notes.append("自选模式保留非实时涨跌幅（可勾选“排除非实时涨跌幅”过滤）。")
        stats["time_score_ms"] = round(float(stats.get("time_score_ms", 0.0)) + (time.time() - t_score) * 1000, 2)
        ranked_main = ranked_full.head(top_n).copy()

        ranked_out = ranked_main
        if "model_score" in ranked_full.columns:
            ms = pd.to_numeric(ranked_full["model_score"], errors="coerce")
            extra = ranked_full[(ms >= model_ref_show_min)].copy()
            if not extra.empty:
                extra = extra[~extra["symbol"].isin(set(ranked_main["symbol"]))].copy()
                if not extra.empty:
                    reason_add = f"模型>={model_ref_show_min:.2f}(仅参考)"
                    if "reason" in extra.columns:
                        extra["reason"] = extra["reason"].fillna("").astype(str).apply(
                            lambda x: reason_add if (not x or x == "nan") else (x if reason_add in x else f"{x}; {reason_add}")
                        )
                    ranked_out = pd.concat([ranked_main, extra], ignore_index=True)
                    attempt_notes.append(f"模型强信号追加: {len(extra)} 条 (阈值>={model_ref_show_min:.2f})")

        return ranked_out, ranked_full, current_factors_rule, score, attempt_notes

    try:
        expand_enabled = dynamic_preselect_expand and (not is_watchlist) and (not model_top_only) and (not pause_signals)
        candidate_sizes = dynamic_preselect_steps if expand_enabled else [preselect_n]
        chosen_preselect_n = preselect_n
        chosen_spot_rule = spot_rule
        chosen_factors_rule = pd.DataFrame()
        chosen_score = None
        chosen_notes: list[str] = []
        ranked_full = pd.DataFrame()

        for attempt_idx, current_n in enumerate(candidate_sizes):
            preselect_notes: list[str] = []
            if attempt_idx == 0:
                current_spot_rule = spot_rule
            else:
                current_spot_rule = _preselect_spot(spot_base, current_n, signals_cfg, preselect_notes)
                if len(current_spot_rule) < current_n:
                    preselect_notes.append(f"预筛后候选不足: {len(current_spot_rule)}/{current_n}（可能由基础过滤或板块排除导致）")

            ranked_candidate, ranked_full_candidate, factors_candidate, score_candidate, attempt_notes = _run_rule_pipeline(current_spot_rule, current_n)
            buy_count = int((ranked_full_candidate["action"] == "BUY").sum()) if "action" in ranked_full_candidate.columns else 0
            chosen_preselect_n = current_n
            chosen_spot_rule = current_spot_rule
            chosen_factors_rule = factors_candidate
            ranked = ranked_candidate
            ranked_full = ranked_full_candidate
            chosen_score = score_candidate
            chosen_notes = preselect_notes + attempt_notes

            if buy_count >= dynamic_preselect_buy_min or attempt_idx == len(candidate_sizes) - 1:
                if current_n > preselect_n:
                    notes.append(f"动态扩池结果: 预选={current_n}, BUY={buy_count}")
                break

            next_n = candidate_sizes[attempt_idx + 1]
            notes.append(f"动态扩池: BUY={buy_count}，预选 {current_n}->{next_n}")

        stats["after_preselect"] = int(len(chosen_spot_rule))
        stats["after_preselect_target"] = int(chosen_preselect_n)
        notes.extend(chosen_notes)
        ranked = _attach_fundflow_tags(ranked, fundflow_dir=fundflow_dir, streak_days=fundflow_streak_days)
        if isinstance(chosen_score, pd.Series) and not chosen_score.empty:
            model_top = _build_model_top_from_score(chosen_score, chosen_factors_rule)
    except Exception as e:
        notes.append(f"因子/评分失败: {e}")
        tb_lines = [line.strip() for line in traceback.format_exc(limit=8).splitlines() if line.strip()]
        if tb_lines:
            notes.append("异常定位: " + " | ".join(tb_lines[-4:]))
        _finalize_timings()
        return {"df": pd.DataFrame(), "notes": notes, "stats": stats, "model_top": model_top}

    notes.append(
        f"历史行情统计: ok={stats.get('ok', 0)}, empty={stats.get('hist_empty', 0)}, short={stats.get('hist_short', 0)}, error={stats.get('hist_error', 0)}"
    )
    if any(k in stats for k in ("time_hist_io_ms", "time_realtime_fix_ms", "time_factor_calc_ms", "time_realtime_fix_batch_ms")):
        notes.append(
            "因子耗时拆分(累计): "
            f"hist_io_sum={round(float(stats.get('time_hist_io_ms', 0.0)), 2)}ms, "
            f"realtime_fix_sum={round(float(stats.get('time_realtime_fix_ms', 0.0)), 2)}ms, "
            f"calc_sum={round(float(stats.get('time_factor_calc_ms', 0.0)), 2)}ms, "
            f"batch_prefetch={round(float(stats.get('time_realtime_fix_batch_ms', 0.0)), 2)}ms"
        )
        if int(stats.get("spot_fix_batch_candidates", 0)) > 0:
            notes.append(
                "补价统计: "
                f"batch_candidates={int(stats.get('spot_fix_batch_candidates', 0))}, "
                f"batch_hits={int(stats.get('spot_fix_batch_hits', 0))}, "
                f"batch_used={int(stats.get('spot_fix_batch_used', 0))}, "
                f"single_try={int(stats.get('spot_fix_single_try', 0))}, "
                f"single_hit={int(stats.get('spot_fix_single_hit', 0))}, "
                f"single_error={int(stats.get('spot_fix_single_error', 0))}"
            )
    if stats.get("spot_close_fixed") or stats.get("spot_pct_fixed"):
        notes.append(
            f"现价/涨跌幅纠正: close_fixed={stats.get('spot_close_fixed', 0)}, pct_fixed={stats.get('spot_pct_fixed', 0)}"
        )
        if stats.get("hist_error_msgs"):
            notes.append("hist_error 示例: " + " | ".join(stats["hist_error_msgs"]))

    _finalize_timings()
    return {
        "df": ranked,
        "notes": notes,
        "stats": stats,
        "proxy_used": chosen_proxy,
        "use_proxy": chosen_use_proxy,
        "model_top": model_top,
    }


def refresh_realtime_for_view(
    df: pd.DataFrame,
    *,
    use_proxy: bool,
    proxy: str,
    max_rows: int = 50,
    provider: str = "auto",
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    n = min(len(out), max_rows)
    symbols = [str(s).strip() for s in out.head(n)["symbol"].tolist() if str(s).strip()]
    if not symbols:
        return out

    def _apply_valid_spot(spot_df: pd.DataFrame, source: str) -> bool:
        if spot_df is None or spot_df.empty:
            return False
        local_spot = spot_df.set_index("symbol")
        valid_close = pd.to_numeric(local_spot.get("close"), errors="coerce") > 0
        valid_pct = pd.to_numeric(local_spot.get("pct_chg"), errors="coerce").notna()
        valid_symbols = set(local_spot.index[valid_close | valid_pct].astype(str))
        if not valid_symbols:
            return False
        valid_index = out["symbol"].astype(str).isin(valid_symbols)
        if "close" in local_spot.columns and "close" in out.columns:
            mapped_close = out["symbol"].map(local_spot["close"])
            close_ok = pd.to_numeric(mapped_close, errors="coerce") > 0
            out.loc[valid_index & close_ok, "close"] = mapped_close[valid_index & close_ok]
        if "pct_chg" in local_spot.columns and "pct_chg" in out.columns:
            mapped_pct = out["symbol"].map(local_spot["pct_chg"])
            pct_ok = pd.to_numeric(mapped_pct, errors="coerce").notna()
            out.loc[valid_index & pct_ok, "pct_chg"] = mapped_pct[valid_index & pct_ok]
        if "pct_source" in out.columns:
            out.loc[valid_index, "pct_source"] = source
        return True

    providers = [provider]
    if provider and provider != "tencent":
        providers.append("tencent")
    for active_provider in providers:
        try:
            spot = fetch_a_share_daily_panel(
                signals_cfg={"symbols": symbols, "realtime_provider": active_provider},
                use_proxy=use_proxy,
                proxy=proxy,
                allow_eastmoney_fallback=False,
            )
        except Exception:
            spot = pd.DataFrame()
        if _apply_valid_spot(spot, "spot" if active_provider == provider else f"spot:{active_provider}"):
            return out

    try:
        realtime = get_realtime(",".join(symbols), provider=provider, use_proxy=use_proxy, proxy=proxy)
    except Exception:
        realtime = pd.DataFrame()
    if realtime is None or realtime.empty:
        return out
    realtime = realtime.set_index("symbol")
    for col in ["close", "pct_chg", "pct_source"]:
        if col in realtime.columns and col in out.columns:
            out.loc[out["symbol"].isin(realtime.index), col] = out["symbol"].map(realtime[col])
    return out
