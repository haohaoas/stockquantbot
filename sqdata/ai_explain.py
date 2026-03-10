from __future__ import annotations

import json
from typing import Any

import requests


def _fmt_num(val: Any, digits: int = 2) -> str:
    try:
        num = float(val)
    except Exception:
        return "--"
    return f"{num:.{digits}f}"


def _fmt_pct(val: Any, digits: int = 2) -> str:
    try:
        num = float(val) * 100
    except Exception:
        return "--"
    return f"{num:.{digits}f}%"


def _safe(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if not row:
        return default
    val = row.get(key, default)
    if val is None:
        return default
    return val


def _build_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": _safe(row, "symbol", ""),
        "name": _safe(row, "name", ""),
        "action": _safe(row, "action", ""),
        "score": _safe(row, "score", None),
        "model_score": _safe(row, "model_score", None),
        "close": _safe(row, "close", None),
        "pct_chg": _safe(row, "pct_chg", None),
        "pct_source": _safe(row, "pct_source", ""),
        "breakout": _safe(row, "breakout", None),
        "breakout_level": _safe(row, "breakout_level", None),
        "breakout_strength": _safe(row, "breakout_strength", None),
        "breakout_days_since": _safe(row, "breakout_days_since", None),
        "vol_ratio": _safe(row, "vol_ratio", None),
        "vol_ok": _safe(row, "vol_ok", None),
        "ma5": _safe(row, "ma5", None),
        "ma10": _safe(row, "ma10", None),
        "ma20": _safe(row, "ma20", None),
        "ma60": _safe(row, "ma60", None),
        "price_ma20_dist": _safe(row, "price_ma20_dist", None),
        "price_ma60_dist": _safe(row, "price_ma60_dist", None),
        "rsi": _safe(row, "rsi", None),
        "macd_hist": _safe(row, "macd_hist", None),
        "trend": _safe(row, "trend", ""),
        "trend_score": _safe(row, "trend_score", None),
        "sector": _safe(row, "sector", ""),
        "sector_boost": _safe(row, "sector_boost", None),
        "reason": _safe(row, "reason", ""),
        "entry": _safe(row, "entry", None),
        "stop": _safe(row, "stop", None),
        "target": _safe(row, "target", None),
        "buy_trigger": _safe(row, "buy_trigger", ""),
        "stop_ref": _safe(row, "stop_ref", ""),
        "entry_hint": _entry_hint(row),
    }


def _build_prompt(row: dict[str, Any]) -> str:
    summary = _build_summary(row)
    summary_json = json.dumps(summary, ensure_ascii=False)
    return (
        "你是量化交易助手。请根据以下结构化信息，用简洁中文解释该股票"
        "为什么被系统标记为对应动作，并说明主要指标信号。"
        "不要给出主观买卖建议，不要引用不存在的数据，4-6句话即可。"
        "如果存在 entry/stop/target，请给出“参考入场/参考止损/参考目标”，"
        "若为空则不要编造。"
        "需要给出“更好的入场机会”，可用回踩/站稳/放量确认等条件表述。"
        f"\n\n数据:{summary_json}\n"
    )


def _parse_float_from_text(text: str) -> float | None:
    if not text:
        return None
    import re

    m = re.search(r"([0-9]+(?:\\.[0-9]+)?)", str(text))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _derive_levels(row: dict[str, Any]) -> dict[str, float | None]:
    entry = _safe(row, "entry", None)
    if entry is None:
        entry = _safe(row, "breakout_level", None)
    stop = _safe(row, "stop", None)
    if stop is None:
        stop = _parse_float_from_text(str(_safe(row, "stop_ref", "")))
    target = _safe(row, "target", None)
    return {"entry": entry, "stop": stop, "target": target}


def _entry_hint(row: dict[str, Any]) -> str:
    pct = _safe(row, "pct_chg", None)
    ma20_dist = _safe(row, "price_ma20_dist", None)
    ma60_dist = _safe(row, "price_ma60_dist", None)
    breakout = _safe(row, "breakout", None)
    breakout_level = _safe(row, "breakout_level", None)
    vol_ok = _safe(row, "vol_ok", None)

    high_move = False
    try:
        if pct is not None and float(pct) >= 6.0:
            high_move = True
    except Exception:
        pass
    try:
        if ma20_dist is not None and float(ma20_dist) >= 0.08:
            high_move = True
    except Exception:
        pass
    try:
        if ma60_dist is not None and float(ma60_dist) >= 0.15:
            high_move = True
    except Exception:
        pass

    if high_move:
        if breakout_level is not None:
            return "涨幅偏高，优先等待回踩至突破位附近或MA10/MA20站稳后再观察。"
        return "涨幅偏高，优先等待回踩至MA10/MA20站稳后再观察。"

    if breakout and vol_ok:
        return "可关注突破位附近的回踩确认，放量继续有效时再观察。"
    if breakout and not vol_ok:
        return "突破但量能一般，优先等待放量确认再观察。"
    return "可关注回踩均线后的企稳信号作为更佳入场机会。"


def _local_explain(row: dict[str, Any]) -> str:
    action = _safe(row, "action", "") or "N/A"
    reason = _safe(row, "reason", "") or ""
    parts: list[str] = [f"规则动作为 {action}。"]
    if reason:
        parts.append(f"规则原因: {reason}。")
    breakout = _safe(row, "breakout", None)
    if breakout is not None:
        parts.append(f"突破信号: {int(bool(breakout))}。")
    vol_ratio = _safe(row, "vol_ratio", None)
    if vol_ratio is not None:
        parts.append(f"量比约 {_fmt_num(vol_ratio)}。")
    ma20 = _safe(row, "ma20", None)
    ma60 = _safe(row, "ma60", None)
    if ma20 is not None and ma60 is not None:
        parts.append(f"MA20 {_fmt_num(ma20)} / MA60 {_fmt_num(ma60)}。")
    rsi = _safe(row, "rsi", None)
    if rsi is not None:
        parts.append(f"RSI {_fmt_num(rsi)}。")
    pct = _safe(row, "pct_chg", None)
    if pct is not None:
        parts.append(f"今日涨跌幅 {_fmt_num(pct)}%。")
    levels = _derive_levels(row)
    if any(v is not None for v in levels.values()):
        parts.append(
            "参考入场/止损/目标: "
            f"{_fmt_num(levels.get('entry'))} / {_fmt_num(levels.get('stop'))} / {_fmt_num(levels.get('target'))}。"
        )
    hint = _entry_hint(row)
    if hint:
        parts.append(f"更好的入场机会: {hint}")
    parts.append("该解读仅用于信号解释，不构成投资建议。")
    return "".join(parts)


def _resolve_deepseek_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def explain_row(
    row: dict[str, Any],
    cfg: dict[str, Any],
    *,
    api_key: str,
    proxy: str = "",
) -> tuple[str, str, dict[str, float | None]]:
    deep_cfg = cfg.get("deepseek", {}) or {}
    if not deep_cfg.get("enabled", False) or not api_key:
        return _local_explain(row), "local", _derive_levels(row)

    url = _resolve_deepseek_url(str(deep_cfg.get("base_url", "https://api.deepseek.com")))
    model = str(deep_cfg.get("model", "deepseek-chat"))
    temperature = float(deep_cfg.get("temperature", 0.2))
    max_tokens = int(deep_cfg.get("max_tokens", 300))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是谨慎的量化研究助手。"},
            {"role": "user", "content": _build_prompt(row)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20, proxies=proxies)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        text = (text or "").strip()
        if not text:
            return _local_explain(row), "local", _derive_levels(row)
        return text, "deepseek", _derive_levels(row)
    except Exception:
        return _local_explain(row), "local", _derive_levels(row)
