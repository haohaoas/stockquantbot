from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any

import requests

from sqdata.news_fetcher import fetch_news_eastmoney, simple_sentiment


def _resolve_deepseek_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _flatten_keywords(items: list[dict[str, Any]]) -> list[str]:
    keywords: list[str] = []
    for it in items:
        raw = str(it.get("keywords", "") or "")
        if raw:
            for part in raw.replace("；", ",").replace("，", ",").split(","):
                kw = part.strip()
                if kw:
                    keywords.append(kw)
    return keywords


def _extract_hot(items: list[dict[str, Any]], limit: int = 6) -> list[str]:
    keywords = _flatten_keywords(items)
    if not keywords:
        titles = " ".join(str(i.get("title", "")) for i in items)
        for token in titles.replace("，", " ").replace("、", " ").split():
            token = token.strip()
            if len(token) >= 2:
                keywords.append(token)
    if not keywords:
        return []
    counts = Counter(keywords)
    return [k for k, _ in counts.most_common(limit)]


def _simple_market_sentiment(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "market_sentiment": "neutral",
            "score": 50,
            "risk_level": "medium",
            "hot_themes": [],
            "hot_sectors": [],
            "suggested_style": "缺少新闻样本，建议保持中性与轻仓。",
            "comment": "当前样本不足，情绪判断偏中性。",
            "news_sample_size": 0,
            "last_news_source": "",
        }

    total = 0
    for it in items:
        title = str(it.get("title", "") or "")
        content = str(it.get("content", "") or "")
        score, _ = simple_sentiment(title + " " + content)
        total += score
    avg = total / max(len(items), 1)
    score = int(max(0, min(100, 50 + avg * 8)))
    if score >= 62:
        market = "bullish"
        risk = "low"
        style = "可偏强势/趋势型，注意回撤控制。"
    elif score <= 38:
        market = "bearish"
        risk = "high"
        style = "偏防守，降低仓位与追高频率。"
    else:
        market = "neutral"
        risk = "medium"
        style = "轻仓试错，优先关注确定性信号。"
    hot = _extract_hot(items)
    return {
        "market_sentiment": market,
        "score": score,
        "risk_level": risk,
        "hot_themes": hot,
        "hot_sectors": hot,
        "suggested_style": style,
        "comment": "基于新闻标题的简单情绪汇总。",
        "news_sample_size": len(items),
        "last_news_source": "",
    }


def _llm_market_sentiment(
    items: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    proxy: str = "",
) -> dict[str, Any] | None:
    if not items or not api_key:
        return None
    titles = [str(i.get("title", "") or "") for i in items][:40]
    content = "\n".join(f"- {t}" for t in titles if t)
    prompt = (
        "你是市场情绪分析助手。根据新闻标题判断整体市场情绪。"
        "请输出 JSON，字段包括: market_sentiment(bullish/bearish/neutral), "
        "score(0-100), risk_level(low/medium/high), hot_themes(list), "
        "hot_sectors(list), suggested_style, comment。"
        "不要输出多余文本。\n\n新闻标题:\n"
        f"{content}\n"
    )
    url = _resolve_deepseek_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你输出严格 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20, proxies=proxies)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not text:
            return None
        result = json.loads(text)
        if not isinstance(result, dict):
            return None
        result["news_sample_size"] = len(items)
        return result
    except Exception:
        return None


def get_market_news_sentiment(
    *,
    limit: int = 30,
    use_proxy: bool = False,
    proxy: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "deepseek-chat",
) -> dict[str, Any]:
    items = fetch_news_eastmoney("A股", limit=limit, use_proxy=use_proxy, proxy=proxy)
    source = "eastmoney"

    if api_key and base_url:
        llm = _llm_market_sentiment(items, api_key=api_key, base_url=base_url, model=model, proxy=proxy if use_proxy else "")
        if llm:
            llm["last_news_source"] = source
            llm["fetch_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return llm

    simple = _simple_market_sentiment(items)
    simple["last_news_source"] = source
    simple["fetch_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return simple
