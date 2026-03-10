from __future__ import annotations

import json
import time
import re
import html
from typing import Any

import requests


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.eastmoney.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

POSITIVE_WORDS = [
    "上涨",
    "上行",
    "利好",
    "超预期",
    "大涨",
    "涨停",
    "增持",
    "回购",
    "盈利",
    "扭亏",
    "中标",
    "突破",
    "高增",
    "扩产",
    "获批",
    "上市",
    "业绩预增",
]

NEGATIVE_WORDS = [
    "下跌",
    "下行",
    "利空",
    "暴跌",
    "跌停",
    "亏损",
    "预亏",
    "减持",
    "回撤",
    "被罚",
    "调查",
    "停产",
    "破产",
    "爆雷",
    "终止",
    "取消",
    "下修",
    "业绩预减",
    "风险",
    "警示",
]


def _jsonp_to_json(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("jQuery") and "(" in t and ")" in t:
        t = t[t.find("(") + 1 : t.rfind(")")]
    if not t:
        return {}
    return json.loads(t)


def _build_em_params(keyword: str, limit: int) -> dict[str, str]:
    payload = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": limit,
                "preTag": "",
                "postTag": "",
            }
        },
    }
    ts = str(int(time.time() * 1000))
    return {
        "cb": f"jQuery{ts}",
        "param": json.dumps(payload, ensure_ascii=False),
        "_": ts,
    }


def _request_jsonp(url: str, params: dict[str, str], *, use_proxy: bool, proxy: str) -> dict[str, Any]:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        from curl_cffi import requests as curl_requests  # type: ignore

        resp = curl_requests.get(url, params=params, timeout=10, impersonate="chrome120")
        resp.raise_for_status()
        return _jsonp_to_json(resp.text)
    except Exception:
        r = requests.get(
            url,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=10,
            proxies=proxies,
        )
        r.raise_for_status()
        return _jsonp_to_json(r.text)


def _strip_html(text: str) -> str:
    t = html.unescape(text or "")
    t = re.sub(r"<[^>]+>", "", t)
    return t.strip()


def fetch_news_eastmoney(
    keyword: str,
    limit: int = 20,
    *,
    use_proxy: bool = False,
    proxy: str = "",
) -> list[dict[str, Any]]:
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    params = _build_em_params(keyword, limit)
    data = _request_jsonp(url, params, use_proxy=use_proxy, proxy=proxy)
    result = (data.get("result") or {}).get("cmsArticleWebOld") or []
    items = []
    for it in result:
        title = _strip_html(it.get("title", ""))
        content = _strip_html(it.get("content", ""))
        items.append(
            {
                "title": title,
                "content": content,
                "time": it.get("date", ""),
                "url": it.get("url", ""),
                "source": it.get("source", "东方财富"),
                "keywords": _strip_html(it.get("keywords", "")),
            }
        )
    return items


def simple_sentiment(text: str) -> tuple[int, str]:
    t = (text or "").lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    score = pos - neg
    if score > 0:
        return score, "正面"
    if score < 0:
        return score, "负面"
    return score, "中性"
