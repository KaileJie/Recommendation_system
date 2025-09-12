# service/recall_service.py
"""
Keyword recall (OR semantics) over multi-strategy Redis Sets:
  pbs:index:<strategy>:<keyword> -> Set(article_url)

Public APIs:
- recall_by_keywords(keywords, strategy="final", limit=None) -> List[str]
- recall_by_query(query, strategy="final", limit=None) -> List[str]
"""

from __future__ import annotations
from typing import Iterable, List, Set

from config.recall_config import REDIS, STRATEGIES
from dao.redis_cache import RedisCache


def _normalize_keywords(kw_iter: Iterable[str]) -> List[str]:
    """lower + strip + 去重，过滤空串"""
    seen: Set[str] = set()
    out: List[str] = []
    for kw in kw_iter or []:
        k = str(kw).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _base_prefix(strategy: str) -> str:
    s = (strategy or "final").lower()
    if s not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {s}. Allowed: {STRATEGIES}")
    return f"{REDIS['INDEX_PREFIX']}:{s}"


def recall_by_keywords(keywords: Iterable[str], strategy: str = "final", limit: int | None = None) -> List[str]:
    """
    OR 召回：对多个关键词的 Set 做 SUNION，返回去重后的 URL 列表
    """
    kws = _normalize_keywords(keywords)
    if not kws:
        return []
    base = _base_prefix(strategy)
    keys = [f"{base}:{k}" for k in kws]

    cache = RedisCache()
    urls = cache.client.sunion(*keys) if keys else set()

    res = list(urls)
    return res[:limit] if (limit and limit > 0) else res


def recall_by_query(query: str, strategy: str = "final", limit: int | None = None) -> List[str]:
    """
    从原始 query 文本做极简切词（空白分词）后 OR 召回
    """
    kws = _normalize_keywords(query.split()) if query else []
    return recall_by_keywords(kws, strategy=strategy, limit=limit)
