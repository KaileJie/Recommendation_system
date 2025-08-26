# service/recall_builder.py
"""
Build multi-strategy Redis inverted index (keyword -> Set[article_url]) from MongoDB.

Strategies:
  - tfidf      -> uses Mongo field: tfidf_keywords
  - textrank   -> uses Mongo field: textrank_keywords
  - final      -> uses Mongo field: final_keywords

Redis keys written:
  pbs:index:<strategy>:<keyword> -> Set(article_url)
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Set

import pymongo

from config.recall_config import MONGO, REDIS, STRATEGIES
from dao.redis_cache import RedisCache


FIELD_MAP = {
    "tfidf": "tfidf_keywords",
    "textrank": "textrank_keywords",
    "final": "final_keywords",
}


def _connect_mongo():
    client = pymongo.MongoClient(MONGO["URI"])
    coll = client[MONGO["DATABASE"]][MONGO["COLLECTION"]]
    return client, coll


def _normalize_kw_list(kws) -> List[str]:
    return sorted({str(k).strip().lower() for k in (kws or []) if str(k).strip()})


def _load_articles(days: int | None, limit: int | None):
    """
    从 Mongo 读取文章，带齐三套策略字段：
      url, tfidf_keywords, textrank_keywords, final_keywords
    """
    client, coll = _connect_mongo()
    try:
        query = {}
        if days is not None:
            cutoff = datetime.utcnow() - timedelta(days=days)

            query = {"publish_date": {"$gte": cutoff.isoformat()}}

        projection = {
            "url": 1,
            FIELD_MAP["tfidf"]: 1,
            FIELD_MAP["textrank"]: 1,
            FIELD_MAP["final"]: 1,
            "_id": 0,
        }

        cursor = coll.find(query, projection)
        if limit:
            cursor = cursor.limit(int(limit))

        for doc in cursor:
            url = (doc.get("url") or "").strip()
            if not url:
                continue
            yield {
                "url": url,
                "tfidf": _normalize_kw_list(doc.get(FIELD_MAP["tfidf"])),
                "textrank": _normalize_kw_list(doc.get(FIELD_MAP["textrank"])),
                "final": _normalize_kw_list(doc.get(FIELD_MAP["final"])),
            }
    finally:
        client.close()


def _build_inverted_index(articles: List[dict], strategy: str) -> Dict[str, List[str]]:
    """
    构建单一策略的倒排索引：
      {keyword: [url, ...]}
    """
    inv = defaultdict(set)  # type: Dict[str, Set[str]]
    for doc in articles:
        for kw in doc.get(strategy, []):
            inv[kw].add(doc["url"])
    return {k: sorted(v) for k, v in inv.items()}


def main():
    parser = argparse.ArgumentParser(description="Rebuild multi-strategy Redis inverted index from MongoDB.")
    parser.add_argument("--days", type=int, default=None, help="只处理最近 N 天（默认全量）")
    parser.add_argument("--limit", type=int, default=None, help="最多处理文章数（默认不限制）")
    parser.add_argument("--no-clear", action="store_true", help="不清空旧索引，直接写入")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入 Redis")
    args = parser.parse_args()

    expire = REDIS.get("EXPIRE")

    print(f"[builder] Reading from Mongo: {MONGO['URI']} / {MONGO['DATABASE']}.{MONGO['COLLECTION']}")
    if args.days:
        print(f"[builder] Time window: last {args.days} days")
    if args.limit:
        print(f"[builder] Limit: {args.limit} articles")

    articles = list(_load_articles(days=args.days, limit=args.limit))
    print(f"[builder] Loaded articles: {len(articles)}")

    # 预览每篇文章的三套关键词是否存在（仅统计）
    counts = {s: 0 for s in STRATEGIES}
    for d in articles:
        for s in STRATEGIES:
            if d.get(s):
                counts[s] += 1
    print(f"[builder] Docs with keywords - tfidf={counts['tfidf']}, textrank={counts['textrank']}, final={counts['final']}")

    cache = RedisCache()

    # 可选清理：分别清空三套策略的旧 key
    if not args.no_clear:
        for s in STRATEGIES:
            base_prefix = f"{REDIS['INDEX_PREFIX']}:{s}"
            print(f"[builder] Clearing old keys: {base_prefix}:*")
            cache.clear_all_keys_with_prefix(base_prefix)

    # 逐策略构建并写入
    for s in STRATEGIES:
        inv_index = _build_inverted_index(articles, s)
        print(f"[builder] Strategy='{s}': {len(inv_index)} keywords")

        # 预览前 5 个关键词
        for i, (kw, urls) in enumerate(sorted(inv_index.items())[:5], 1):
            print(f"  [{s}] #{i} {kw} -> {len(urls)} urls")

        if args.dry_run:
            continue

        base_prefix = f"{REDIS['INDEX_PREFIX']}:{s}"
        print(f"[builder] Writing index to Redis: {base_prefix}:<keyword>")
        cache.save_inverted_index(base_prefix, inv_index, expire=expire)

    if args.dry_run:
        print("[builder] Dry run: no write to Redis.")
    else:
        print("[builder] Done.")


if __name__ == "__main__":
    main()
