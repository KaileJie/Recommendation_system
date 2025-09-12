# data/update_article_keywords.py
"""
Recompute article keywords (tfidf / textrank / final) and write back to Mongo.

Usage examples:
    # 最近60天，Top10，按需训练TF-IDF（若无模型则训练），更新所有文档
    python data/update_article_keywords.py

    # 最近180天，Top15，并强制重训TF-IDF
    python data/update_article_keywords.py --days 180 --topk 15 --force-retrain

    # 只更新缺少关键词字段的文档（存在就跳过），最多处理 200 篇
    python data/update_article_keywords.py --only-missing --limit 200

    # 使用融合 RRF 策略，并采用 zscore 归一化
    python data/update_article_keywords.py --method rrf --normalize zscore
"""

from __future__ import annotations
import argparse
import logging
from typing import List, Dict, Any

from config.recall_config import MONGO, FUSION, ARTIFACTS
from dao.mongo_db import ArticleDAO
from model.tfidf_model import load_fitted_extractor, TFIDFKeywordExtractor
from model.textrank_model import TextRankKeywordExtractor
from model.final_model import FinalKeywordExtractor

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("update_kws")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update TF-IDF / TextRank / Final keywords for articles in MongoDB.")
    p.add_argument("--days", type=int, default=60, help="Only process articles with publish_date in last N days.")
    p.add_argument("--limit", type=int, default=None, help="Max number of articles to process.")
    p.add_argument("--topk", type=int, default=FUSION.get("TOP_K", 20), help="Top K keywords per strategy.")
    p.add_argument("--force-retrain", action="store_true", help="Force re-train TF-IDF model even if cached model exists.")
    p.add_argument("--only-missing", action="store_true", help="Only update docs that are missing keyword fields or have empty lists.")
    p.add_argument("--method", choices=("weighted", "rrf"), default=FUSION.get("METHOD", "weighted"), help="Fusion method for final keywords.")
    p.add_argument("--alpha", type=float, default=FUSION.get("ALPHA", 0.6), help="Weight for TF-IDF in weighted fusion.")
    p.add_argument("--normalize", choices=("minmax", "zscore"), default=FUSION.get("NORMALIZE", "minmax"), help="Normalization method for fusion.")
    return p.parse_args()


def _need_update(doc: Dict[str, Any]) -> bool:
    """判断该文档是否缺三套关键词（用于 --only-missing）"""
    def _empty(x):
        return (x is None) or (isinstance(x, list) and len(x) == 0)
    return _empty(doc.get("tfidf_keywords")) or _empty(doc.get("textrank_keywords")) or _empty(doc.get("final_keywords"))


def main():
    args = _parse_args()
    log.info("Target Mongo: %s / %s.%s", MONGO["URI"], MONGO["DATABASE"], MONGO["COLLECTION"])
    log.info("Params: days=%s limit=%s topk=%s force_retrain=%s only_missing=%s method=%s alpha=%.3f normalize=%s",
             args.days, args.limit, args.topk, args.force_retrain, args.only_missing, args.method, args.alpha, args.normalize)

    # 1) 读取文章（仅取需要字段；ArticleDAO 默认返回 _id/url/content/publish_date/title/可能的 final_keywords）
    dao = ArticleDAO()
    articles = dao.get_recent_articles(days=args.days, limit=(args.limit or 1000000))
    if not articles:
        log.warning("No articles found in the time window.")
        dao.close()
        return

    # 若只处理缺字段文档，则再查一次每个文档的关键词字段（轻微多一次IO，但逻辑最清晰）
    if args.only_missing:
        filtered = []
        ids = [a["_id"] for a in articles if "_id" in a]
        # 分批读取，避免一次性太大
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            batch_ids = ids[i:i+BATCH]
            cur = dao.collection.find(
                {"_id": {"$in": batch_ids}},
                {"tfidf_keywords": 1, "textrank_keywords": 1, "final_keywords": 1}
            )
            kws_map = {d["_id"]: d for d in cur}
            for a in articles[i:i+BATCH]:
                d = kws_map.get(a["_id"], {})
                # 把现有字段合并到 a 上用于判断
                a.update({
                    "tfidf_keywords": d.get("tfidf_keywords"),
                    "textrank_keywords": d.get("textrank_keywords"),
                    "final_keywords": d.get("final_keywords"),
                })
                if _need_update(a):
                    filtered.append(a)
        articles = filtered
        log.info("Only-missing mode: %d articles need update.", len(articles))

    # 2) 训练/加载 TF-IDF（用当前语料；你也可以另传更大的全局语料）
    corpus: List[str] = [a.get("content", "") for a in articles if a.get("content")]
    if not corpus:
        log.warning("No non-empty content found; nothing to do.")
        dao.close()
        return

    tfidf_model_path = ARTIFACTS.get("TFIDF_MODEL_PATH", "artifacts/tfidf_model.pkl")
    tfidf: TFIDFKeywordExtractor = load_fitted_extractor(
        corpus=corpus if args.force_retrain or tfidf_model_path is None else None,
        model_path=tfidf_model_path,
        max_features=1000,
        force_retrain=args.force_retrain,
        stop_words="english",
    )

    # 3) TextRank
    tr = TextRankKeywordExtractor()

    # 4) Final 融合器
    final_extractor = FinalKeywordExtractor(
        tfidf_extractor=tfidf,
        textrank_extractor=tr,
        alpha=args.alpha,
        normalize=args.normalize,
        method=args.method,
        rrf_k=float(FUSION.get("RRF_K", 60.0)),
    )

    # 5) 遍历写回
    updated = 0
    skipped = 0
    for doc in articles:
        _id = doc.get("_id")
        content = doc.get("content") or ""
        if not _id or not content.strip():
            skipped += 1
            continue

        try:
            # TF-IDF
            tfidf_kws = tfidf.transform(content, topk=args.topk)

            # TextRank（若 gensim 不可用，会在内部抛异常；这里兜底为空列表）
            try:
                tr_kws = tr.extract_keywords(content, topn=args.topk)
            except Exception as e:
                log.warning("TextRank failed for _id=%s: %s", _id, e)
                tr_kws = []

            # Final 融合（取最终关键词列表）
            try:
                fused_pairs = final_extractor.extract(content, top_k=args.topk, with_scores=True)
                final_kws = [kw for kw, *_ in fused_pairs]
            except Exception as e:
                # 融合失败则退回并集
                log.warning("Final fusion failed for _id=%s: %s; fallback to union", _id, e)
                final_kws = list(dict.fromkeys(tfidf_kws + tr_kws))

            dao.update_article_keywords(
                article_id=_id,
                tfidf_keywords=tfidf_kws,
                textrank_keywords=tr_kws,
                final_keywords=final_kws
            )
            updated += 1
            if updated % 50 == 0:
                log.info("Updated %d articles...", updated)

        except Exception as e:
            log.error("Update failed for _id=%s url=%s: %s", _id, doc.get("url"), e)

    log.info("Done. Updated=%d, Skipped=%d, Total=%d", updated, skipped, len(articles))
    dao.close()


if __name__ == "__main__":
    main()
