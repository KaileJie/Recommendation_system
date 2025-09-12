# src/recall/keyword/keyword_hybrid_recall.py

import os
import logging
import pandas as pd

from src.recall.keyword.keyword_tfidf_recall import tfidf_recall
from src.recall.keyword.keyword_textrank_recall import textrank_recall
from src.recall.keyword.dataset_keyword import load_all_for_keyword

# ----------- Logger Setup -----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ----------- Config -----------
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
OUTPUT_PATH = "output/keyword/keyword_hybrid_recall.csv"
DAYS_WINDOW = 30
TOP_K = 10
ALPHA = 0.6  # weight for TF-IDF, (1 - ALPHA) for TextRank

# ----------- Hybrid Recall -----------

def hybrid_recall(user_text_dict, item_text_dict, top_k=10, alpha=0.6):
    logger.info("Running TF-IDF and TextRank recalls...")
    tfidf_results = tfidf_recall(user_text_dict, item_text_dict, top_k=100)
    textrank_results = textrank_recall(user_text_dict, item_text_dict, top_k=100)

    # Build lookup: user_id -> item_id -> score
    def build_score_map(results):
        score_map = {}
        for user_id, item_id, _, score, _ in results:
            score_map.setdefault(user_id, {})[item_id] = score
        return score_map

    tfidf_map = build_score_map(tfidf_results)
    textrank_map = build_score_map(textrank_results)

    all_users = set(tfidf_map.keys()) | set(textrank_map.keys())

    final_results = []
    for user_id in all_users:
        items = set(tfidf_map.get(user_id, {}).keys()) | set(textrank_map.get(user_id, {}).keys())
        item_scores = []
        for item_id in items:
            tfidf_score = tfidf_map.get(user_id, {}).get(item_id, 0.0)
            textrank_score = textrank_map.get(user_id, {}).get(item_id, 0.0)
            hybrid_score = alpha * tfidf_score + (1 - alpha) * textrank_score
            item_scores.append((item_id, hybrid_score))

        top_items = sorted(item_scores, key=lambda x: x[1], reverse=True)[:top_k]
        for rank, (item_id, score) in enumerate(top_items):
            final_results.append((user_id, item_id, rank + 1, float(score), "keyword_hybrid"))

    return final_results

# ----------- Save Function -----------

def save_recall_results(results, output_path):
    df = pd.DataFrame(results, columns=["user_id", "item_id", "rank", "score", "source"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Hybrid recall results saved to {output_path}, total rows: {len(df)}")

# ----------- Main -----------
if __name__ == "__main__":
    logger.info("🤝 Starting hybrid keyword recall...")
    behavior_df, item_text_dict, user_text_dict = load_all_for_keyword(
        behavior_csv=BEHAVIOR_CSV,
        item_csv=ITEM_CSV,
        days_window=DAYS_WINDOW
    )

    results = hybrid_recall(user_text_dict, item_text_dict, top_k=TOP_K, alpha=ALPHA)
    save_recall_results(results, OUTPUT_PATH)
    logger.info("✅ Hybrid keyword recall finished.")
