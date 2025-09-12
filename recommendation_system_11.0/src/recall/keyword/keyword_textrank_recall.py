# src/recall/keyword/keyword_textrank_recall.py

import os
import logging
import pandas as pd
from collections import defaultdict
from gensim.summarization import keywords as textrank_keywords


from src.recall.keyword.dataset_keyword import load_all_for_keyword

# ----------- Logger Setup -----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ----------- Config -----------
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
OUTPUT_PATH = "output/keyword/keyword_textrank_recall.csv"
TRAIN_WINDOW_DAYS = 30   # 训练窗口
CUTOFF_DAYS = 7          # 留出最后7天做评估
TOP_K = 20

# ----------- Keyword Extraction -----------

def extract_keywords(text, ratio=0.2):
    try:
        return set(textrank_keywords(text, ratio=ratio, split=True, lemmatize=True))
    except ValueError:
        return set()

# ----------- Recall Logic -----------

def textrank_recall(user_text_dict, item_text_dict, top_k=10):
    logger.info("Extracting keywords for all items...")
    item_keyword_map = {iid: extract_keywords(text) for iid, text in item_text_dict.items()}

    logger.info("Extracting keywords for all users...")
    user_keyword_map = {uid: extract_keywords(text) for uid, text in user_text_dict.items()}

    results = []
    for user_id, user_keywords in user_keyword_map.items():
        item_scores = []
        for item_id, item_keywords in item_keyword_map.items():
            if not user_keywords or not item_keywords:
                continue
            overlap = user_keywords & item_keywords
            score = len(overlap) / len(item_keywords)
            if score > 0:
                item_scores.append((item_id, score))

        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)[:top_k]
        for rank, (item_id, score) in enumerate(item_scores):
            results.append((user_id, item_id, rank + 1, float(score), "textrank"))

    return results

# ----------- Save Function -----------

def save_recall_results(results, output_path):
    df = pd.DataFrame(results, columns=["user_id", "item_id", "rank", "score", "source"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"TextRank recall results saved to {output_path}, total rows: {len(df)}")

# ----------- Main -----------
if __name__ == "__main__":
    logger.info("🧠 Starting TextRank keyword recall...")
    behavior_df, item_text_dict, user_text_dict = load_all_for_keyword(
        behavior_csv=BEHAVIOR_CSV,
        item_csv=ITEM_CSV,
        train_window_days=TRAIN_WINDOW_DAYS,
        cutoff_days=CUTOFF_DAYS
    )

    results = textrank_recall(user_text_dict, item_text_dict, top_k=TOP_K)
    save_recall_results(results, OUTPUT_PATH)
    logger.info("✅ TextRank keyword recall finished.")
