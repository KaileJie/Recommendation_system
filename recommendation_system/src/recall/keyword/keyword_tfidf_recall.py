# src/recall/keyword/keyword_tfidf_recall.py

import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recall.keyword.dataset_keyword import load_all_for_keyword

# ----------- Logger Setup -----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ----------- Config -----------
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
OUTPUT_PATH = "output/keyword/keyword_tfidf_recall.csv"
DAYS_WINDOW = 30
TOP_K = 10

# ----------- Recall Logic -----------

def tfidf_recall(user_text_dict, item_text_dict, top_k=10):
    """
    Perform TF-IDF recall using cosine similarity between user text and item text.
    Returns list of recall results in format: (user_id, item_id, score)
    """
    # 1. 构造文本列表和索引映射
    item_ids = list(item_text_dict.keys())
    item_texts = [item_text_dict[iid] for iid in item_ids]

    # 2. 构建 TF-IDF 模型
    vectorizer = TfidfVectorizer(max_features=10000)
    item_matrix = vectorizer.fit_transform(item_texts)

    top_k = min(top_k, len(item_ids))  # 🔒 TopK 边界保护

    results = []
    for user_id, user_text in user_text_dict.items():
        user_vec = vectorizer.transform([user_text])  # shape: (1, vocab)
        scores = cosine_similarity(user_vec, item_matrix).flatten()  # shape: (n_items,)

        top_indices = scores.argsort()[::-1][:top_k]
        for rank, idx in enumerate(top_indices):
            item_id = item_ids[idx]
            score = float(scores[idx])
            results.append((user_id, item_id, rank + 1, score, "tfidf"))

    return results

# ----------- Save Function -----------

def save_recall_results(results, output_path):
    df = pd.DataFrame(results, columns=["user_id", "item_id", "rank", "score", "source"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"TF-IDF recall results saved to {output_path}, total rows: {len(df)}")

# ----------- Main -----------
if __name__ == "__main__":
    logger.info("🔍 Starting TF-IDF keyword recall...")
    behavior_df, item_text_dict, user_text_dict = load_all_for_keyword(
        behavior_csv=BEHAVIOR_CSV,
        item_csv=ITEM_CSV,
        days_window=DAYS_WINDOW
    )

    results = tfidf_recall(user_text_dict, item_text_dict, top_k=TOP_K)
    save_recall_results(results, OUTPUT_PATH)
    logger.info("✅ TF-IDF keyword recall finished.")
