# src/recall/keyword/evaluate_keyword_recall.py

import os
import logging
import pandas as pd
from collections import defaultdict

# ----------- Logger Setup -----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ----------- Config -----------
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
RECALL_DIR = "output/keyword/"
OUTPUT_CSV = "output/keyword/keyword_recall_evaluation.csv"
DAYS_WINDOW = 30
TOP_K = 10

# ----------- Helper Functions -----------

def load_ground_truth(behavior_csv, days_window=30, top_actions=('click', 'read', 'like', 'fav')) -> dict:
    """
    Load real user behavior logs to build ground truth.
    Return: dict {user_id: set(item_ids)}
    """
    from src.data.load_behavior_log import find_window_bounds, iter_clean_behavior

    logger.info("📦 Loading ground truth user-item pairs...")
    _, cutoff, upper = find_window_bounds(behavior_csv, chunksize=100_000, days_window=days_window)

    user_clicks = defaultdict(set)

    for chunk in iter_clean_behavior(
        input_behavior_csv=behavior_csv,
        chunksize=100_000,
        cutoff_date=cutoff,
        upper=upper,
        extra_usecols=['item_id'],
        valid_actions=set(top_actions)
    ):
        for _, row in chunk.iterrows():
            user_clicks[int(row['user_id'])].add(int(row['item_id']))

    logger.info(f"✅ Loaded ground truth for {len(user_clicks)} users.")
    return user_clicks

def load_recall_results(recall_path: str) -> pd.DataFrame:
    """
    Load a recall result CSV and ensure required columns and type consistency.
    """
    try:
        df = pd.read_csv(recall_path)
        if set(['user_id', 'item_id', 'rank', 'score', 'source']).issubset(df.columns):
            df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').fillna(-1).astype(int)
            df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce').fillna(-1).astype(int)
            logger.info(f"📄 Loaded recall results from: {recall_path}, shape={df.shape}")
            return df
        else:
            raise ValueError("Missing required columns in recall CSV.")
    except Exception as e:
        logger.error(f"❌ Failed to load recall file {recall_path}: {e}")
        return pd.DataFrame()

def evaluate_recall_precision(recall_df: pd.DataFrame, ground_truth: dict, k: int = 10) -> pd.DataFrame:
    """
    Evaluate Recall@K and Precision@K for a given recall result DataFrame.
    """
    results = []
    grouped = recall_df.groupby('user_id')
    for user_id, group in grouped:
        if user_id not in ground_truth:
            continue
        gt_items = ground_truth[user_id]
        if not gt_items:
            continue

        rec_items = group.sort_values('rank').head(k)['item_id'].tolist()

        hit_count = sum(1 for item in rec_items if item in gt_items)
        recall = hit_count / len(gt_items)
        precision = hit_count / k

        source = group['source'].iloc[0]
        results.append((user_id, recall, precision, source))

    return pd.DataFrame(results, columns=['user_id', 'recall', 'precision', 'source'])

# ----------- Main -----------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    ground_truth = load_ground_truth(BEHAVIOR_CSV, days_window=DAYS_WINDOW)

    final_results = []

    for fname in os.listdir(RECALL_DIR):
        if not fname.endswith('.csv'):
            continue
        recall_path = os.path.join(RECALL_DIR, fname)
        recall_df = load_recall_results(recall_path)
        if recall_df.empty:
            continue

        eval_df = evaluate_recall_precision(recall_df, ground_truth, k=TOP_K)
        if eval_df.empty:
            logger.warning(f"No valid evaluation for {fname}")
            continue

        mean_recall = eval_df['recall'].mean()
        mean_precision = eval_df['precision'].mean()
        source = eval_df['source'].iloc[0] if not eval_df.empty else fname

        logger.info(f"📊 {source}: Recall@{TOP_K} = {mean_recall:.4f}, Precision@{TOP_K} = {mean_precision:.4f}")

        final_results.append({
            'source': source,
            f'recall@{TOP_K}': round(mean_recall, 4),
            f'precision@{TOP_K}': round(mean_precision, 4)
        })

    pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False)
    logger.info(f"📈 Evaluation results saved to {OUTPUT_CSV}")