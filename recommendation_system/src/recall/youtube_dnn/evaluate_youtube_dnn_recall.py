# src/recall/youtube_dnn/evaluate_youtube_dnn_recall.py

import os
import pandas as pd
from collections import defaultdict
import logging
from src.data.load_behavior_log import build_datetime

# ----------- Logging Setup -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
OUTPUT_DIR = "output/youtube_dnn"
RECALL_CSV = os.path.join(OUTPUT_DIR, "youtube_dnn_faiss_recall.csv")
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
EVAL_CSV = os.path.join(OUTPUT_DIR, "youtube_dnn_eval_metrics.csv")

TOP_K = 10
GROUND_TRUTH_WINDOW_DAYS = 7


def load_ground_truth(behavior_csv, item_csv, top_k=TOP_K, window_days=GROUND_TRUTH_WINDOW_DAYS):
    """
    Load real clicked items for each user within the last `window_days`,
    filtered by item_metadata.
    """
    df = pd.read_csv(behavior_csv)
    item_df = pd.read_csv(item_csv)
    valid_item_ids = set(item_df["item_id"].astype(str).unique())

    # 手动构建 dt 列（从 year, time_stamp, timestamp）
    df['dt'] = build_datetime(df['year'], df['time_stamp'], df['timestamp'])
    max_dt = df["dt"].max()
    min_dt = max_dt - pd.Timedelta(days=window_days)

    logger.info(f"Evaluating with ground truth window: {min_dt.date()} ~ {max_dt.date()}")

    click_df = df[
        (df["dt"] >= min_dt) &
        (df["dt"] <= max_dt) &
        (df["action_type"].str.lower() == "click")
    ].copy()

    # ✅ 只保留出现在 item_metadata 中的点击行为
    click_df["item_id"] = click_df["item_id"].astype(str)
    click_df = click_df[click_df["item_id"].isin(valid_item_ids)]

    gt_dict = defaultdict(set)
    for row in click_df.itertuples(index=False):
        gt_dict[str(row.user_id)].add(str(row.item_id))

    logger.info(f"✅ Ground Truth 总点击数: {len(click_df)}")
    logger.info(f"📦 其中出现在 item_metadata 的 item 数: {sum(len(v) for v in gt_dict.values())}")
    return gt_dict


def evaluate_recall_precision(gt_dict, recall_df, top_k=TOP_K):
    """
    Compute Recall@K and Precision@K
    """
    recall_df = recall_df[recall_df["rank"] <= top_k]
    user_groups = recall_df.groupby("user_id")["item_id"].apply(list)

    recall_list = []
    precision_list = []

    for user_id, rec_items in user_groups.items():
        gt_items = gt_dict.get(str(user_id), set())
        if not gt_items:
            continue

        hit_set = set(rec_items) & gt_items
        recall = len(hit_set) / len(gt_items)
        precision = len(hit_set) / len(rec_items)

        recall_list.append(recall)
        precision_list.append(precision)

    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0

    return avg_recall, avg_precision


def main():
    logger.info("🔍 Evaluating YouTube DNN Recall...")

    # 1. Load recall result
    recall_df = pd.read_csv(RECALL_CSV)
    logger.info(f"✅ Loaded recall results: {len(recall_df)} rows")

    # 2. Load ground truth
    gt_dict = load_ground_truth(BEHAVIOR_CSV, ITEM_CSV)

    # 3. Evaluate
    recall_k, precision_k = evaluate_recall_precision(gt_dict, recall_df)

    logger.info(f"🎯 Recall@{TOP_K}: {recall_k:.4f}")
    logger.info(f"🎯 Precision@{TOP_K}: {precision_k:.4f}")

    # 4. Save metrics to CSV
    result_df = pd.DataFrame([{
        "metric": f"Recall@{TOP_K}",
        "value": round(recall_k, 6)
    }, {
        "metric": f"Precision@{TOP_K}",
        "value": round(precision_k, 6)
    }])

    result_df.to_csv(EVAL_CSV, index=False)
    logger.info(f"📦 Evaluation results saved to {EVAL_CSV}")


if __name__ == "__main__":
    main()
