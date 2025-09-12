# src/recall/youtube_dnn/evaluate_youtube_dnn_recall.py

import os
import pandas as pd
from collections import defaultdict
import logging
import numpy as np
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

TOP_K = 50
TRAIN_WINDOW_DAYS = 30
CUTOFF_DAYS = 7


def load_ground_truth(behavior_csv, item_csv, window_days=CUTOFF_DAYS):
    df = pd.read_csv(behavior_csv)
    item_df = pd.read_csv(item_csv)
    valid_item_ids = set(item_df["item_id"].astype(str).unique())

    df["dt"] = build_datetime(df["year"], df["time_stamp"], df["timestamp"])
    max_dt = df["dt"].max()
    min_dt = max_dt - pd.Timedelta(days=window_days)

    logger.info(f"Evaluating with ground truth window: {min_dt.date()} ~ {max_dt.date()}")

    click_df = df[
        (df["dt"] >= min_dt) &
        (df["dt"] <= max_dt) &
        (df["action_type"].str.lower() == "click")
    ].copy()

    click_df["user_id"] = click_df["user_id"].astype(str)
    click_df["item_id"] = click_df["item_id"].astype(str)
    click_df = click_df[click_df["item_id"].isin(valid_item_ids)]

    gt_dict = defaultdict(set)
    for row in click_df.itertuples(index=False):
        gt_dict[row.user_id].add(row.item_id)

    logger.info(f"✅ Ground Truth 用户数: {len(gt_dict)}")
    logger.info(f"✅ Ground Truth 总点击数: {len(click_df)}")
    logger.info(f"📊 平均每个用户点击数: {len(click_df)/max(1,len(gt_dict)):.2f}")

    return gt_dict


def dcg_at_k(rel_scores, k):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_scores[:k]))

def evaluate_metrics(gt_dict, recall_df, top_k=TOP_K):
    recall_df = recall_df[recall_df["rank"] <= top_k]
    recall_df["user_id"] = recall_df["user_id"].astype(str)
    recall_df["item_id"] = recall_df["item_id"].astype(str)

    user_groups = recall_df.groupby("user_id")["item_id"].apply(list)

    recall_list, precision_list, hitrate_list, ndcg_list = [], [], [], []

    for user_id, rec_items in user_groups.items():
        gt_items = gt_dict.get(user_id, set())
        if not gt_items:
            continue

        hit_set = set(rec_items) & gt_items
        recall = len(hit_set) / len(gt_items)
        precision = len(hit_set) / len(rec_items)
        hitrate = 1.0 if hit_set else 0.0

        rel_scores = [1 if item in gt_items else 0 for item in rec_items]
        dcg = dcg_at_k(rel_scores, top_k)
        idcg = dcg_at_k(sorted(rel_scores, reverse=True), min(len(gt_items), top_k))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        recall_list.append(recall)
        precision_list.append(precision)
        hitrate_list.append(hitrate)
        ndcg_list.append(ndcg)

    return {
        f"Recall@{top_k}": np.mean(recall_list) if recall_list else 0.0,
        f"Precision@{top_k}": np.mean(precision_list) if precision_list else 0.0,
        f"HitRate@{top_k}": np.mean(hitrate_list) if hitrate_list else 0.0,
        f"NDCG@{top_k}": np.mean(ndcg_list) if ndcg_list else 0.0,
    }


def main():
    logger.info("🔍 Evaluating YouTube DNN Recall...")

    recall_df = pd.read_csv(RECALL_CSV)
    logger.info(f"✅ Recall 结果用户数: {recall_df['user_id'].nunique()} (总行数 {len(recall_df)})")

    gt_dict = load_ground_truth(BEHAVIOR_CSV, ITEM_CSV)

    recall_users = set(recall_df["user_id"].astype(str).unique())
    gt_users = set(gt_dict.keys())
    overlap_users = recall_users & gt_users
    logger.info(f"🔗 Ground Truth 用户数: {len(gt_users)}")
    logger.info(f"🔗 Recall 用户数: {len(recall_users)}")
    logger.info(f"🔗 两边交集用户数: {len(overlap_users)}")

    metrics = evaluate_metrics(gt_dict, recall_df)

    for k, v in metrics.items():
        logger.info(f"🎯 {k}: {v:.4f}")

    result_df = pd.DataFrame([
        {"metric": k, "value": round(v, 6)} for k, v in metrics.items()
    ])
    result_df.to_csv(EVAL_CSV, index=False)
    logger.info(f"📦 Evaluation results saved to {EVAL_CSV}")


if __name__ == "__main__":
    main()

