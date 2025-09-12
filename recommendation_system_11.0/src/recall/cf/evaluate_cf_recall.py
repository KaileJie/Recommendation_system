import os
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from src.data.load_behavior_log import build_datetime

# ----------- Logging Setup -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
OUTPUT_DIR = "output/item_cf"
RECALL_CSV = os.path.join(OUTPUT_DIR, "itemcf_recall.csv")
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
EVAL_CSV = os.path.join(OUTPUT_DIR, "itemcf_eval_metrics.csv")

TOP_K = 20
CUTOFF_DAYS = 7   # ⚠️ Ground Truth 窗口 = 最后7天


def load_ground_truth(behavior_csv, item_csv, valid_users=None, cutoff_days=CUTOFF_DAYS):
    """
    Load ground truth from behavior logs.
    - 用户限制在 valid_users
    - item_id 只保留出现在 item_metadata.csv 的
    Return: dict {user_id: set(item_ids)}
    """
    df = pd.read_csv(behavior_csv)
    item_df = pd.read_csv(item_csv)
    valid_item_ids = set(item_df["item_id"].astype(str).unique())

    # 构建 datetime
    df["dt"] = build_datetime(df["year"], df["time_stamp"], df["timestamp"])
    max_dt = df["dt"].max()
    min_dt = max_dt - pd.Timedelta(days=cutoff_days)

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
        uid, iid = row.user_id, row.item_id
        if valid_users is not None and uid not in valid_users:
            continue
        gt_dict[uid].add(iid)

    logger.info(f"✅ Ground Truth 用户数: {len(gt_dict)}, 总点击数: {len(click_df)}")
    return gt_dict


def dcg_at_k(rel_scores, k):
    """计算DCG@K"""
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_scores[:k]))

def evaluate_metrics(gt_dict, recall_df, top_k=TOP_K):
    """
    计算多种评估指标：Recall@K, Precision@K, HitRate@K, NDCG@K
    """
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
        
        # Recall@K: 推荐中命中的真实物品 / 真实物品总数
        recall = len(hit_set) / len(gt_items)
        
        # Precision@K: 推荐中命中的真实物品 / 推荐物品总数
        precision = len(hit_set) / len(rec_items)
        
        # HitRate@K: 是否至少命中一个真实物品 (0或1)
        hitrate = 1.0 if hit_set else 0.0
        
        # NDCG@K: 归一化折扣累积增益
        rel_scores = [1 if item in gt_items else 0 for item in rec_items]
        dcg = dcg_at_k(rel_scores, top_k)
        idcg = dcg_at_k(sorted(rel_scores, reverse=True), min(len(gt_items), top_k))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        recall_list.append(recall)
        precision_list.append(precision)
        hitrate_list.append(hitrate)
        ndcg_list.append(ndcg)

    return {
        f"Recall@{top_k}": sum(recall_list) / len(recall_list) if recall_list else 0.0,
        f"Precision@{top_k}": sum(precision_list) / len(precision_list) if precision_list else 0.0,
        f"HitRate@{top_k}": sum(hitrate_list) / len(hitrate_list) if hitrate_list else 0.0,
        f"NDCG@{top_k}": sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0,
    }


def main():
    logger.info("🔍 Evaluating Item-CF Recall...")

    # 1. Load recall results
    recall_df = pd.read_csv(RECALL_CSV)
    logger.info(f"✅ Recall 结果用户数: {recall_df['user_id'].nunique()} (总行数 {len(recall_df)})")

    # 2. Build ground truth with filtering
    recall_users = set(recall_df["user_id"].astype(str).unique())
    gt_dict = load_ground_truth(BEHAVIOR_CSV, ITEM_CSV, valid_users=recall_users)

    logger.info(f"🔗 Recall 用户数: {len(recall_users)}, GT 用户数: {len(gt_dict)}, 交集用户数: {len(set(gt_dict) & recall_users)}")

    # 3. Evaluate with multiple metrics for Top_K=20
    logger.info(f"📊 Evaluating metrics for K={TOP_K}...")
    metrics = evaluate_metrics(gt_dict, recall_df, top_k=TOP_K)

    # 4. Log results
    for metric_name, metric_value in metrics.items():
        logger.info(f"🎯 {metric_name}: {metric_value:.4f}")

    # 5. Save metrics
    result_df = pd.DataFrame([
        {"metric": metric_name, "value": round(metric_value, 6)} 
        for metric_name, metric_value in metrics.items()
    ])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_df.to_csv(EVAL_CSV, index=False)
    logger.info(f"📦 Evaluation results saved to {EVAL_CSV}")


if __name__ == "__main__":
    main()
