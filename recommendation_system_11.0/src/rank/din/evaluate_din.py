# src/rank/din/evaluate_din.py

import os
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from src.data.load_behavior_log import build_datetime

# ----------- Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
OUTPUT_DIR = "output/din"
DIN_RESULT_PATH = os.path.join(OUTPUT_DIR, "din_infer.csv")
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
EVAL_CSV = os.path.join(OUTPUT_DIR, "din_eval_metrics.csv")

TOP_K = 20
CUTOFF_DAYS = 7


def load_ground_truth(behavior_csv, recall_csv, window_days=CUTOFF_DAYS):
    """加载ground truth - 只使用召回结果中的用户和物品"""
    df = pd.read_csv(behavior_csv)
    recall_df = pd.read_csv(recall_csv)
    
    # 只使用召回结果中的用户和物品
    recall_users = set(recall_df["user_id"].astype(str).unique())
    recall_items = set(recall_df["item_id"].astype(str).unique())

    df["dt"] = build_datetime(df["year"], df["time_stamp"], df["timestamp"])
    max_dt = df["dt"].max()
    min_dt = max_dt - pd.Timedelta(days=window_days)

    logger.info(f"Evaluating with ground truth window: {min_dt.date()} ~ {max_dt.date()}")
    logger.info(f"Using recall users: {len(recall_users)}, recall items: {len(recall_items)}")

    click_df = df[
        (df["dt"] >= min_dt) &
        (df["dt"] <= max_dt) &
        (df["action_type"].str.lower() == "click")
    ].copy()

    click_df["user_id"] = click_df["user_id"].astype(str)
    click_df["item_id"] = click_df["item_id"].astype(str)
    click_df = click_df[
        (click_df["user_id"].isin(recall_users)) &
        (click_df["item_id"].isin(recall_items))
    ]

    gt_dict = defaultdict(set)
    for row in click_df.itertuples(index=False):
        gt_dict[row.user_id].add(row.item_id)

    logger.info(f"✅ Ground Truth 用户数: {len(gt_dict)}")
    logger.info(f"✅ Ground Truth 总点击数: {len(click_df)}")
    logger.info(f"📊 平均每个用户点击数: {len(click_df)/max(1,len(gt_dict)):.2f}")

    return gt_dict


def dcg_at_k(rel_scores, k):
    """计算DCG@K"""
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_scores[:k]))


def evaluate_metrics(gt_dict, result_df, top_k=TOP_K):
    """评估指标"""
    result_df = result_df[result_df["rank"] <= top_k]
    result_df["user_id"] = result_df["user_id"].astype(str)
    result_df["item_id"] = result_df["item_id"].astype(str)

    user_groups = result_df.groupby("user_id")["item_id"].apply(list)

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


def compare_with_meta_ranker():
    """与meta-ranker结果对比"""
    meta_ranker_path = "output/meta_ranker/meta_ranker_infer.json"
    
    if not os.path.exists(meta_ranker_path):
        logger.warning("Meta-ranker results not found, skipping comparison")
        return None
    
    try:
        import json
        with open(meta_ranker_path, "r") as f:
            meta_ranker_results = json.load(f)
        
        # 转换为DataFrame格式
        meta_ranker_df = pd.DataFrame(meta_ranker_results)
        meta_ranker_df["user_id"] = meta_ranker_df["user_id"].astype(str)
        meta_ranker_df["item_id"] = meta_ranker_df["item_id"].astype(str)
        
        return meta_ranker_df
    except Exception as e:
        logger.warning(f"Failed to load meta-ranker results: {e}")
        return None


def main():
    logger.info("🔍 Evaluating DIN Model...")

    # 加载DIN结果
    din_df = pd.read_csv(DIN_RESULT_PATH)
    logger.info(f"✅ DIN 结果用户数: {din_df['user_id'].nunique()} (总行数 {len(din_df)})")

    # 加载ground truth
    recall_csv = "output/fusion/fusion_recall.csv"
    gt_dict = load_ground_truth(BEHAVIOR_CSV, recall_csv)

    # 评估DIN
    din_metrics = evaluate_metrics(gt_dict, din_df)
    
    logger.info("🎯 DIN Model Results:")
    for k, v in din_metrics.items():
        logger.info(f"   {k}: {v:.4f}")

    # 与meta-ranker对比
    meta_ranker_df = compare_with_meta_ranker()
    if meta_ranker_df is not None:
        meta_ranker_metrics = evaluate_metrics(gt_dict, meta_ranker_df)
        
        logger.info("🎯 Meta-Ranker Results:")
        for k, v in meta_ranker_metrics.items():
            logger.info(f"   {k}: {v:.4f}")
        
        # 计算改进
        logger.info("📈 Improvement (DIN vs Meta-Ranker):")
        for k in din_metrics.keys():
            din_val = din_metrics[k]
            meta_val = meta_ranker_metrics[k]
            improvement = ((din_val - meta_val) / meta_val * 100) if meta_val > 0 else 0
            logger.info(f"   {k}: {improvement:+.2f}%")

    # 保存评估结果
    result_data = []
    for k, v in din_metrics.items():
        result_data.append({"metric": k, "value": round(v, 6), "model": "DIN"})
    
    if meta_ranker_df is not None:
        for k, v in meta_ranker_metrics.items():
            result_data.append({"metric": k, "value": round(v, 6), "model": "Meta-Ranker"})
    
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(EVAL_CSV, index=False)
    logger.info(f"📦 Evaluation results saved to {EVAL_CSV}")


if __name__ == "__main__":
    main()
