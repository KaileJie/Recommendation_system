import os
import pandas as pd
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


def evaluate_recall_precision(gt_dict, recall_df, top_k=TOP_K):
    """
    Compute Recall@K and Precision@K
    """
    recall_df = recall_df[recall_df["rank"] <= top_k]

    recall_df["user_id"] = recall_df["user_id"].astype(str)
    recall_df["item_id"] = recall_df["item_id"].astype(str)

    user_groups = recall_df.groupby("user_id")["item_id"].apply(list)

    recall_list, precision_list = [], []

    for user_id, rec_items in user_groups.items():
        gt_items = gt_dict.get(user_id, set())
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
    logger.info("🔍 Evaluating Item-CF Recall...")

    # 1. Load recall results
    recall_df = pd.read_csv(RECALL_CSV)
    logger.info(f"✅ Recall 结果用户数: {recall_df['user_id'].nunique()} (总行数 {len(recall_df)})")

    # 2. Build ground truth with filtering
    recall_users = set(recall_df["user_id"].astype(str).unique())
    gt_dict = load_ground_truth(BEHAVIOR_CSV, ITEM_CSV, valid_users=recall_users)

    logger.info(f"🔗 Recall 用户数: {len(recall_users)}, GT 用户数: {len(gt_dict)}, 交集用户数: {len(set(gt_dict) & recall_users)}")

    # 3. Evaluate
    recall_k, precision_k = evaluate_recall_precision(gt_dict, recall_df)

    logger.info(f"🎯 Recall@{TOP_K}: {recall_k:.4f}")
    logger.info(f"🎯 Precision@{TOP_K}: {precision_k:.4f}")

    # 4. Save metrics
    result_df = pd.DataFrame([{
        "metric": f"Recall@{TOP_K}",
        "value": round(recall_k, 6)
    }, {
        "metric": f"Precision@{TOP_K}",
        "value": round(precision_k, 6)
    }])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_df.to_csv(EVAL_CSV, index=False)
    logger.info(f"📦 Evaluation results saved to {EVAL_CSV}")


if __name__ == "__main__":
    main()
