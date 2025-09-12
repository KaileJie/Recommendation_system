# src/rank/meta_ranker/build_training_data.py

import os
import pandas as pd
import json
import logging

# -------- Logger --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------- Config --------
OUTPUT_DIR = "output/meta_ranker"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 三个召回结果
ITEMCF_PATH = "output/item_cf/itemcf_recall.csv"
KEYWORD_PATH = "output/keyword/keyword_hybrid_recall.csv"
DNN_PATH = "output/youtube_dnn/youtube_dnn_faiss_recall.csv"

# 用户画像
USER_PROFILE_PATH = "data/processed/user_profile_feature.csv"

# 行为日志（做 ground truth）
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
TOP_K = 50
CUTOFF_DAYS = 7   # Ground truth 窗口
NEG_POS_RATIO = 5  # ⚠️ 每个正样本配多少负样本


def load_ground_truth(behavior_csv, cutoff_days=CUTOFF_DAYS):
    """构造最后7天的点击作为正样本"""
    from src.data.load_behavior_log import build_datetime

    df = pd.read_csv(behavior_csv)
    df["dt"] = build_datetime(df["year"], df["time_stamp"], df["timestamp"])
    max_dt = df["dt"].max()
    min_dt = max_dt - pd.Timedelta(days=cutoff_days)

    df = df[(df["dt"] >= min_dt) & (df["dt"] <= max_dt)]
    df = df[df["action_type"].str.lower() == "click"]

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    gt = set(zip(df["user_id"], df["item_id"]))
    logger.info(f"✅ Ground truth 样本数: {len(gt)}")
    return gt


def load_recall_results():
    """合并三路召回结果"""
    dfs = []
    for path, source in [(ITEMCF_PATH, "itemcf"),
                         (KEYWORD_PATH, "keyword"),
                         (DNN_PATH, "youtube_dnn")]:
        if not os.path.exists(path):
            logger.warning(f"⚠️ Recall file missing: {path}")
            continue
        df = pd.read_csv(path)
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df = df[["user_id", "item_id", "score"]].rename(columns={"score": f"score_{source}"})
        dfs.append(df)

    # 外连接，保证没命中的地方是 NaN → 0
    from functools import reduce
    merged = reduce(lambda left, right: pd.merge(left, right, on=["user_id", "item_id"], how="outer"), dfs)
    merged = merged.fillna(0.0)
    return merged


def build_training_samples():
    """构造训练样本"""
    gt = load_ground_truth(BEHAVIOR_CSV)
    recall_df = load_recall_results()

    # 加入 label
    recall_df["label"] = recall_df.apply(
        lambda row: 1 if (row["user_id"], row["item_id"]) in gt else 0,
        axis=1
    )

    # ⚠️ 做负采样
    positives = recall_df[recall_df["label"] == 1]
    negatives = recall_df[recall_df["label"] == 0]

    neg_sample_size = min(len(negatives), len(positives) * NEG_POS_RATIO)
    negatives_sampled = negatives.sample(n=neg_sample_size, random_state=42)

    balanced = pd.concat([positives, negatives_sampled], ignore_index=True)
    logger.info(f"✅ Balanced dataset: {len(positives)} positives, {len(negatives_sampled)} negatives")


    # 加入用户画像
    profile = pd.read_csv(USER_PROFILE_PATH)
    profile["user_id"] = profile["user_id"].astype(str)

    # ⚠️ 新增：把 city 转换成类别 ID（整数）
    if "city" in profile.columns:
        profile["city"] = profile["city"].astype("category").cat.codes

    keep_cols = ["user_id", "age_range", "gender", "city", "cluster_id",
                 "recency", "frequency", "actions_per_active_day_30d",
                 "monetary", "rfm_score"]

    profile = profile[keep_cols]
    merged = balanced.merge(profile, on="user_id", how="left")

    # 🔍 统计正负样本比例
    pos_count = (merged["label"] == 1).sum()
    neg_count = (merged["label"] == 0).sum()
    logger.info(f"📊 Label distribution after sampling: "
                f"positives={pos_count}, negatives={neg_count}, "
                f"ratio={pos_count/(neg_count+1e-8):.4f}")
    logger.info(f"✅ Final training samples: {merged.shape}")

    # 转换为 JSON list 格式
    out_path = os.path.join(OUTPUT_DIR, "training_data.json")
    with open(out_path, "w") as f:
        for row in merged.itertuples(index=False):
            record = {
                "user_id": row.user_id,
                "item_id": row.item_id,
                "label": row.label,
                "scores": {
                    "itemcf": row.score_itemcf,
                    "keyword": row.score_keyword,
                    "youtube_dnn": row.score_youtube_dnn
                },
                "features": {
                    "age_range": row.age_range,
                    "gender": row.gender,
                    "city": row.city,
                    "cluster_id": row.cluster_id,
                    "recency": row.recency,
                    "frequency": row.frequency,
                    "actions_per_active_day_30d": row.actions_per_active_day_30d,
                    "monetary": row.monetary,
                    "rfm_score": row.rfm_score
                }
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"📦 Training data saved to {out_path}")


if __name__ == "__main__":
    build_training_samples()
