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
TRAIN_DAYS = 14   # 训练召回结果窗口（在label窗口之前）
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


def load_recall_results_with_time_split():
    """加载召回结果，确保时间窗口不重叠"""
    from src.data.load_behavior_log import build_datetime
    
    # 读取行为日志获取时间信息
    behavior_df = pd.read_csv(BEHAVIOR_CSV)
    behavior_df["dt"] = build_datetime(behavior_df["year"], behavior_df["time_stamp"], behavior_df["timestamp"])
    max_dt = behavior_df["dt"].max()
    
    # 计算召回结果的时间窗口（在label窗口之前）
    recall_end_dt = max_dt - pd.Timedelta(days=CUTOFF_DAYS)
    recall_start_dt = recall_end_dt - pd.Timedelta(days=TRAIN_DAYS)
    
    logger.info(f"📅 Recall window: {recall_start_dt} to {recall_end_dt}")
    logger.info(f"📅 Label window: {max_dt - pd.Timedelta(days=CUTOFF_DAYS)} to {max_dt}")
    
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


def normalize_scores(df):
    """对召回分数进行归一化处理"""
    score_cols = [col for col in df.columns if col.startswith('score_')]
    
    for col in score_cols:
        if df[col].max() > df[col].min():  # 避免除零
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[col] = 0.0  # 如果所有值相同，设为0
    
    logger.info(f"✅ Normalized score columns: {score_cols}")
    return df


def handle_missing_features(df):
    """处理特征缺失值 - 参考 load_user_info.py 的处理方式"""
    # 数值特征：根据业务逻辑选择合适的填充策略
    numerical_cols = ['age_range', 'recency', 'frequency', 'actions_per_active_day_30d', 
                     'monetary', 'rfm_score']
    
    for col in numerical_cols:
        if col in df.columns:
            if df[col].isna().any():
                if col == 'age_range':
                    # age_range: 缺失用0填充，表示未知年龄（参考 load_user_info.py）
                    df[col] = df[col].fillna(0)
                    logger.info(f"✅ Filled missing age_range with 0 (unknown age)")
                elif col == 'recency':
                    # recency: 缺失表示从未活跃，用比现有最大值更大的值填充
                    # 这样在RFM分析中会被归类为最不活跃的用户
                    max_recency = df[col].max()
                    if pd.isna(max_recency):
                        fill_value = 31.0  # 默认值
                    else:
                        fill_value = max_recency + 1.0  # 比最大值大1
                    df[col] = df[col].fillna(fill_value)
                    logger.info(f"✅ Filled missing recency with {fill_value} (never active, worse than max={max_recency})")
                elif col in ['frequency', 'monetary', 'rfm_score']:
                    # 这些特征缺失表示没有相关行为，用0填充
                    df[col] = df[col].fillna(0.0)
                    logger.info(f"✅ Filled missing {col} with 0.0 (no activity)")
                else:
                    # 其他数值特征用均值填充
                    df[col] = df[col].fillna(df[col].mean())
                    logger.info(f"✅ Filled missing {col} with mean value")
    
    # 类别特征：参考 load_user_info.py 的处理方式
    categorical_cols = ['gender', 'city', 'cluster_id']
    
    for col in categorical_cols:
        if col in df.columns:
            if df[col].isna().any():
                if col == 'gender':
                    # gender: 缺失用2填充，表示未知性别（参考 load_user_info.py）
                    df[col] = df[col].fillna(2)
                    logger.info(f"✅ Filled missing gender with 2 (unknown gender)")
                elif col == 'cluster_id':
                    # cluster_id: 缺失用-1填充，表示未知聚类
                    df[col] = df[col].fillna(-1)
                    logger.info(f"✅ Filled missing cluster_id with -1 (unknown cluster)")
                else:
                    # city: 缺失用'unknown'填充
                    df[col] = df[col].fillna('unknown')
                    logger.info(f"✅ Filled missing {col} with 'unknown'")
    
    return df


def build_training_samples():
    """构造训练样本"""
    gt = load_ground_truth(BEHAVIOR_CSV)
    recall_df = load_recall_results_with_time_split()  # 使用时间分割版本

    # 对召回分数进行归一化
    recall_df = normalize_scores(recall_df)

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
    
    # 处理特征缺失值
    merged = handle_missing_features(merged)

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
