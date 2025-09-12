# src/rank/din/infer_din.py

import os
import json
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

from src.rank.din.din_model import DINModel
from src.rank.din.dataset_din import collate_fn
from src.data.load_behavior_log import build_datetime

# ----------- Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
OUTPUT_DIR = "output/din"
MODEL_PATH = os.path.join(OUTPUT_DIR, "din_model.pt")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "din_config.json")
INFER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "din_infer.json")

# 数据路径
RECALL_DATA_PATH = "output/fusion/fusion_recall.csv"
USER_PROFILE_PATH = "data/processed/user_profile_feature.csv"
ITEM_METADATA_PATH = "data/raw/item_metadata.csv"
BEHAVIOR_LOG_PATH = "data/raw/user_behavior_log_info.csv"

TOP_K = 20


class DINInferenceDataset(Dataset):
    """DIN 推理数据集 - 只包含召回结果中的用户-物品对"""
    
    def __init__(self, 
                 recall_df,
                 user_profile_df,
                 item_metadata_df,
                 user_sequences,
                 user_cate_dims,
                 item_cate_dims,
                 user_numeric_cols,
                 item_numeric_cols,
                 user2idx,
                 item2idx,
                 max_seq_len=50):
        
        self.recall_df = recall_df
        self.user_profile_df = user_profile_df
        self.item_metadata_df = item_metadata_df
        self.user_sequences = user_sequences
        self.user_cate_dims = user_cate_dims
        self.item_cate_dims = item_cate_dims
        self.user_numeric_cols = user_numeric_cols
        self.item_numeric_cols = item_numeric_cols
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.max_seq_len = max_seq_len
        
        # 创建样本列表
        self.samples = []
        for _, row in recall_df.iterrows():
            user_id = str(row["user_id"])
            item_id = str(row["item_id"])
            
            if user_id in self.user2idx and item_id in self.item2idx:
                self.samples.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "cf_score": row.get("cf_score", 0.0),
                    "keyword_score": row.get("keyword_score", 0.0),
                    "dnn_score": row.get("dnn_score", 0.0)
                })
        
        logger.info(f"✅ Created inference dataset with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample["user_id"]
        item_id = sample["item_id"]
        
        # 获取用户特征
        user_profile = self.user_profile_df[
            self.user_profile_df["user_id"] == user_id
        ].iloc[0]
        
        user_cate_feats = {
            "gender": int(user_profile["gender"]),
            "age_range": int(user_profile["age_range"]),
            "city": int(user_profile["city"]),
            "cluster_id": int(user_profile["cluster_id"])
        }
        
        user_numeric_feats = [
            user_profile[col] if pd.notna(user_profile[col]) else 0.0
            for col in self.user_numeric_cols
        ]
        
        # 获取物品特征
        item_metadata = self.item_metadata_df[
            self.item_metadata_df["item_id"] == item_id
        ].iloc[0]
        
        # 物品类别特征 
        item_cate_feats = {}
        
        item_numeric_feats = [
            item_metadata[col] if pd.notna(item_metadata[col]) else 0.0
            for col in self.item_numeric_cols
        ]
        
        # 召回分数
        recall_scores = [sample["cf_score"], sample["keyword_score"], sample["dnn_score"]]
        
        # 获取用户历史序列
        if user_id in self.user_sequences:
            seq_items = self.user_sequences[user_id]
            # 填充到固定长度
            while len(seq_items) < self.max_seq_len:
                seq_items = [0] + seq_items
            
            seq_items = seq_items[-self.max_seq_len:]
            
            # 获取序列中物品的特征
            seq_cate_feats = {}
            seq_mask = []
            
            for seq_item_id in seq_items:
                if seq_item_id == 0:  # 填充位置
                    seq_mask.append(0)
                else:
                    if seq_item_id in self.item2idx:
                        seq_mask.append(1)
                    else:
                        seq_mask.append(0)
        else:
            # 没有历史序列
            seq_cate_feats = {}
            seq_mask = [0] * self.max_seq_len
        
        return {
            "user_id": user_id,
            "item_id": item_id,
            "user_cate_feats": user_cate_feats,
            "user_numeric_feats": torch.tensor(user_numeric_feats, dtype=torch.float32),
            "item_cate_feats": item_cate_feats,
            "item_numeric_feats": torch.tensor(item_numeric_feats, dtype=torch.float32),
            "seq_cate_feats": {k: torch.tensor(v, dtype=torch.long) for k, v in seq_cate_feats.items()},
            "recall_scores": torch.tensor(recall_scores, dtype=torch.float32),
            "seq_mask": torch.tensor(seq_mask, dtype=torch.long)
        }


def load_model_and_config():
    """加载模型和配置"""
    # 加载配置
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    # 创建模型
    model = DINModel(**config)
    
    # 加载权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    
    logger.info("✅ Model loaded successfully")
    return model, config


def build_user_sequences(behavior_df, max_dt, train_days=14):
    """构建用户行为序列"""
    train_start_dt = max_dt - pd.Timedelta(days=train_days)
    train_end_dt = max_dt
    
    train_behavior = behavior_df[
        (behavior_df["dt"] >= train_start_dt) &
        (behavior_df["dt"] <= train_end_dt) &
        (behavior_df["action_type"].str.lower() == "click")
    ].copy()
    
    train_behavior = train_behavior.sort_values(["user_id", "dt"])
    
    user_sequences = {}
    for user_id, group in train_behavior.groupby("user_id"):
        items = group["item_id"].tolist()
        if items:
            user_sequences[user_id] = items[-50:]  # 保留最近50个
    
    return user_sequences


def main():
    logger.info("🚀 Running DIN Inference...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载模型
    model, config = load_model_and_config()
    model = model.to(device)
    
    # 加载数据
    logger.info("Loading data...")
    recall_df = pd.read_csv(RECALL_DATA_PATH)
    recall_df["user_id"] = recall_df["user_id"].astype(str)
    recall_df["item_id"] = recall_df["item_id"].astype(str)
    
    user_profile_df = pd.read_csv(USER_PROFILE_PATH)
    user_profile_df["user_id"] = user_profile_df["user_id"].astype(str)
    
    item_metadata_df = pd.read_csv(ITEM_METADATA_PATH)
    item_metadata_df["item_id"] = item_metadata_df["item_id"].astype(str)
    
    behavior_df = pd.read_csv(BEHAVIOR_LOG_PATH)
    behavior_df["dt"] = build_datetime(behavior_df["year"], behavior_df["time_stamp"], behavior_df["timestamp"])
    behavior_df["user_id"] = behavior_df["user_id"].astype(str)
    behavior_df["item_id"] = behavior_df["item_id"].astype(str)
    
    # 构建用户序列
    max_dt = behavior_df["dt"].max()
    user_sequences = build_user_sequences(behavior_df, max_dt)
    
    # 创建ID映射
    user2idx = {uid: idx for idx, uid in enumerate(user_profile_df["user_id"].unique())}
    item2idx = {iid: idx for idx, iid in enumerate(item_metadata_df["item_id"].unique())}
    
    # 创建推理数据集
    inference_dataset = DINInferenceDataset(
        recall_df=recall_df,
        user_profile_df=user_profile_df,
        item_metadata_df=item_metadata_df,
        user_sequences=user_sequences,
        user_cate_dims=config["user_cate_dims"],
        item_cate_dims=config["item_cate_dims"],
        user_numeric_cols=["recency", "frequency", "monetary", "rfm_score",
                          "actions_per_active_day_30d", "morning_ratio_30d",
                          "afternoon_ratio_30d", "evening_ratio_30d", "late_night_ratio_30d"],
        item_numeric_cols=["price", "rating", "review_count"],
        user2idx=user2idx,
        item2idx=item2idx,
        max_seq_len=config["max_seq_len"]
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        inference_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 推理
    logger.info("Running inference...")
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            for key in batch:
                if isinstance(batch[key], dict):
                    batch[key] = {k: v.to(device) for k, v in batch[key].items()}
                else:
                    batch[key] = batch[key].to(device)
            
            # 前向传播
            outputs, attention_weights = model(
                user_cate_feats=batch["user_cate_feats"],
                user_numeric_feats=batch["user_numeric_feats"],
                item_cate_feats=batch["item_cate_feats"],
                item_numeric_feats=batch["item_numeric_feats"],
                seq_cate_feats=batch["seq_cate_feats"],
                recall_scores=batch["recall_scores"],
                seq_mask=batch["seq_mask"]
            )
            
            # 计算概率
            probs = torch.sigmoid(outputs)
            
            # 收集结果
            for i in range(len(batch["user_id"])):
                all_predictions.append({
                    "user_id": batch["user_id"][i],
                    "item_id": batch["item_id"][i],
                    "din_score": probs[i].item(),
                    "cf_score": batch["recall_scores"][i][0].item(),
                    "keyword_score": batch["recall_scores"][i][1].item(),
                    "dnn_score": batch["recall_scores"][i][2].item()
                })
    
    # 按用户分组并排序
    logger.info("Ranking items for each user...")
    user_rankings = defaultdict(list)
    
    for pred in all_predictions:
        user_rankings[pred["user_id"]].append(pred)
    
    # 为每个用户排序并取Top-K
    final_results = []
    for user_id, items in user_rankings.items():
        # 按DIN分数排序
        sorted_items = sorted(items, key=lambda x: x["din_score"], reverse=True)
        
        for rank, item in enumerate(sorted_items[:TOP_K], 1):
            final_results.append({
                "user_id": user_id,
                "item_id": item["item_id"],
                "rank": rank,
                "din_score": item["din_score"],
                "cf_score": item["cf_score"],
                "keyword_score": item["keyword_score"],
                "dnn_score": item["dnn_score"]
            })
    
    # 保存结果
    result_df = pd.DataFrame(final_results)
    result_df.to_csv(INFER_OUTPUT_PATH.replace('.json', '.csv'), index=False)
    
    # 保存JSON格式
    with open(INFER_OUTPUT_PATH, "w") as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"✅ Inference completed!")
    logger.info(f"📊 Total users: {len(user_rankings)}")
    logger.info(f"📊 Total recommendations: {len(final_results)}")
    logger.info(f"📦 Results saved to: {INFER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
