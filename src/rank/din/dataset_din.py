# src/rank/din/dataset_din_corrected.py - 修正版本的数据集

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from collections import defaultdict
from src.data.load_behavior_log import build_datetime

logger = logging.getLogger(__name__)


class DINDataset(Dataset):
    """DIN 模型数据集 - 修正版本"""
    
    def __init__(self, 
                 recall_data_path,
                 user_profile_path,
                 item_metadata_path,
                 behavior_log_path,
                 max_seq_len=20,
                 cutoff_days=7,
                 train_days=30,  # 修正：与召回模型保持一致
                 neg_pos_ratio=2,
                 max_samples=10000,
                 max_users=1000):
        """
        Args:
            recall_data_path: 召回结果文件路径
            user_profile_path: 用户画像文件路径
            item_metadata_path: 物品元数据文件路径
            behavior_log_path: 行为日志文件路径
            max_seq_len: 最大序列长度
            cutoff_days: 标签窗口天数
            train_days: 训练数据窗口天数 (修正为30天)
            neg_pos_ratio: 负样本比例
            max_samples: 最大样本数
            max_users: 最大用户数
        """
        self.max_seq_len = max_seq_len
        self.cutoff_days = cutoff_days
        self.train_days = train_days
        self.neg_pos_ratio = neg_pos_ratio
        self.max_samples = max_samples
        self.max_users = max_users
        
        # 加载数据
        self._load_data_corrected(recall_data_path, user_profile_path, item_metadata_path, behavior_log_path)
        
        # 构建特征映射
        self._build_feature_mappings()
        
        # 构建训练样本
        self._build_samples()
        
    def _load_data_corrected(self, recall_data_path, user_profile_path, item_metadata_path, behavior_log_path):
        """修正版本的数据加载策略"""
        logger.info("Loading data with corrected strategy...")
        
        # 1. 加载召回数据
        logger.info("Step 1: Loading recall data...")
        self.recall_df = pd.read_csv(recall_data_path)
        self.recall_df["user_id"] = self.recall_df["user_id"].astype(str)
        self.recall_df["item_id"] = self.recall_df["item_id"].astype(str)
        
        # 采样用户
        unique_users = self.recall_df["user_id"].unique()
        if len(unique_users) > self.max_users:
            sampled_users = np.random.choice(unique_users, size=self.max_users, replace=False)
            self.recall_df = self.recall_df[self.recall_df["user_id"].isin(sampled_users)]
        
        logger.info(f"✅ Loaded recall data: {self.recall_df.shape}")
        
        # 2. 加载用户画像 - 关键修正：使用召回数据中的用户ID
        logger.info("Step 2: Loading user profile...")
        user_profile_df = pd.read_csv(user_profile_path)
        user_profile_df["user_id"] = user_profile_df["user_id"].astype(str)
        
        # 由于融合召回将用户ID重新映射为1,2,3...，我们需要找到原始用户ID
        # 从召回数据中获取用户ID列表
        recall_users = set(self.recall_df["user_id"].unique())
        
        # 从用户画像中采样相同数量的用户
        available_users = user_profile_df["user_id"].unique()
        if len(available_users) >= len(recall_users):
            # 随机采样与召回数据相同数量的用户
            sampled_profile_users = np.random.choice(
                available_users, 
                size=min(len(recall_users), len(available_users)), 
                replace=False
            )
            self.user_profile_df = user_profile_df[user_profile_df["user_id"].isin(sampled_profile_users)]
        else:
            # 如果用户画像中的用户数不够，使用所有用户
            self.user_profile_df = user_profile_df
        
        # 创建用户ID映射：召回用户ID -> 用户画像用户ID
        recall_users_list = list(recall_users)
        profile_users_list = list(self.user_profile_df["user_id"].unique())
        
        self.user_id_mapping = {}
        for i, recall_user in enumerate(recall_users_list):
            if i < len(profile_users_list):
                self.user_id_mapping[recall_user] = profile_users_list[i]
        
        logger.info(f"✅ Created user ID mapping for {len(self.user_id_mapping)} users")
        logger.info(f"✅ Loaded user profile: {self.user_profile_df.shape}")
        
        # 3. 加载物品元数据
        logger.info("Step 3: Loading item metadata...")
        item_metadata_df = pd.read_csv(item_metadata_path)
        item_metadata_df["item_id"] = item_metadata_df["item_id"].astype(str)
        
        recall_items = set(self.recall_df["item_id"].unique())
        self.item_metadata_df = item_metadata_df[item_metadata_df["item_id"].isin(recall_items)]
        logger.info(f"✅ Loaded item metadata: {self.item_metadata_df.shape}")
        
        # 4. 加载行为日志 - 使用映射后的用户ID
        logger.info("Step 4: Loading behavior log...")
        
        behavior_chunks = []
        chunk_size = 100000
        mapped_behavior_users = set(self.user_id_mapping.values())
        
        for chunk in pd.read_csv(behavior_log_path, chunksize=chunk_size):
            chunk["user_id"] = chunk["user_id"].astype(str)
            chunk = chunk[chunk["user_id"].isin(mapped_behavior_users)]
            if not chunk.empty:
                behavior_chunks.append(chunk)
            if len(behavior_chunks) * chunk_size > self.max_samples * 10:
                break
        
        if behavior_chunks:
            self.behavior_df = pd.concat(behavior_chunks, ignore_index=True)
            self.behavior_df["dt"] = build_datetime(
                self.behavior_df["year"], 
                self.behavior_df["time_stamp"], 
                self.behavior_df["timestamp"]
            )
            self.behavior_df["item_id"] = self.behavior_df["item_id"].astype(str)
            # 只保留召回结果中的物品
            self.behavior_df = self.behavior_df[self.behavior_df["item_id"].isin(recall_items)]
        else:
            # 如果没有行为数据，创建空的DataFrame
            self.behavior_df = pd.DataFrame(columns=["user_id", "item_id", "action_type", "dt"])
        
        logger.info(f"✅ Loaded behavior log: {self.behavior_df.shape}")
        
    def _build_feature_mappings(self):
        """构建特征映射"""
        logger.info("Building feature mappings...")
        
        # 创建类别特征编码器
        from sklearn.preprocessing import LabelEncoder
        
        # 用户特征映射
        self.user_encoders = {}
        self.user_cate_dims = {}
        
        # 性别编码
        self.user_encoders["gender"] = LabelEncoder()
        self.user_profile_df["gender_encoded"] = self.user_encoders["gender"].fit_transform(self.user_profile_df["gender"])
        self.user_cate_dims["gender"] = len(self.user_encoders["gender"].classes_)
        
        # 年龄范围编码
        self.user_encoders["age_range"] = LabelEncoder()
        self.user_profile_df["age_range_encoded"] = self.user_encoders["age_range"].fit_transform(self.user_profile_df["age_range"])
        self.user_cate_dims["age_range"] = len(self.user_encoders["age_range"].classes_)
        
        # 城市编码
        self.user_encoders["city"] = LabelEncoder()
        self.user_profile_df["city_encoded"] = self.user_encoders["city"].fit_transform(self.user_profile_df["city"])
        self.user_cate_dims["city"] = len(self.user_encoders["city"].classes_)
        
        # 聚类ID编码
        self.user_encoders["cluster_id"] = LabelEncoder()
        self.user_profile_df["cluster_id_encoded"] = self.user_encoders["cluster_id"].fit_transform(self.user_profile_df["cluster_id"])
        self.user_cate_dims["cluster_id"] = len(self.user_encoders["cluster_id"].classes_)
        
        # 物品特征映射 
        self.item_encoders = {}
        self.item_cate_dims = {}
        
        # 用户数值特征
        self.user_numeric_cols = [
            "recency", "frequency", "monetary", "rfm_score",
            "actions_per_active_day_30d", "morning_ratio_30d",
            "afternoon_ratio_30d", "evening_ratio_30d", "late_night_ratio_30d"
        ]
        
        self.item_numeric_cols = []
        
        # 创建ID映射 - 只使用召回结果中的用户和物品
        recall_users = set(self.recall_df["user_id"].unique())
        recall_items = set(self.recall_df["item_id"].unique())
        
        # 创建ID映射
        self.user2idx = {uid: idx for idx, uid in enumerate(sorted(recall_users))}
        self.item2idx = {iid: idx for idx, iid in enumerate(sorted(recall_items))}
        
        logger.info(f"✅ User features: {self.user_cate_dims}")
        logger.info(f"✅ Item features: {self.item_cate_dims}")
        logger.info(f"✅ User numeric: {len(self.user_numeric_cols)} features")
        logger.info(f"✅ Item numeric: {len(self.item_numeric_cols)} features")
        logger.info(f"✅ Unique users: {len(self.user2idx)}")
        logger.info(f"✅ Unique items: {len(self.item2idx)}")
        
    def _build_samples(self):
        """构建训练样本"""
        logger.info("Building training samples...")
        
        # 获取标签窗口 - 使用行为数据的时间范围
        if not self.behavior_df.empty:
            max_dt = self.behavior_df["dt"].max()
            label_start_dt = max_dt - pd.Timedelta(days=self.cutoff_days)
            label_end_dt = max_dt
            
            # 获取训练窗口 - 修正为30天
            train_start_dt = label_start_dt - pd.Timedelta(days=self.train_days)
            train_end_dt = label_start_dt
            
            logger.info(f"📅 Label window: {label_start_dt.date()} to {label_end_dt.date()}")
            logger.info(f"📅 Train window: {train_start_dt.date()} to {train_end_dt.date()}")
            
            # 获取正样本（标签窗口内的点击，且必须在召回结果中）
            positive_df = self.behavior_df[
                (self.behavior_df["dt"] >= label_start_dt) &
                (self.behavior_df["dt"] <= label_end_dt) &
                (self.behavior_df["action_type"].str.lower() == "click")
            ].copy()
            
            # 将行为日志中的用户ID映射回召回数据中的用户ID
            reverse_mapping = {v: k for k, v in self.user_id_mapping.items()}
            positive_df["recall_user_id"] = positive_df["user_id"].map(reverse_mapping)
            positive_df = positive_df.dropna(subset=["recall_user_id"])
            
            # 只保留召回结果中存在的用户-物品对
            recall_pairs = set(zip(self.recall_df["user_id"], self.recall_df["item_id"]))
            positive_pairs = set(zip(positive_df["recall_user_id"].astype(str), positive_df["item_id"])) & recall_pairs
            logger.info(f"✅ Positive samples: {len(positive_pairs)}")
            
            # 获取用户历史行为序列（训练窗口内）
            self.user_sequences = self._build_user_sequences(train_start_dt, train_end_dt)
        else:
            # 如果没有行为数据，创建空的正样本
            positive_pairs = set()
            self.user_sequences = {}
            logger.info("⚠️  No behavior data available, creating empty positive samples")
        
        # 构建样本
        self.samples = []
        
        # 正样本
        for user_id, item_id in positive_pairs:
            if user_id in self.user2idx and item_id in self.item2idx:
                self.samples.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "label": 1
                })
        
        # 负样本（召回结果中但未点击的用户-物品对）
        recall_pairs = set(zip(self.recall_df["user_id"], self.recall_df["item_id"]))
        negative_candidates = recall_pairs - positive_pairs
        
        # 为每个正样本采样负样本
        user_positive_items = defaultdict(set)
        for user_id, item_id in positive_pairs:
            user_positive_items[user_id].add(item_id)
        
        for user_id, positive_items in user_positive_items.items():
            user_negative_candidates = [
                (uid, iid) for uid, iid in negative_candidates 
                if uid == user_id and iid not in positive_items
            ]
            
            # 随机采样负样本
            n_neg = min(len(positive_items) * self.neg_pos_ratio, len(user_negative_candidates))
            if n_neg > 0:
                neg_samples = np.random.choice(
                    len(user_negative_candidates), 
                    size=n_neg, 
                    replace=False
                )
                for idx in neg_samples:
                    user_id, item_id = user_negative_candidates[idx]
                    if user_id in self.user2idx and item_id in self.item2idx:
                        self.samples.append({
                            "user_id": user_id,
                            "item_id": item_id,
                            "label": 0
                        })
        
        # 如果正样本太少，从召回结果中随机采样一些作为负样本
        if len(self.samples) < self.max_samples // 2:
            logger.info("⚠️  Too few positive samples, sampling from recall data...")
            recall_pairs_list = list(recall_pairs)
            n_samples = min(self.max_samples - len(self.samples), len(recall_pairs_list))
            sampled_pairs = np.random.choice(len(recall_pairs_list), size=n_samples, replace=False)
            
            for idx in sampled_pairs:
                user_id, item_id = recall_pairs_list[idx]
                if user_id in self.user2idx and item_id in self.item2idx:
                    self.samples.append({
                        "user_id": user_id,
                        "item_id": item_id,
                        "label": 0
                    })
        
        # 限制样本数量
        if len(self.samples) > self.max_samples:
            logger.info(f"⚠️  Limiting samples from {len(self.samples)} to {self.max_samples}")
            # 保持正负样本比例
            pos_samples = [s for s in self.samples if s['label'] == 1]
            neg_samples = [s for s in self.samples if s['label'] == 0]
            
            # 按比例采样
            pos_ratio = len(pos_samples) / len(self.samples) if self.samples else 0.5
            n_pos = int(self.max_samples * pos_ratio)
            n_neg = self.max_samples - n_pos
            
            if len(pos_samples) > 0:
                pos_samples = np.random.choice(len(pos_samples), size=min(n_pos, len(pos_samples)), replace=False)
                pos_samples = [self.samples[i] for i in pos_samples]
            else:
                pos_samples = []
                
            if len(neg_samples) > 0:
                neg_samples = np.random.choice(len(neg_samples), size=min(n_neg, len(neg_samples)), replace=False)
                neg_samples = [self.samples[i] for i in neg_samples]
            else:
                neg_samples = []
            
            self.samples = pos_samples + neg_samples
        
        logger.info(f"✅ Total samples: {len(self.samples)}")
        logger.info(f"✅ Positive: {sum(1 for s in self.samples if s['label'] == 1)}")
        logger.info(f"✅ Negative: {sum(1 for s in self.samples if s['label'] == 0)}")
        
    def _build_user_sequences(self, start_dt, end_dt):
        """构建用户行为序列"""
        logger.info("Building user sequences...")
        
        # 获取训练窗口内的行为数据
        train_behavior = self.behavior_df[
            (self.behavior_df["dt"] >= start_dt) &
            (self.behavior_df["dt"] <= end_dt) &
            (self.behavior_df["action_type"].str.lower() == "click")
        ].copy()
        
        # 将行为日志中的用户ID映射回召回数据中的用户ID
        reverse_mapping = {v: k for k, v in self.user_id_mapping.items()}
        train_behavior["recall_user_id"] = train_behavior["user_id"].map(reverse_mapping)
        train_behavior = train_behavior.dropna(subset=["recall_user_id"])
        
        # 只保留召回结果中的用户和物品
        recall_users = set(self.recall_df["user_id"].unique())
        recall_items = set(self.recall_df["item_id"].unique())
        train_behavior = train_behavior[
            (train_behavior["recall_user_id"].astype(str).isin(recall_users)) &
            (train_behavior["item_id"].isin(recall_items))
        ]
        
        # 按用户和时间排序
        train_behavior = train_behavior.sort_values(["recall_user_id", "dt"])
        
        user_sequences = {}
        for user_id, group in train_behavior.groupby("recall_user_id"):
            items = group["item_id"].tolist()
            # 只保留在item2idx中的物品
            valid_items = [item for item in items if item in self.item2idx]
            if valid_items:
                user_sequences[str(user_id)] = valid_items[-self.max_seq_len:]  # 保留最近的序列
        
        logger.info(f"✅ Built sequences for {len(user_sequences)} users")
        return user_sequences
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample["user_id"]
        item_id = sample["item_id"]
        label = sample["label"]
        
        # 获取用户特征 - 使用映射后的用户ID
        mapped_user_id = self.user_id_mapping.get(user_id, user_id)
        user_profile = self.user_profile_df[
            self.user_profile_df["user_id"] == mapped_user_id
        ].iloc[0]
        
        user_cate_feats = {
            "gender": int(user_profile["gender_encoded"]),
            "age_range": int(user_profile["age_range_encoded"]),
            "city": int(user_profile["city_encoded"]),
            "cluster_id": int(user_profile["cluster_id_encoded"])
        }
        
        user_numeric_feats = [
            user_profile[col] if pd.notna(user_profile[col]) else 0.0
            for col in self.user_numeric_cols
        ]
        
        # 获取召回分数
        recall_scores = self.recall_df[
            (self.recall_df["user_id"] == user_id) &
            (self.recall_df["item_id"] == item_id)
        ]
        
        if not recall_scores.empty:
            cf_score = recall_scores["cf_score"].iloc[0] if "cf_score" in recall_scores.columns else 0.0
            keyword_score = recall_scores["keyword_score"].iloc[0] if "keyword_score" in recall_scores.columns else 0.0
            dnn_score = recall_scores["dnn_score"].iloc[0] if "dnn_score" in recall_scores.columns else 0.0
        else:
            cf_score = keyword_score = dnn_score = 0.0
        
        recall_scores = [cf_score, keyword_score, dnn_score]
        
        # 获取用户历史序列
        if user_id in self.user_sequences:
            seq_items = self.user_sequences[user_id]
            # 填充到固定长度
            while len(seq_items) < self.max_seq_len:
                seq_items = [0] + seq_items  # 用0填充
            
            seq_items = seq_items[-self.max_seq_len:]  # 截断到最大长度
            
            # 简化的序列特征 - 只使用物品ID的嵌入
            seq_cate_feats = {}
            seq_mask = []
            
            for item_id in seq_items:
                if item_id == 0:  # 填充位置
                    seq_mask.append(0)
                else:
                    if item_id in self.item2idx:
                        seq_mask.append(1)
                    else:
                        seq_mask.append(0)
        else:
            # 没有历史序列
            seq_cate_feats = {}
            seq_mask = [0] * self.max_seq_len
        
        return {
            "user_cate_feats": user_cate_feats,
            "user_numeric_feats": torch.tensor(user_numeric_feats, dtype=torch.float32),
            "item_ids": torch.tensor(self.item2idx[item_id], dtype=torch.long),
            "seq_cate_feats": {k: torch.tensor(v, dtype=torch.long) for k, v in seq_cate_feats.items()},
            "recall_scores": torch.tensor(recall_scores, dtype=torch.float32),
            "seq_mask": torch.tensor(seq_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32)
        }
