# src/rank/din/train_din_corrected.py - 使用修正数据集的训练脚本

import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from collections import defaultdict

from src.rank.din.din_model import DINModel
from src.rank.din.dataset_din import DINDataset

# ----------- Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
OUTPUT_DIR = "output/din"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据路径
RECALL_DATA_PATH = "output/fusion/fusion_recall.csv"
USER_PROFILE_PATH = "data/processed/user_profile_feature.csv"
ITEM_METADATA_PATH = "data/raw/item_metadata.csv"
BEHAVIOR_LOG_PATH = "data/raw/user_behavior_log_info.csv"

# 修正配置 - 与召回模型保持一致
CONFIG = {
    "embed_dim": 16,           # 更小的嵌入维度
    "max_seq_len": 10,         # 更短的序列长度
    "hidden_dims": [32, 16],   # 更小的隐藏层
    "learning_rate": 0.001,
    "batch_size": 64,          # 更小的批次
    "num_epochs": 3,           # 更少的训练轮数
    "cutoff_days": 7,          # 与召回模型一致
    "train_days": 30,          # 修正：与召回模型一致
    "neg_pos_ratio": 2,
    "max_samples": 10000,      # 增加样本数
    "max_users": 1000          # 增加用户数
}

MODEL_PATH = os.path.join(OUTPUT_DIR, "din_model.pt")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "din_config.json")
LOSS_LOG_PATH = os.path.join(OUTPUT_DIR, "din_loss_log.csv")


def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    user_cate_feats = defaultdict(list)
    user_numeric_feats = []
    item_ids = []
    seq_cate_feats = defaultdict(list)
    recall_scores = []
    seq_masks = []
    labels = []
    
    for sample in batch:
        # 用户类别特征
        for k, v in sample["user_cate_feats"].items():
            user_cate_feats[k].append(v)
        
        # 用户数值特征
        user_numeric_feats.append(sample["user_numeric_feats"])
        
        # 物品ID
        item_ids.append(sample["item_ids"])
        
        # 序列特征
        for k, v in sample["seq_cate_feats"].items():
            seq_cate_feats[k].append(v)
        
        # 召回分数
        recall_scores.append(sample["recall_scores"])
        
        # 序列mask
        seq_masks.append(sample["seq_mask"])
        
        # 标签
        labels.append(sample["label"])
    
    # 转换为tensor
    result = {
        "user_cate_feats": {k: torch.tensor(v, dtype=torch.long) for k, v in user_cate_feats.items()},
        "user_numeric_feats": torch.stack(user_numeric_feats),
        "item_ids": torch.stack(item_ids),
        "seq_cate_feats": {k: torch.stack(v) for k, v in seq_cate_feats.items()},
        "recall_scores": torch.stack(recall_scores),
        "seq_mask": torch.stack(seq_masks),
        "labels": torch.stack(labels)
    }
    
    return result


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            for key in batch:
                if isinstance(batch[key], dict):
                    batch[key] = {k: v.to(device) for k, v in batch[key].items()}
                else:
                    batch[key] = batch[key].to(device)
            
            # 前向传播
            outputs, _ = model(
                user_cate_feats=batch["user_cate_feats"],
                user_numeric_feats=batch["user_numeric_feats"],
                item_ids=batch["item_ids"],
                seq_cate_feats=batch["seq_cate_feats"],
                recall_scores=batch["recall_scores"],
                seq_mask=batch["seq_mask"]
            )
            
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    # 计算指标
    if len(set(all_labels)) > 1:  # 确保有正负样本
        auc_score = roc_auc_score(all_labels, all_preds)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)
    else:
        auc_score = 0.5
        pr_auc = 0.5
    
    return auc_score, pr_auc, all_preds, all_labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], dict):
                batch[key] = {k: v.to(device) for k, v in batch[key].items()}
            else:
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs, attention_weights = model(
            user_cate_feats=batch["user_cate_feats"],
            user_numeric_feats=batch["user_numeric_feats"],
            item_ids=batch["item_ids"],
            seq_cate_feats=batch["seq_cate_feats"],
            recall_scores=batch["recall_scores"],
            seq_mask=batch["seq_mask"]
        )
        
        # 计算损失
        loss = criterion(outputs, batch["labels"])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    logger.info("🚀 Training DIN Model (Corrected Version)...")
    logger.info(f"📅 Time window: train_days={CONFIG['train_days']}, cutoff_days={CONFIG['cutoff_days']}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载数据集
    logger.info("Loading dataset...")
    dataset = DINDataset(
        recall_data_path=RECALL_DATA_PATH,
        user_profile_path=USER_PROFILE_PATH,
        item_metadata_path=ITEM_METADATA_PATH,
        behavior_log_path=BEHAVIOR_LOG_PATH,
        max_seq_len=CONFIG["max_seq_len"],
        cutoff_days=CONFIG["cutoff_days"],
        train_days=CONFIG["train_days"],
        neg_pos_ratio=CONFIG["neg_pos_ratio"],
        max_samples=CONFIG["max_samples"],
        max_users=CONFIG["max_users"]
    )
    
    if len(dataset) == 0:
        logger.error("❌ No samples generated! Please check your data.")
        return
    
    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # 避免多进程问题
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    logger.info(f"✅ Train samples: {len(train_dataset)}")
    logger.info(f"✅ Val samples: {len(val_dataset)}")
    
    # 创建模型
    model_config = {
        "user_cate_dims": dataset.user_cate_dims,
        "user_numeric_dim": len(dataset.user_numeric_cols),
        "embed_dim": CONFIG["embed_dim"],
        "max_seq_len": CONFIG["max_seq_len"],
        "hidden_dims": CONFIG["hidden_dims"]
    }
    
    model = DINModel(**model_config).to(device)
    logger.info(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练循环
    best_auc = 0
    train_losses = []
    val_aucs = []
    
    for epoch in range(CONFIG["num_epochs"]):
        logger.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_auc, val_pr_auc, val_preds, val_labels = evaluate_model(model, val_loader, device)
        val_aucs.append(val_auc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Val PR-AUC: {val_pr_auc:.4f}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"✅ New best model saved (AUC: {val_auc:.4f})")
    
    # 保存配置
    with open(CONFIG_PATH, "w") as f:
        json.dump(model_config, f, indent=2)
    
    # 保存训练日志
    log_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_auc": val_aucs
    })
    log_df.to_csv(LOSS_LOG_PATH, index=False)
    
    logger.info(f"✅ Training completed! Best AUC: {best_auc:.4f}")
    logger.info(f"📦 Model saved to: {MODEL_PATH}")
    logger.info(f"📦 Config saved to: {CONFIG_PATH}")
    logger.info(f"📦 Loss log saved to: {LOSS_LOG_PATH}")


if __name__ == "__main__":
    main()
