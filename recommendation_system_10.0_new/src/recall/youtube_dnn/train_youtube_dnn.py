# src/recall/youtube_dnn/train_youtube_dnn.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
import json
import pandas as pd
import argparse
import numpy as np  # ✅ 新增，用于 float32 转换

from src.recall.youtube_dnn.dataset_youtube_dnn import YoutubeDNNDataset
from src.recall.youtube_dnn.youtube_dnn import YoutubeDNN
from src.data.load_behavior_log import find_window_bounds

# ----------- Logging Setup -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------- Output Directory -----------
OUTPUT_DIR = "output/youtube_dnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------- Config -----------
USER_CSV = "data/processed/user_profile_feature.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
CHUNKSIZE = 50000
TRAIN_WINDOW_DAYS = 30   # 训练窗口
CUTOFF_DAYS = 7          # 留出最后7天做评估
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
EMB_DIM = 64

parser = argparse.ArgumentParser()
parser.add_argument("--num_negatives", type=int, default=4, help="Number of negative samples per positive")
args = parser.parse_args()

# ----------- Softmax Loss -----------
class SoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits):
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return self.loss_fn(logits, labels)

# ----------- Recall@K Eval -----------
def evaluate_recall_at_k(model, dataset, k=10):
    model.eval()
    recalls = []
    for user_id in dataset.interactions["user_id"].unique():
        g, a, c, cl, num_z = dataset.get_user_features(user_id)
        uidx = torch.tensor([dataset.user2idx[user_id]])
        gender = torch.tensor([g])
        age = torch.tensor([a])
        city = torch.tensor([c])
        cluster = torch.tensor([cl])
        num_feat = torch.tensor([num_z])

        with torch.no_grad():
            user_emb = model.get_user_embedding(uidx, gender, age, city, cluster, num_feat)

        all_item_ids = list(dataset.item2idx.keys())
        item_embs = model.get_item_embedding(
            torch.tensor([dataset.item2idx[iid] for iid in all_item_ids]),
            torch.tensor(np.array([dataset.item_text_feats[iid] for iid in all_item_ids], dtype=np.float32))  # ✅ 修复 Float64 → Float32 错误
        )
        with torch.no_grad():
            scores = torch.matmul(user_emb, item_embs.T).squeeze()
            topk = torch.topk(scores, k).indices.tolist()

        gt_items = list(dataset.user2items[user_id])
        pred_items = [all_item_ids[i] for i in topk]
        hit = len(set(gt_items) & set(pred_items))
        recall = hit / len(gt_items) if gt_items else 0
        recalls.append(recall)
    return sum(recalls) / len(recalls)


def main():
    logger.info("🚀 Training YouTube DNN")

    item_df = pd.read_csv(ITEM_CSV)
    valid_item_ids = set(item_df["item_id"].astype(str).unique())
    logger.info(f"✅ Loaded {len(valid_item_ids)} valid item_ids from item metadata")

    max_date, cutoff_date, upper_date = find_window_bounds(
        BEHAVIOR_CSV, chunksize=CHUNKSIZE, days_window=TRAIN_WINDOW_DAYS
    )
    if not max_date:
        logger.error("❌ Cannot determine date range")
        return

    cutoff_dt = upper_date - pd.Timedelta(days=CUTOFF_DAYS)
    logger.info(f"⚠️ Training cutoff applied: using data < {cutoff_dt}")

    dataset = YoutubeDNNDataset(
        user_profile_csv=USER_CSV,
        item_csv=ITEM_CSV,
        behavior_csv=BEHAVIOR_CSV,
        cutoff_date=cutoff_date,
        upper_date=cutoff_dt,
        chunksize=CHUNKSIZE,
        num_negatives=args.num_negatives,
        max_text_feat_dim=300,
        filter_item_ids=valid_item_ids
    )

    if len(dataset) == 0:
        logger.error("❌ Dataset is empty, aborting.")
        return

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_users, num_items = dataset.get_num_users_items()
    user_cate_dims = dataset.get_user_categorical_vocab_sizes()
    user_numeric_dim = len(dataset.get_user_numeric_norm()["cols"])
    item_text_dim = dataset.get_item_text_dim()

    model = YoutubeDNN(
        num_users=num_users,
        num_items=num_items,
        user_cate_dims=user_cate_dims,
        user_numeric_dim=user_numeric_dim,
        item_text_dim=item_text_dim,
        embed_dim=EMB_DIM
    )

    criterion = SoftmaxLoss()  # ✅ 替换 loss 函数为 SoftmaxLoss
    optimizer = optim.Adam(model.parameters(), lr=LR)

    config = {
        "num_users": num_users,
        "num_items": num_items,
        "user_cate_dims": user_cate_dims,
        "user_numeric_dim": user_numeric_dim,
        "item_text_dim": item_text_dim,
        "embed_dim": EMB_DIM
    }
    with open(os.path.join(OUTPUT_DIR, "youtube_dnn_config.json"), "w") as f:
        json.dump(config, f)
    logger.info("✅ Model config saved")

    model.train()
    loss_log = []
    recall_log = []

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            (
                user_ids, item_ids, labels,
                gender, age_range, city, cluster_id,
                user_numeric, item_text_feat,
                time_weight
            ) = batch

            optimizer.zero_grad()
            logits = model(
                user_id=user_ids,
                item_id=item_ids,
                gender=gender,
                age_range=age_range,
                city=city,
                cluster_id=cluster_id,
                user_numeric=user_numeric,
                item_text_feat=item_text_feat
            )
            logits = torch.stack([logits, 1 - logits], dim=1)  # ✅ Softmax 格式
            loss = criterion(logits)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_log.append(avg_loss)

        recall_k = evaluate_recall_at_k(model, dataset, k=10)
        recall_log.append(recall_k)

        logger.info(f"✅ Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Recall@10: {recall_k:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "youtube_dnn_model.pt"))
    logger.info("✅ Model weights saved")

    pd.DataFrame({
        "epoch": list(range(1, EPOCHS + 1)),
        "loss": loss_log,
        "recall@10": recall_log
    }).to_csv(os.path.join(OUTPUT_DIR, "youtube_dnn_loss_log.csv"), index=False)
    logger.info("📉 Loss & Recall@10 log saved")


if __name__ == "__main__":
    main()
