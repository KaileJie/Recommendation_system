# src/recall/youtube_dnn/train_youtube_dnn.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
import json
import pandas as pd

from src.recall.youtube_dnn.dataset_youtube_dnn import YoutubeDNNDataset
from src.recall.youtube_dnn.youtube_dnn import YoutubeDNN
from src.data.load_behavior_log import find_window_bounds

# ----------- Logging Setup -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# ----------- Output Directory -----------
OUTPUT_DIR = "output/youtube_dnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------- Config -----------
USER_CSV = "data/processed/user_profile_feature.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
CHUNKSIZE = 50000
DAYS_WINDOW = 30
BATCH_SIZE = 512
EPOCHS = 5
LR = 1e-3
EMB_DIM = 64


def main():
    logger.info("🚀 Training YouTube DNN")

    # 1. Load item metadata & filter item_ids
    item_df = pd.read_csv(ITEM_CSV)
    valid_item_ids = set(item_df["item_id"].astype(str).unique())
    logger.info(f"✅ Loaded {len(valid_item_ids)} valid item_ids from item metadata")

    # 2. 时间窗口
    max_date, cutoff_date, upper_date = find_window_bounds(
        BEHAVIOR_CSV, chunksize=CHUNKSIZE, days_window=DAYS_WINDOW
    )
    if not max_date:
        logger.error("❌ Cannot determine date range")
        return

    # 3. 构建 Dataset & DataLoader
    dataset = YoutubeDNNDataset(
        user_profile_csv=USER_CSV,
        item_csv=ITEM_CSV,
        behavior_csv=BEHAVIOR_CSV,
        cutoff_date=cutoff_date,
        upper_date=upper_date,
        chunksize=CHUNKSIZE,
        num_negatives=4,
        max_text_feat_dim=300,
        filter_item_ids=valid_item_ids  # ✅ 只保留 metadata 中出现的 items
    )

    if len(dataset) == 0:
        logger.error("❌ Dataset is empty, aborting.")
        return

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 构建模型
    num_users, num_items = dataset.get_num_users_items()
    user_cate_dims = dataset.get_user_categorical_vocab_sizes()
    user_numeric_dim = len(dataset.get_user_numeric_norm()["cols"])
    item_text_dim = dataset.get_item_text_dim()

    logger.info(f"Embedding sizes: users={num_users}, items={num_items}, "
                f"user_cate={user_cate_dims}, user_num_dim={user_numeric_dim}, item_text_dim={item_text_dim}")

    model = YoutubeDNN(
        num_users=num_users,
        num_items=num_items,
        user_cate_dims=user_cate_dims,
        user_numeric_dim=user_numeric_dim,
        item_text_dim=item_text_dim,
        embed_dim=EMB_DIM
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 保存模型结构参数 config
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

    # 5. 训练
    model.train()
    loss_log = []

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
            ).squeeze()

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        loss_log.append(avg_loss)
        logger.info(f"✅ Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 6. 保存模型参数
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "youtube_dnn_model.pt"))
    logger.info("✅ Model weights saved")

    # 7. 保存 Loss 日志
    pd.DataFrame({"epoch": list(range(1, EPOCHS + 1)), "loss": loss_log}).to_csv(
        os.path.join(OUTPUT_DIR, "youtube_dnn_loss_log.csv"), index=False
    )
    logger.info("📉 Loss log saved")


if __name__ == "__main__":
    main()
