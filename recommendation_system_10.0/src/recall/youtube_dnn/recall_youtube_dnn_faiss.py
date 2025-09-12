# src/recall/youtube_dnn/recall_youtube_dnn_faiss.py

import os
import torch
import faiss
import numpy as np
import pickle
import pandas as pd
import logging
from tqdm import tqdm
from datetime import timedelta 

from src.recall.youtube_dnn.dataset_youtube_dnn import YoutubeDNNDataset
from src.recall.youtube_dnn.youtube_dnn import YoutubeDNN
from src.data.load_behavior_log import find_window_bounds

# ----------- Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
OUTPUT_DIR = "output/youtube_dnn"
USER_CSV = "data/processed/user_profile_feature.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
MODEL_PATH = os.path.join(OUTPUT_DIR, "youtube_dnn_model.pt")
INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_item.index")
ID_MAPPING_PATH = os.path.join(OUTPUT_DIR, "faiss_item_id_mapping.pkl")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "youtube_dnn_faiss_recall.csv")

CHUNKSIZE = 50000
TRAIN_WINDOW_DAYS = 30   # 训练窗口
CUTOFF_DAYS = 7          # 留出最后7天做评估
EMB_DIM = 64
TOP_K = 50


def recall_batch_users(top_k=TOP_K):
    logger.info("🚀 Running recall using prebuilt FAISS index...")

    # 1. 时间窗口
    _, cutoff_date, upper_date = find_window_bounds(
        BEHAVIOR_CSV, chunksize=CHUNKSIZE, days_window=TRAIN_WINDOW_DAYS
    )

    # ⚠️ 召回时也去掉最后 7 天，保证和训练/评估一致
    cutoff_dt = upper_date - pd.Timedelta(days=CUTOFF_DAYS)
    logger.info(f"⚠️ Recall cutoff applied: using data < {cutoff_dt}")

    # 2. 加载有效 item_id
    item_df = pd.read_csv(ITEM_CSV)
    valid_item_ids = set(item_df["item_id"].astype(str).unique())
    logger.info(f"✅ Loaded {len(valid_item_ids)} valid item_ids from item metadata")

    # 3. 构建 Dataset
    dataset = YoutubeDNNDataset(
        user_profile_csv=USER_CSV,
        item_csv=ITEM_CSV,
        behavior_csv=BEHAVIOR_CSV,
        cutoff_date=cutoff_date,   # 起点 = 30 天前
        upper_date=cutoff_dt,      # 终点 = 去掉最后 7 天
        chunksize=CHUNKSIZE,
        num_negatives=0,
        filter_item_ids=valid_item_ids
    )

    if len(dataset) == 0:
        logger.error("❌ Dataset is empty.")
        return

    num_users, num_items = dataset.get_num_users_items()
    user_cate_dims = dataset.get_user_categorical_vocab_sizes()
    user_numeric_dim = len(dataset.get_user_numeric_norm()["cols"])
    item_text_dim = dataset.get_item_text_dim()

    # 4. 加载模型
    model = YoutubeDNN(
        num_users=num_users,
        num_items=num_items,
        user_cate_dims=user_cate_dims,
        user_numeric_dim=user_numeric_dim,
        item_text_dim=item_text_dim,
        embed_dim=EMB_DIM
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    logger.info(f"✅ Model loaded from {MODEL_PATH}")

    # 5. 加载 FAISS 索引
    index = faiss.read_index(INDEX_PATH)
    with open(ID_MAPPING_PATH, "rb") as f:
        item_ids = pickle.load(f)
    logger.info(f"✅ FAISS index loaded with {len(item_ids)} items")

    # ⚠️ 只取有效交互用户
    valid_users = set(dataset.interactions["user_id"].unique())
    logger.info(f"🎯 Valid users for recall (三方交集): {len(valid_users)}")

    # 6. 遍历有效用户并召回 Top-K
    results = []
    for user_id in tqdm(valid_users, desc="🔍 Recalling users"):
        uidx = torch.tensor([dataset.user2idx[user_id]], dtype=torch.long)
        g, a, c, cl, num_z = dataset._user_feat_cache[user_id]

        gender = torch.tensor([g], dtype=torch.long)
        age = torch.tensor([a], dtype=torch.long)
        city = torch.tensor([c], dtype=torch.long)
        cluster = torch.tensor([cl], dtype=torch.long)
        num_feat = torch.tensor([num_z], dtype=torch.float)

        with torch.no_grad():
            user_emb = model.get_user_embedding(uidx, gender, age, city, cluster, num_feat)
            user_emb_np = user_emb.cpu().numpy().astype(np.float32)

        scores, indices = index.search(user_emb_np, top_k)
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            results.append({
                "user_id": str(user_id),           # ⚠️ 强制转字符串
                "item_id": str(item_ids[idx]),     # ⚠️ 强制转字符串
                "rank": rank,
                "score": float(score),
                "source": "youtube_dnn"
            })

    # 7. 保存 CSV
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    logger.info(f"✅ Recall results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    recall_batch_users()
