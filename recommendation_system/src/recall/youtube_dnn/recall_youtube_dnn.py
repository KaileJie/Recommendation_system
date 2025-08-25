# src/recall/youtube_dnn/recall_youtube_dnn.py

import torch
import faiss
import logging
import random
import numpy as np

from src.recall.youtube_dnn.dataset_youtube_dnn import YoutubeDNNDataset
from src.recall.youtube_dnn.youtube_dnn import YoutubeDNN
from src.data.load_behavior_log import find_window_bounds

# ----------- Logging Setup -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
USER_CSV = "data/processed/user_profile_feature.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
MODEL_PATH = "output/youtube_dnn_model.pt"
CHUNKSIZE = 50000
DAYS_WINDOW = 30
EMB_DIM = 64
TOP_K = 10


def recall_for_user(user_id: str = None, top_k: int = TOP_K):
    logger.info("🚀 Running YouTube DNN recall with FAISS...")

    # 1. 时间窗口
    max_date, cutoff_date, upper_date = find_window_bounds(
        BEHAVIOR_CSV, chunksize=CHUNKSIZE, days_window=DAYS_WINDOW
    )
    if not max_date:
        logger.error("❌ Cannot determine date range")
        return []

    # 2. 构建 Dataset（不采负样本）
    dataset = YoutubeDNNDataset(
        user_profile_csv=USER_CSV,
        item_csv=ITEM_CSV,
        behavior_csv=BEHAVIOR_CSV,
        cutoff_date=cutoff_date,
        upper_date=upper_date,
        chunksize=CHUNKSIZE,
        num_negatives=0
    )

    if len(dataset) == 0:
        logger.error("❌ Dataset is empty, aborting recall.")
        return []

    num_users, num_items = dataset.get_num_users_items()

    # 3. 加载模型
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
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    logger.info("✅ Model loaded")

    # 4. 获取 user_id
    if user_id is None:
        user_id = random.choice(list(dataset.user2idx.keys()))
        logger.info(f"🎲 Randomly picked user: {user_id}")

    if user_id not in dataset.user2idx:
        logger.error(f"❌ User {user_id} not found.")
        return []

    # 5. 计算所有 item embedding
    item_id_list = list(dataset.item2idx.keys())
    item_tensor = torch.tensor([dataset.item2idx[i] for i in item_id_list], dtype=torch.long)
    item_text_feat = torch.stack([
        torch.tensor(dataset.item_text_feats.get(iid, np.zeros(item_text_dim)), dtype=torch.float)
        for iid in item_id_list
    ])

    with torch.no_grad():
        item_emb = model.get_item_embedding(item_tensor, item_text_feat).numpy()

    # 6. 构建 FAISS 索引
    faiss_index = faiss.IndexFlatIP(EMB_DIM)
    faiss_index.add(item_emb)

    # 7. 计算 user embedding
    uidx = torch.tensor([dataset.user2idx[user_id]], dtype=torch.long)
    gender, age, city, cluster, num_z, _ = dataset._user_feat_cache[user_id]
    gender = torch.tensor([gender])
    age = torch.tensor([age])
    city = torch.tensor([city])
    cluster = torch.tensor([cluster])
    num_z = torch.tensor([num_z], dtype=torch.float)

    with torch.no_grad():
        user_emb = model.get_user_embedding(uidx, gender, age, city, cluster, num_z).numpy()

    # 8. FAISS 检索 Top-K
    scores, indices = faiss_index.search(user_emb, top_k)
    top_items = [item_id_list[i] for i in indices[0]]

    logger.info(f"🎯 Recall for user {user_id}: {top_items}")
    return top_items


if __name__ == "__main__":
    recall_for_user(user_id=None, top_k=10)
