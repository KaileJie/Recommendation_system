# src/recall/youtube_dnn/faiss_index_youtube_dnn.py
import os
import torch
import faiss
import numpy as np
import pickle
import logging

from src.recall.youtube_dnn.dataset_youtube_dnn import YoutubeDNNDataset
from src.recall.youtube_dnn.youtube_dnn import YoutubeDNN
from src.data.load_behavior_log import find_window_bounds

# ----------- Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# ----------- Output Directory Setup -----------
OUTPUT_DIR = "output/youtube_dnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------- Config -----------
USER_CSV = "data/processed/user_profile_feature.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
MODEL_PATH = os.path.join(OUTPUT_DIR, "youtube_dnn_model.pt")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "youtube_dnn_config.json")
CHUNKSIZE = 50000
DAYS_WINDOW = 30
INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_item.index")
ID_MAPPING_PATH = os.path.join(OUTPUT_DIR, "faiss_item_id_mapping.pkl")


@torch.no_grad()
def build_faiss_index():
    logger.info("🚀 Building FAISS index for YouTube DNN...")

    # 0. 读取 item_metadata 中的合法 item_id
    import pandas as pd
    item_df = pd.read_csv(ITEM_CSV)
    valid_item_ids = set(item_df["item_id"].astype(str).unique())
    logger.info(f"✅ Loaded {len(valid_item_ids)} valid item_ids from item metadata")

    # 1. 时间窗口确定
    _, cutoff_date, upper_date = find_window_bounds(
        BEHAVIOR_CSV, chunksize=CHUNKSIZE, days_window=DAYS_WINDOW
    )

    # 2. 构建 Dataset（无需负采样，但要过滤 item）
    dataset = YoutubeDNNDataset(
        user_profile_csv=USER_CSV,
        item_csv=ITEM_CSV,
        behavior_csv=BEHAVIOR_CSV,
        cutoff_date=cutoff_date,
        upper_date=upper_date,
        chunksize=CHUNKSIZE,
        num_negatives=0,
        filter_item_ids=valid_item_ids
    )

    num_users, num_items = dataset.get_num_users_items()
    user_cate_dims = dataset.get_user_categorical_vocab_sizes()
    user_numeric_dim = len(dataset.get_user_numeric_norm()["cols"])
    item_text_dim = dataset.get_item_text_dim()

    # 3. 构建模型并加载参数
    model = YoutubeDNN(
        num_users=num_users,
        num_items=num_items,
        user_cate_dims=user_cate_dims,
        user_numeric_dim=user_numeric_dim,
        item_text_dim=item_text_dim
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    logger.info("✅ Model loaded.")

    # 4. 生成 item embedding
    item_ids = list(dataset.item2idx.keys())
    item_indices = torch.tensor([dataset.item2idx[iid] for iid in item_ids], dtype=torch.long)
    item_text_feats = torch.tensor([dataset.item_text_feats[iid] for iid in item_ids], dtype=torch.float)

    item_emb = model.get_item_embedding(item_indices, item_text_feats).numpy()
    logger.info(f"✅ Item embeddings shape: {item_emb.shape}")

    # 5. 建立 FAISS 索引
    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积相似度
    faiss.normalize_L2(item_emb)
    index.add(item_emb)
    logger.info(f"✅ FAISS index built: {index.ntotal} items")

    # 6. 保存 index 和 item_id 映射
    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAPPING_PATH, "wb") as f:
        pickle.dump(item_ids, f)

    logger.info(f"📦 Saved FAISS index to {INDEX_PATH}")
    logger.info(f"📦 Saved item_id mapping to {ID_MAPPING_PATH}")


if __name__ == "__main__":
    build_faiss_index()
