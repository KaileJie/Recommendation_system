# src/recall/youtube_dnn/faiss_index_youtube_dnn.py

import os
import torch
import faiss
import numpy as np
import pickle
import logging
import json
import pandas as pd

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
CONFIG_PATH = os.path.join(OUTPUT_DIR, "youtube_dnn_config.json")  # ✅ 新增：读取 config
CHUNKSIZE = 50000
TRAIN_WINDOW_DAYS = 30
CUTOFF_DAYS = 7
INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_item.index")
ID_MAPPING_PATH = os.path.join(OUTPUT_DIR, "faiss_item_id_mapping.pkl")


@torch.no_grad()
def build_faiss_index():
    logger.info("🚀 Building FAISS index for YouTube DNN...")

    item_df = pd.read_csv(ITEM_CSV)
    valid_item_ids = set(item_df["item_id"].astype(str).unique())
    logger.info(f"✅ Loaded {len(valid_item_ids)} valid item_ids from item metadata")

    max_dt, cutoff_date, upper_date = find_window_bounds(
        BEHAVIOR_CSV, chunksize=CHUNKSIZE, days_window=TRAIN_WINDOW_DAYS
    )

    cutoff_dt = upper_date - pd.Timedelta(days=CUTOFF_DAYS)
    logger.info(f"⚠️ Index cutoff applied: using data < {cutoff_dt}")

    dataset = YoutubeDNNDataset(
        user_profile_csv=USER_CSV,
        item_csv=ITEM_CSV,
        behavior_csv=BEHAVIOR_CSV,
        cutoff_date=cutoff_date,
        upper_date=cutoff_dt,
        chunksize=CHUNKSIZE,
        num_negatives=0,
        filter_item_ids=valid_item_ids
    )

    num_users, num_items = dataset.get_num_users_items()
    user_cate_dims = dataset.get_user_categorical_vocab_sizes()
    user_numeric_dim = len(dataset.get_user_numeric_norm()["cols"])
    item_text_dim = dataset.get_item_text_dim()

    # ✅ 新增：读取训练 config 中的 embed_dim 和 hidden_dims
    with open(CONFIG_PATH, "r") as f:
        model_config = json.load(f)

    model = YoutubeDNN(
        num_users=num_users,
        num_items=num_items,
        user_cate_dims=user_cate_dims,
        user_numeric_dim=user_numeric_dim,
        item_text_dim=item_text_dim,
        embed_dim=model_config.get("embed_dim", 64),  # ✅ 明确指定
        hidden_dims=tuple(model_config.get("hidden_dims", [128, 64])),
        dropout=model_config.get("dropout", 0.2)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    logger.info("✅ Model loaded.")

    idx2item = {v: k for k, v in dataset.item2idx.items()}
    ordered_item_ids = [idx2item[i] for i in range(len(idx2item))]

    item_indices = torch.tensor(list(range(len(idx2item))), dtype=torch.long)
    item_text_feats = torch.tensor([
        dataset.item_text_feats[iid] for iid in ordered_item_ids
    ], dtype=torch.float)

    item_emb = model.get_item_embedding(item_indices, item_text_feats).numpy()
    logger.info(f"✅ Item embeddings shape: {item_emb.shape}")

    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(item_emb)  # ✅ 保证和召回阶段一致
    index.add(item_emb)
    logger.info(f"✅ FAISS index built: {index.ntotal} items")

    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAPPING_PATH, "wb") as f:
        pickle.dump(ordered_item_ids, f)

    logger.info(f"📦 Saved FAISS index to {INDEX_PATH}")
    logger.info(f"📦 Saved item_id mapping to {ID_MAPPING_PATH}")


if __name__ == "__main__":
    build_faiss_index()
