# src/recall/youtube_dnn/recall_youtube_dnn_faiss.py

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_DISABLE_FAST_MM"] = "1"  # ✅ 避免 AVX 崩溃（CPU 兼容）

import json
import faiss
faiss.omp_set_num_threads(1)
import torch
torch.set_num_threads(1)
import numpy as np
import pandas as pd
import logging
import torch.nn.functional as F

from src.recall.youtube_dnn.dataset_youtube_dnn import YoutubeDNNDataset
from src.recall.youtube_dnn.youtube_dnn import YoutubeDNN

# ----------- Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
OUTPUT_DIR = "output/youtube_dnn"
RECALL_CSV = os.path.join(OUTPUT_DIR, "youtube_dnn_faiss_recall.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "youtube_dnn_model.pt")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "youtube_dnn_config.json")
USER_CSV = "data/processed/user_profile_feature.csv"
ITEM_CSV = "data/raw/item_metadata.csv"
BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
TOP_K = 50
CHUNKSIZE = 50000


def build_and_run_recall():
    logger.info("🚀 Running FAISS Recall...")

    # 1. Load config & model
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model = YoutubeDNN(
        num_users=config["num_users"],
        num_items=config["num_items"],
        user_cate_dims=config["user_cate_dims"],
        user_numeric_dim=config["user_numeric_dim"],
        item_text_dim=config["item_text_dim"],
        embed_dim=config["embed_dim"]
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))  # ✅ CPU-safe
    model.eval()
    logger.info("✅ Model loaded")

    # 2. Load dataset
    dataset = YoutubeDNNDataset(
        user_profile_csv=USER_CSV,
        item_csv=ITEM_CSV,
        behavior_csv=BEHAVIOR_CSV,
        cutoff_date=None,     
        upper_date=None,      
        chunksize=CHUNKSIZE,
        num_negatives=0,
        recall_mode=True  # ✅ 自动填充时间
)

    logger.info("✅ Dataset loaded for recall")

    # 3. Prepare item embeddings
    item_ids = list(dataset.item2idx.keys())
    item_indices = torch.tensor([dataset.item2idx[iid] for iid in item_ids])
    item_text_feats = torch.tensor(
        np.array([dataset.item_text_feats[iid] for iid in item_ids], dtype=np.float32)
    )

    with torch.no_grad():
        item_embs = model.get_item_embedding(item_indices, item_text_feats)
        item_embs = F.normalize(item_embs, dim=1)  # ✅ Normalize
        item_embs_np = item_embs.cpu().numpy().astype(np.float32)

    # 4. FAISS Index (CPU-safe)
    dim = item_embs_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_embs_np)
    logger.info(f"✅ FAISS index built with {len(item_ids)} items")

    # 5. Prepare user embeddings (batch)
    user_ids = list(dataset.user2idx.keys())
    user_feats = [dataset.get_user_features(uid) for uid in user_ids]

    uidx = torch.tensor([dataset.user2idx[uid] for uid in user_ids])
    gender = torch.tensor([f[0] for f in user_feats])
    age = torch.tensor([f[1] for f in user_feats])
    city = torch.tensor([f[2] for f in user_feats])
    cluster = torch.tensor([f[3] for f in user_feats])
    num_feat = torch.tensor(np.stack([f[4] for f in user_feats]), dtype=torch.float32)

    with torch.no_grad():
        user_embs = model.get_user_embedding(uidx, gender, age, city, cluster, num_feat)
        user_embs = F.normalize(user_embs, dim=1)  # ✅ Normalize
        user_embs_np = user_embs.cpu().numpy().astype(np.float32)

    # 6. FAISS Search (Batch)
    scores, indices = index.search(user_embs_np, TOP_K)

    # 7. Write to CSV
    rows = []
    for i, uid in enumerate(user_ids):
        for rank, (score, idx) in enumerate(zip(scores[i], indices[i]), 1):
            item_id = item_ids[idx]
            rows.append({
                "user_id": uid,
                "item_id": item_id,
                "rank": rank,
                "score": float(score),
                "source": "youtube_dnn"
            })

    recall_df = pd.DataFrame(rows)
    recall_df.to_csv(RECALL_CSV, index=False)
    logger.info(f"✅ Recall results saved to {RECALL_CSV}")


if __name__ == "__main__":
    build_and_run_recall()