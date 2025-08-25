# src/recall/cf/item_cf_recall.py

import os
import logging
import pandas as pd

from src.utils.weight_utils import get_time_weight
from src.recall.cf.dataset_cf import load_cf_behavior_df, load_valid_item_ids
from src.recall.cf.build_cf_matrix import build_user_item_matrix
from src.recall.cf.compute_similarity import compute_item_sim_matrix
from src.recall.cf.batch_recommend import recommend_for_users_batch
from src.data.load_behavior_log import ensure_outdir

# ----------- Config -----------
INPUT_BEHAVIOR_CSV = "data/raw/user_behavior_log_info.csv"
ITEM_METADATA_CSV = "data/raw/item_metadata.csv"
OUTDIR = "output/item_cf/"
DAYS_WINDOW = 30
CHUNKSIZE = 50000
TOP_K = 10
MIN_SIM = 0.2
BLOCK_SIZE = 500
MAX_USERS = None

# ----------- Logging Setup -----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    logger.info("🚀 Starting Item-based CF recall pipeline")

    # 1. 加载合法 item id
    valid_item_ids = load_valid_item_ids(ITEM_METADATA_CSV)

    # 2. 加载并清洗行为数据
    behavior_df = load_cf_behavior_df(
        behavior_csv=INPUT_BEHAVIOR_CSV,
        days_window=DAYS_WINDOW,
        valid_item_ids=valid_item_ids,
        chunksize=CHUNKSIZE
    )

    if behavior_df.empty:
        logger.error("❌ No valid behavior data found.")
        return

    # ✅ 快速测试模式限制用户数量
    if MAX_USERS is not None:
        top_users = behavior_df["user_id"].unique()[:MAX_USERS]
        behavior_df = behavior_df[behavior_df["user_id"].isin(top_users)]
        logger.info(f"🧪 Quick test mode: restricted to {len(top_users)} users")

    # 3. 构建用户-物品矩阵（内部自动加权）
    ref_date = behavior_df["dt"].max()
    UI, u2idx, i2idx = build_user_item_matrix(
        df=behavior_df,
        ref_date=ref_date,
        time_weight_method="exp",
        decay_days=DAYS_WINDOW,
        min_weight=0.1,
        return_mappings=True,
        strict=True
    )
    idx2it = {v: k for k, v in i2idx.items()}

    # 4. 计算 item-item 相似度
    S = compute_item_sim_matrix(UI, min_sim=MIN_SIM, block_size=BLOCK_SIZE)

    # 5. 推荐召回
    all_users = [u for u in u2idx if UI[u2idx[u]].getnnz() > 0]
    logger.info(f"🎯 Performing recall for {len(all_users)} users...")

    recs_df = recommend_for_users_batch(
        UI, S, u2idx, idx2it,
        user_ids=all_users,
        n_recs=TOP_K,
        batch_size=512,
        filter_seen=True,
        min_score=None,
        as_dataframe=True,
        source_tag="itemcf"
    )


    # 6. 保存结果
    ensure_outdir(OUTDIR)
    output_path = os.path.join(OUTDIR, "itemcf_recall.csv")
    recs_df.to_csv(output_path, index=False)

    # 打印矩阵稀疏信息
    logger.info("📊 Matrix stats:")
    logger.info(f"  🔸 User-Item matrix shape: {UI.shape}, non-zero entries: {UI.nnz}")
    logger.info(f"  🔸 Item-Item similarity matrix shape: {S.shape}, non-zero entries: {S.nnz}")

    logger.info(f"✅ Recommendation results saved to: {output_path}")



if __name__ == "__main__":
    main()
