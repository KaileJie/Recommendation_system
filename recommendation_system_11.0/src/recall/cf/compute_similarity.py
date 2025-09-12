import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Optional
from scipy.sparse import vstack

logger = logging.getLogger(__name__)

def compute_item_sim_matrix(user_item_matrix: csr_matrix,
                            min_sim: float = 0.0,
                            block_size: int = 500,
                            topk: Optional[int] = None) -> csr_matrix:
    """
    Compute item-item similarity matrix with filtering.

    Args:
        user_item_matrix (csr_matrix): [num_users x num_items] sparse matrix
        min_sim (float): Minimum similarity threshold to keep
        block_size (int): Number of items per block when computing similarity
        topk (int or None): If set, keep only top-k similar items per item

    Returns:
        csr_matrix: [num_items x num_items] sparse item-item similarity matrix
    """
    try:
        logger.info(f"[compute_similarity] Start computing item-item similarity...")
        num_items = user_item_matrix.shape[1]
        all_blocks = []

        for start in range(0, num_items, block_size):
            end = min(start + block_size, num_items)
            logger.info(f"[compute_similarity] Computing block: {start} to {end}")

            block = user_item_matrix[:, start:end].T

            # 计算余弦相似度
            cosine_sim = cosine_similarity(block, user_item_matrix.T)

            # 当前块的物品流行度
            block_popularity = np.array((block > 0).sum(axis=1)).flatten()
            
            # 所有物品的流行度
            all_popularity = np.array((user_item_matrix > 0).sum(axis=0)).flatten()

            # 流行度归一化：相似度 / sqrt(流行度乘积)
            for i in range(block.shape[0]):
                for j in range(user_item_matrix.shape[1]):
                    if cosine_sim[i, j] > 0:
                        # 使用正确的索引：i是块内索引，j是全局索引
                        pop_factor = np.sqrt(block_popularity[i] * all_popularity[j])
                        if pop_factor > 0:  # 避免除零
                            cosine_sim[i, j] /= pop_factor

            # Filter by min_sim
            cosine_sim[cosine_sim < min_sim] = 0.0

            # ✅ 限制 Top-K
            if topk is not None and topk < cosine_sim.shape[1]:
                for i in range(cosine_sim.shape[0]):
                    row = cosine_sim[i]
                    if np.count_nonzero(row) > topk:
                        top_idx = np.argsort(row)[-topk:]
                        mask = np.zeros_like(row, dtype=bool)
                        mask[top_idx] = True
                        row[~mask] = 0.0
                        cosine_sim[i] = row

            sim_block_sparse = csr_matrix(cosine_sim)
            all_blocks.append(sim_block_sparse)

        # 使用 scipy.sparse.vstack 避免 OOM
        similarity_matrix = vstack(all_blocks)

        logger.info(f"[compute_similarity] Done. Final shape: {similarity_matrix.shape}")
        return similarity_matrix

    except Exception as e:
        logger.exception(f"[compute_similarity] Error: {e}")
        return None
