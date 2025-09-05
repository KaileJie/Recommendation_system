import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Optional

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
            sim_block = cosine_similarity(block, user_item_matrix.T)

            # Filter by min_sim
            sim_block[sim_block < min_sim] = 0.0

            # ✅ 限制 Top-K
            if topk is not None and topk < sim_block.shape[1]:
                for i in range(sim_block.shape[0]):
                    row = sim_block[i]
                    if np.count_nonzero(row) > topk:
                        top_idx = np.argsort(row)[-topk:]
                        mask = np.zeros_like(row, dtype=bool)
                        mask[top_idx] = True
                        row[~mask] = 0.0
                        sim_block[i] = row

            sim_block_sparse = csr_matrix(sim_block)
            all_blocks.append(sim_block_sparse)

        # Stack vertically to form full matrix
        similarity_matrix = csr_matrix(np.vstack([blk.toarray() for blk in all_blocks]))

        logger.info(f"[compute_similarity] Done. Final shape: {similarity_matrix.shape}")
        return similarity_matrix

    except Exception as e:
        logger.exception(f"[compute_similarity] Error: {e}")
        return None
