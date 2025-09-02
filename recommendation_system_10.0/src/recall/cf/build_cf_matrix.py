# src/recall/cf/build_cf_matrix.py

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import logging

from src.utils.weight_utils import get_time_weight
from src.data.load_behavior_log import RFM_WEIGHTS

logger = logging.getLogger(__name__)

def build_user_item_matrix(df: pd.DataFrame,
                           ref_date: pd.Timestamp,
                           time_weight_method: str = "exp",
                           decay_days: int = 30,
                           min_weight: float = 0.1,
                           return_mappings: bool = False,
                           strict: bool = False) -> csr_matrix:
    """
    Build sparse user-item matrix with action weights and time decay.

    Args:
        df (pd.DataFrame): Behavior log with columns ['user_id', 'item_id', 'action_type', 'dt']
        ref_date (pd.Timestamp): Reference date for time decay
        time_weight_method (str): 'linear' or 'exp'
        decay_days (int): Window size in days for decay
        min_weight (float): Minimum time decay weight
        return_mappings (bool): Whether to return id mappings (user2idx, item2idx)
        strict (bool): Whether to check required columns strictly

    Returns:
        csr_matrix: User-Item matrix [num_users x num_items]
        (Optional) user2idx, item2idx
    """
    try:
        logger.info("[build_cf_matrix] Start building user-item matrix...")
        df = df.copy()

        # 校验字段完整性
        required_cols = ['user_id', 'item_id', 'action_type', 'dt']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"[build_cf_matrix] Missing required column: {col}")

        # 计算 base_weight 和 time_weight
        df['base_weight'] = df['action_type'].map(RFM_WEIGHTS).fillna(0.0)
        df['time_weight'] = get_time_weight(
            dt_series=df['dt'],
            ref_date=ref_date,
            method=time_weight_method,
            decay_days=decay_days,
            min_weight=min_weight
        )
        df['final_weight'] = df['base_weight'] * df['time_weight']

        # ✅ 保留原始顺序，避免 unique() 打乱
        user_ids = df['user_id'].drop_duplicates().tolist()
        item_ids = df['item_id'].drop_duplicates().tolist()

        user2idx = {uid: i for i, uid in enumerate(user_ids)}
        item2idx = {iid: j for j, iid in enumerate(item_ids)}

        df['user_idx'] = df['user_id'].map(user2idx)
        df['item_idx'] = df['item_id'].map(item2idx)

        logger.info(f"[build_cf_matrix] Total users: {len(user2idx)}, items: {len(item2idx)}")

        # 构建稀疏矩阵
        coo = coo_matrix(
            (df['final_weight'], (df['user_idx'], df['item_idx'])),
            shape=(len(user2idx), len(item2idx))
        )

        csr = coo.tocsr()
        logger.info("[build_cf_matrix] User-Item CSR matrix built successfully")

        if return_mappings:
            return csr, user2idx, item2idx
        else:
            return csr

    except Exception as e:
        logger.exception(f"[build_cf_matrix] Error occurred: {e}")
        return None
