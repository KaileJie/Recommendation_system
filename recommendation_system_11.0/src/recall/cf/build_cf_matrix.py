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
                           all_user_ids: set = None, # ⚠️ 新增：所有用户id
                           all_item_ids: set = None, # ⚠️ 新增：所有物品id
                           time_weight_method: str = "exp",
                           decay_days: int = 30,
                           min_weight: float = 0.1,
                           return_mappings: bool = False,
                           strict: bool = False) -> csr_matrix:
    """
    添加冷启动处理：支持新用户和物品
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
        
        # 修改：聚合重复的user_id和item_id，取final_weight最大的
        df_aggregated = df.groupby(['user_id', 'item_id'])['final_weight'].max().reset_index()

        # 冷启动处理：确保所有用户和物品都在矩阵中
        if all_user_ids is not None:
            #添加没有交互行为的新用户
            existing_users = set(df_aggregated['user_id'])
            new_users = all_user_ids - existing_users
            if new_users:
                logger.info(f"[build_cf_matrix] Adding {len(new_users)} new users for cold start")
                new_user_rows = pd.DataFrame({
                    'user_id': list(new_users),
                    'item_id': 0, #占位符
                    'final_weight': 0.0
                })
                df_aggregated = pd.concat([df_aggregated, new_user_rows], ignore_index=True)

        if all_item_ids is not None:
            #添加没有交互行为的新物品
            existing_items = set(df_aggregated['item_id'])
            new_items = all_item_ids - existing_items
            if new_items:
                logger.info(f"[build_cf_matrix] Adding {len(new_items)} new items for cold start")
                new_item_rows = pd.DataFrame({
                    'user_id': 0, #占位符
                    'item_id': list(new_items),
                    'final_weight': 0.0
                })
                df_aggregated = pd.concat([df_aggregated, new_item_rows], ignore_index=True)

        # ✅ 保留原始顺序，避免 unique() 打乱
        user_ids = df_aggregated['user_id'].drop_duplicates().tolist()
        item_ids = df_aggregated['item_id'].drop_duplicates().tolist()

        user2idx = {uid: i for i, uid in enumerate(user_ids)}
        item2idx = {iid: j for j, iid in enumerate(item_ids)}

        df_aggregated['user_idx'] = df_aggregated['user_id'].map(user2idx)
        df_aggregated['item_idx'] = df_aggregated['item_id'].map(item2idx)

        logger.info(f"[build_cf_matrix] Total users: {len(user2idx)}, items: {len(item2idx)}")

        # 构建稀疏矩阵
        coo = coo_matrix(
            (df_aggregated['final_weight'], (df_aggregated['user_idx'], df_aggregated['item_idx'])),
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
