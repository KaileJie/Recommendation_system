# src/utils/weight_utils.py

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ✅ 新增：RFM 权重配置（用于 recall + rank 权重融合等）
RFM_WEIGHTS = {
    "click": 1.0,
    "read": 2.0,
    "like": 3.0,
    "fav": 4.0,
}

def get_time_weight(dt_series: pd.Series,
                    ref_date: pd.Timestamp,
                    method: str = "linear",
                    decay_days: int = 30,
                    min_weight: float = 0.1) -> pd.Series:
    """
    Calculate time decay weights based on behavior timestamp.
    """
    try:
        days_diff = (ref_date - dt_series).dt.days
        days_diff = np.maximum(days_diff, 0)  # ✅ 保证非负

        if method == "linear":
            weights = 1.0 - (days_diff / decay_days)
            weights = weights.where(days_diff <= decay_days, min_weight)  # ✅ 超窗设为 min_weight
            weights = weights.clip(lower=min_weight)  # 防止计算误差导致低于 min_weight

        elif method == "exp":
            decay_rate = np.log(1 / min_weight) / decay_days
            weights = np.exp(-decay_rate * days_diff)
            weights = weights.where(days_diff <= decay_days, min_weight)  # ✅ 超窗设为 min_weight

        else:
            logger.warning(f"[weight_utils] Unsupported method '{method}', fallback to uniform=1.0")
            weights = pd.Series(1.0, index=dt_series.index)

        return weights

    except Exception as e:
        logger.error(f"[weight_utils] Error in get_time_weight: {e}")
        return pd.Series(1.0, index=dt_series.index)  # ✅ 保持不变，返回全 1

def apply_time_weight(
    df: pd.DataFrame,
    dt_col: str = "dt",
    method: str = "linear",
    ref_date: Optional[pd.Timestamp] = None,
    decay_days: int = 30,
    min_weight: float = 0.1
) -> pd.DataFrame:
    """
    Apply time decay weight to a behavior DataFrame, adding a 'time_weight' column.
    """
    try:
        if ref_date is None:
            ref_date = df[dt_col].max()
        df["time_weight"] = get_time_weight(df[dt_col], ref_date, method, decay_days, min_weight)
        return df

    except Exception as e:
        logger.error(f"[weight_utils] Error in apply_time_weight: {e}")
        df["time_weight"] = 1.0
        return df


def apply_rfm_weight(
    df: pd.DataFrame,
    action_col: str = "action_type",  # ✅ 改为默认 action_type（和主流程一致）
    weight_map: dict = None
) -> pd.DataFrame:
    """
    Apply RFM-style weights to actions (e.g., click=1, read=2, ...), stored in 'rfm_weight'.
    """
    if weight_map is None:
        logger.warning("[apply_rfm_weight] No weight_map provided; using uniform=1.0")
        df["rfm_weight"] = 1.0
        return df

    try:
        mapped = df[action_col].map(weight_map)
        unknowns = mapped.isna()
        if unknowns.any():
            logger.warning(f"[apply_rfm_weight] Found unknown actions (count={unknowns.sum()}). Top examples: {df.loc[unknowns, action_col].value_counts().head(3).to_dict()}")
        df["rfm_weight"] = mapped.fillna(1.0)  # ✅ 改为默认 1.0（和 time_weight 异常一致）
        return df
    except Exception as e:
        logger.error(f"[apply_rfm_weight] Error: {e}")
        df["rfm_weight"] = 1.0  # ✅ 改为返回 1.0（统一异常策略）
        return df
