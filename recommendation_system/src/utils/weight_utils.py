# src/utils/weight_utils.py

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_time_weight(dt_series: pd.Series,
                    ref_date: pd.Timestamp,
                    method: str = "linear",
                    decay_days: int = 30,
                    min_weight: float = 0.1) -> pd.Series:
    """
    Calculate time decay weights based on behavior timestamp.
    
    Args:
        dt_series (pd.Series): A datetime Series for user behavior (column: dt).
        ref_date (pd.Timestamp): Reference date (usually max date in dataset).
        method (str): 'linear' or 'exp'
        decay_days (int): Number of days as the decay window.
        min_weight (float): Minimum weight threshold (used for linear method).

    Returns:
        pd.Series of weights (float)
    """
    try:
        days_diff = (ref_date - dt_series).dt.days.clip(lower=0, upper=decay_days)

        if method == "linear":
            weights = 1.0 - (days_diff / decay_days)
            weights = weights.clip(lower=min_weight)

        elif method == "exp":
            decay_rate = np.log(1 / min_weight) / decay_days
            weights = np.exp(-decay_rate * days_diff)

        else:
            logger.warning(f"[weight_utils] Unsupported method '{method}', fallback to uniform=1.0")
            weights = pd.Series(1.0, index=dt_series.index)

        return weights

    except Exception as e:
        logger.error(f"[weight_utils] Error in get_time_weight: {e}")
        return pd.Series(1.0, index=dt_series.index)

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
    action_col: str = "behavior_type",
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
        df["rfm_weight"] = df[action_col].map(weight_map).fillna(0)
        return df
    except Exception as e:
        logger.error(f"[apply_rfm_weight] Error: {e}")
        df["rfm_weight"] = 0
        return df

