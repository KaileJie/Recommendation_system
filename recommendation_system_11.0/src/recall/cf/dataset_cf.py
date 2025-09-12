import pandas as pd
import logging
from datetime import timedelta
from src.data.load_behavior_log import (
    iter_clean_behavior,
    find_window_bounds,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def load_valid_item_ids(item_metadata_csv: str) -> set:
    df = pd.read_csv(item_metadata_csv)
    if 'item_id' not in df.columns:
        raise ValueError("item_metadata.csv 缺少 item_id 字段")
    return set(pd.to_numeric(df['item_id'], errors='coerce').dropna().astype(int))


def load_cf_behavior_df(behavior_csv: str,
                        days_window: int = 30,
                        valid_item_ids: set = None,
                        chunksize: int = 100_000) -> pd.DataFrame:
    """
    加载并清洗用户行为数据，用于 CF 模型。

    返回字段：['user_id', 'item_id', 'action_type', 'dt']
    """
    logger.info("🔄 Preparing CF behavior data...")

    _, cutoff, upper = find_window_bounds(behavior_csv, chunksize=chunksize, days_window=days_window)

    dfs = []
    for chunk in iter_clean_behavior(
        input_behavior_csv=behavior_csv,
        chunksize=chunksize,
        cutoff_date=cutoff,
        upper=upper,
        extra_usecols=['item_id'],
        valid_item_ids=valid_item_ids
    ):
        dfs.append(chunk)

    if not dfs:
        logger.warning("⚠️ No valid behavior data loaded.")
        return pd.DataFrame(columns=['user_id', 'item_id', 'action_type', 'dt'])

    df = pd.concat(dfs, ignore_index=True)

    return df[['user_id', 'item_id', 'action_type', 'dt']]
