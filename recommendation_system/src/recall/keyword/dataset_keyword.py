# src/recall/keyword/dataset_keyword.py

import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple
import logging

from src.data.load_behavior_log import find_window_bounds, iter_clean_behavior
from src.data.load_item_info import load_item_info

# ----------- Logger Setup -----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ----------- Core Functions -----------

def build_item_text_dict(item_df: pd.DataFrame) -> Dict[int, str]:
    """
    Construct item_id to text dict by combining title and content.
    """
    item_text_dict = {}
    for _, row in item_df.iterrows():
        text = f"{row.get('title', '')} {row.get('content', '')}".strip()
        if text:
            item_text_dict[row['item_id']] = text
    logger.info(f"Constructed item_text_dict with {len(item_text_dict)} entries.")
    return item_text_dict

def build_user_text_dict(behavior_df: pd.DataFrame,
                         item_text_dict: Dict[int, str]) -> Dict[int, str]:
    """
    For each user, combine all texts of their interacted items into one text.
    """
    user_text_dict = defaultdict(list)
    for _, row in behavior_df.iterrows():
        user_id, item_id = row['user_id'], row['item_id']
        if item_id in item_text_dict:
            user_text_dict[user_id].append(item_text_dict[item_id])

    combined = {uid: " ".join(texts) for uid, texts in user_text_dict.items()}
    logger.info(f"Constructed user_text_dict with {len(combined)} users.")
    return combined

def load_all_for_keyword(behavior_csv: str,
                          item_csv: str,
                          days_window: int = 30,
                          chunksize: int = 100_000
                          ) -> Tuple[pd.DataFrame, Dict[int, str], Dict[int, str]]:
    """
    Load everything needed for keyword-based recall.
    Returns:
        - behavior_df: DataFrame with user_id, item_id, dt
        - item_text_dict: {item_id: "title + content"}
        - user_text_dict: {user_id: "merged item texts"}
    """
    # 1. Load item metadata
    item_df = load_item_info(item_csv)
    valid_item_ids = set(item_df['item_id'])
    item_text_dict = build_item_text_dict(item_df)

    # 2. Find time window from behavior log
    _, cutoff, upper = find_window_bounds(behavior_csv, chunksize, days_window)

    # 3. Load valid behavior data in time window
    all_behavior = []
    for chunk in iter_clean_behavior(
        input_behavior_csv=behavior_csv,
        chunksize=chunksize,
        cutoff_date=cutoff,
        upper=upper,
        extra_usecols=['item_id'],
        valid_item_ids=valid_item_ids
    ):
        all_behavior.append(chunk[['user_id', 'item_id', 'dt']])

    if not all_behavior:
        logger.warning("No behavior data found after filtering.")
        behavior_df = pd.DataFrame(columns=['user_id', 'item_id', 'dt'])
    else:
        behavior_df = pd.concat(all_behavior, ignore_index=True)

    # 4. Build user text dict
    user_text_dict = build_user_text_dict(behavior_df, item_text_dict)

    return behavior_df, item_text_dict, user_text_dict
