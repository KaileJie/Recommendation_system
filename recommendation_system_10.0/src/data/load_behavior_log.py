# src/data/load_behavior_log.py

import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# ----------- Logger Setup -----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

# ----------- Config -----------
RFM_WEIGHTS = {
    'click': 1.0,
    'read': 2.0,
    'like': 3.0,
    'fav': 4.0,
}

VALID_ACTIONS = set(RFM_WEIGHTS.keys())

# ----------- Helper functions -----------

def ensure_outdir(path: str):
    """Create output directory if it does not exist."""
    try:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Output directory ensured: {path}")
    except Exception as e:
        logger.error(f"Failed to create output directory {path}: {e}")

def build_datetime(year_series, mmd_series, sec_series):
    """
    Convert year, MMDD, and seconds-since-midnight into a full datetime64.
    """
    try:
        mmd_str = mmd_series.astype(str).str.zfill(4)
        month = pd.to_numeric(mmd_str.str[:2], errors='coerce')
        day   = pd.to_numeric(mmd_str.str[2:], errors='coerce')
        year  = pd.to_numeric(year_series, errors='coerce')

        date_part = pd.to_datetime(
            year.astype('Int64').astype(str) + '-' +
            month.astype('Int64').astype(str).str.zfill(2) + '-' +
            day.astype('Int64').astype(str).str.zfill(2),
            errors='coerce'
        )

        seconds = pd.to_numeric(sec_series, errors='coerce').fillna(0.0)
        return date_part + pd.to_timedelta(seconds, unit='s')
    except Exception as e:
        logger.error(f"Error building datetime: {e}")
        return pd.Series(pd.NaT, index=year_series.index)

def find_window_bounds(input_behavior_csv: str, chunksize: int, days_window: int):
    """
    Find max date from behavior logs, return (max_date, cutoff_date, upper_date)
    """
    max_date = None
    try:
        for chunk in pd.read_csv(input_behavior_csv, chunksize=chunksize,
                                 usecols=['year', 'time_stamp'], dtype=str, index_col=False):
            dt = build_datetime(chunk['year'], chunk['time_stamp'], pd.Series(0, index=chunk.index))
            local_max = dt.max()
            if pd.notna(local_max) and (max_date is None or local_max > max_date):
                max_date = local_max
    except FileNotFoundError:
        logger.error(f"File not found: {input_behavior_csv}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error while scanning behavior log: {e}")
        return None, None, None

    if max_date is None:
        logger.error("No valid dates in behavior log; check 'year' and 'time_stamp' columns.")
        return None, None, None

    cutoff = max_date.normalize() - timedelta(days=days_window - 1)
    upper  = max_date.normalize() + timedelta(days=1)

    logger.info(f"Max behavior date found: {max_date}, window from {cutoff} to {upper}")
    return max_date, cutoff, upper

def iter_clean_behavior(input_behavior_csv: str,
                        chunksize: int,
                        cutoff_date: pd.Timestamp,
                        upper: pd.Timestamp,
                        extra_usecols=None,
                        add_hour=False,
                        valid_actions=VALID_ACTIONS,
                        valid_item_ids=None):
    """
    Stream and yield pre-cleaned behavior dataframes from CSV.
    """
    base_cols = ['user_id', 'item_id', 'year', 'time_stamp', 'action_type', 'timestamp']
    usecols = list(dict.fromkeys(base_cols + (extra_usecols or [])))

    try:
        for chunk in pd.read_csv(input_behavior_csv,
                                 chunksize=chunksize,
                                 usecols=usecols,
                                 dtype=str,
                                 index_col=False):   # ✅ 强制不要把第一列当 index

            # 🔎 调试打印：确认 user_id 是否正确
            logger.debug(f"[iter_clean_behavior] Sample user_id raw: {chunk['user_id'].head(5).tolist()}")

            chunk['dt'] = build_datetime(chunk['year'], chunk['time_stamp'], chunk['timestamp'])

            # Filter by date window
            chunk = chunk[(chunk['dt'] >= cutoff_date) & (chunk['dt'] < upper)].copy()
            if chunk.empty:
                logger.info("Skipped empty chunk after time filtering.")
                continue

            if valid_item_ids is not None:
                chunk['item_id'] = pd.to_numeric(chunk['item_id'], errors='coerce').astype('Int64')
                before = len(chunk)
                chunk = chunk[chunk['item_id'].isin(valid_item_ids)]
                logger.info(f"Filtered chunk by valid item_ids: {before} → {len(chunk)}")
                if chunk.empty:
                    continue

            # Clean and filter user_id
            chunk['user_id'] = pd.to_numeric(chunk['user_id'], errors='coerce').astype('Int64')
            before_drop = len(chunk)
            chunk = chunk.dropna(subset=['user_id'])
            after_drop = len(chunk)
            if before_drop > after_drop:
                logger.warning(f"Dropped {before_drop - after_drop} rows with invalid user_id.")

            chunk['user_id'] = chunk['user_id'].astype(int)

            # Clean action_type
            chunk['action_type'] = chunk['action_type'].astype(str).str.lower().str.strip()
            if valid_actions:
                original = len(chunk)
                chunk = chunk[chunk['action_type'].isin(valid_actions)]
                logger.info(f"Filtered chunk by valid actions: {original} → {len(chunk)}")
                if chunk.empty:
                    continue

            if add_hour:
                chunk['hour'] = chunk['dt'].dt.hour.astype(int)

            keep = ['user_id', 'action_type', 'dt'] + (['hour'] if add_hour else []) + (extra_usecols or [])
            keep = list(dict.fromkeys(keep))  # remove duplicates
            yield chunk[keep]

    except FileNotFoundError:
        logger.error(f"File not found: {input_behavior_csv}")
    except pd.errors.ParserError:
        logger.error(f"CSV parsing error in file: {input_behavior_csv}")
    except Exception as e:
        logger.error(f"Unexpected error processing behavior file: {e}")
