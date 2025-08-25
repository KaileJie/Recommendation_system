# src/data/load_user_profile.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_user_profile_feature(filepath: str) -> pd.DataFrame:
    """
    加载 user_profile_feature.csv，并清洗部分字段类型

    返回字段应包括：
    - user_id, gender, age_range, city, cluster_id
    - recency, frequency, actions_per_active_day_30d
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded user profile from {filepath}, shape = {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load user profile: {e}")
        return pd.DataFrame()

    try:
        # user_id
        df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['user_id'])
        df['user_id'] = df['user_id'].astype(int)

        # gender
        df['gender'] = pd.to_numeric(df['gender'], errors='coerce').fillna(2).astype(int)
        df['gender'] = df['gender'].apply(lambda x: x if x in [0, 1, 2] else 2)

        # age_range
        df['age_range'] = pd.to_numeric(df['age_range'], errors='coerce').fillna(0).astype(int)
        df['age_range'] = df['age_range'].apply(lambda x: x if x in range(0, 9) else 0)

        # city
        df['city'] = df['city'].fillna('unknown').astype(str)

        # cluster_id
        if 'cluster_id' in df.columns:
            df['cluster_id'] = pd.to_numeric(df['cluster_id'], errors='coerce').fillna(-1).astype(int)
        else:
            df['cluster_id'] = -1

        # 数值特征
        for col in ['recency', 'frequency', 'actions_per_active_day_30d']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
            else:
                logger.warning(f"Missing numerical column: {col}, filling with 0.0")
                df[col] = 0.0

    except Exception as e:
        logger.error(f"Error while cleaning user profile feature: {e}")
        return pd.DataFrame()

    return df
