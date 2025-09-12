# src/data/load_user_profile.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_user_profile_feature(filepath: str) -> pd.DataFrame:
    """
    加载 user_profile_feature.csv，并清洗字段类型

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

        # gender: 0=女, 1=男, 2=未知（与 load_user_info.py 保持一致）
        df['gender'] = pd.to_numeric(df['gender'], errors='coerce').fillna(2).astype(int)
        df['gender'] = df['gender'].apply(lambda x: x if x in [0, 1, 2] else 2)

        # age_range: 0~8 为合法年龄段，0=未知（与 load_user_info.py 保持一致）
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
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                
                # 特殊处理 recency 缺失值
                if col == 'recency':
                    if df[col].isna().any():
                        # recency: 缺失表示从未活跃，用比现有最大值更大的值填充
                        # 这样在RFM分析中会被归类为最不活跃的用户
                        max_recency = df[col].max()
                        if pd.isna(max_recency):
                            fill_value = 31.0  # 默认值
                        else:
                            fill_value = max_recency + 1.0  # 比最大值大1
                        df[col] = df[col].fillna(fill_value)
                        logger.info(f"Filled missing recency with {fill_value} (never active, worse than max={max_recency})")
                else:
                    # 其他数值特征缺失用0填充
                    df[col] = df[col].fillna(0.0)
            else:
                if col == 'recency':
                    logger.warning(f"Missing numerical column: {col}, filling with 31.0 (never active)")
                    df[col] = 31.0
                else:
                    logger.warning(f"Missing numerical column: {col}, filling with 0.0")
                    df[col] = 0.0

    except Exception as e:
        logger.error(f"Error while cleaning user profile feature: {e}")
        return pd.DataFrame()

    return df
