import pandas as pd
import logging

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

AGE_MAP = {
    1: 'Under 18',
    2: '18-24',
    3: '25-29',
    4: '30-34',
    5: '35-39',
    6: '40-49',
    7: '50-59',
    8: '60+',
    0: 'unknown'  # 补充 0 表示缺失
}

def load_user_info(filepath: str) -> pd.DataFrame:
    """
    Load and process user profile CSV with logging-based error handling.

    Returns:
        pd.DataFrame with columns: user_id, gender, age_range, age_group, city
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded file: {filepath}")
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    except pd.errors.ParserError:
        logger.error(f"CSV parsing error in file: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading file {filepath}: {e}")
        return pd.DataFrame()

    try:
        # Clean user_id
        df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').astype('Int64')
        missing = df['user_id'].isna().sum()
        if missing > 0:
            logger.warning(f"{missing} rows with invalid user_id will be dropped.")
        df = df.dropna(subset=['user_id'])
        df['user_id'] = df['user_id'].astype(int)

        # Clean gender
        df['gender'] = pd.to_numeric(df['gender'], errors='coerce').fillna(2).astype(int)
        df['gender'] = df['gender'].apply(lambda x: x if x in [0, 1, 2] else 2)

        # Clean age_range
        df['age_range'] = pd.to_numeric(df['age_range'], errors='coerce').fillna(0).astype(int)
        df['age_range'] = df['age_range'].apply(lambda x: x if x in AGE_MAP else 0)
        df['age_group'] = df['age_range'].map(AGE_MAP)

        # City
        if 'city' in df.columns:
            df['city'] = df['city'].fillna('unknown').astype(str)
        else:
            logger.warning("Missing column 'city' in user info. Filling with 'unknown'.")
            df['city'] = 'unknown'

        # Cluster_id
        if 'cluster_id' in df.columns:
            df['cluster_id'] = pd.to_numeric(df['cluster_id'], errors='coerce').fillna(-1).astype(int)
        else:
            logger.warning("Missing column 'cluster_id' in user info. Filling with -1.")
            df['cluster_id'] = -1

    except Exception as e:
        logger.error(f"Error processing user info data: {e}")
        return pd.DataFrame()

    return df
