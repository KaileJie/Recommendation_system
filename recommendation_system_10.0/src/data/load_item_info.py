import pandas as pd
import logging

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_item_info(filepath: str) -> pd.DataFrame:
    """
    Load item metadata CSV with basic cleaning and logging-based error handling.
    
    Args:
        filepath (str): Path to item_metadata.csv

    Returns:
        pd.DataFrame with columns: item_id, title, url, content
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
        # Clean item_id
        df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce').astype('Int64')
        missing_before = len(df)
        df = df.dropna(subset=['item_id'])
        df['item_id'] = df['item_id'].astype(int)
        logger.info(f"Dropped {missing_before - len(df)} rows with invalid item_id")

        # Clean text fields
        for col in ['title', 'url', 'content']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip()
            else:
                logger.warning(f"Missing column '{col}' in file: {filepath}")
                df[col] = ''

    except Exception as e:
        logger.error(f"Error processing item metadata: {e}")
        return pd.DataFrame()

    return df
