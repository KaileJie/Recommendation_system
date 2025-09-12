# src/recall/fusion/fusion_recall.py

import os
import pandas as pd
import logging
from collections import defaultdict

# ----------- Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------- Config -----------
OUTPUT_DIR = "output/fusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 三个召回结果
ITEMCF_PATH = "output/item_cf/itemcf_recall.csv"
KEYWORD_PATH = "output/keyword/keyword_hybrid_recall.csv"
DNN_PATH = "output/youtube_dnn/youtube_dnn_faiss_recall.csv"

# 输出文件
FUSION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fusion_recall.csv")
TOP_K = 50


def load_recall_results():
    """加载三个召回结果"""
    recall_dfs = {}
    
    # 加载 ItemCF 结果
    if os.path.exists(ITEMCF_PATH):
        df = pd.read_csv(ITEMCF_PATH)
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df = df[["user_id", "item_id", "score"]].rename(columns={"score": "cf_score"})
        recall_dfs["itemcf"] = df
        logger.info(f"✅ Loaded ItemCF: {df.shape}")
    else:
        logger.warning(f"⚠️ ItemCF file not found: {ITEMCF_PATH}")
    
    # 加载 Keyword 结果
    if os.path.exists(KEYWORD_PATH):
        df = pd.read_csv(KEYWORD_PATH)
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df = df[["user_id", "item_id", "score"]].rename(columns={"score": "keyword_score"})
        recall_dfs["keyword"] = df
        logger.info(f"✅ Loaded Keyword: {df.shape}")
    else:
        logger.warning(f"⚠️ Keyword file not found: {KEYWORD_PATH}")
    
    # 加载 YouTube DNN 结果
    if os.path.exists(DNN_PATH):
        df = pd.read_csv(DNN_PATH)
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df = df[["user_id", "item_id", "score"]].rename(columns={"score": "dnn_score"})
        recall_dfs["youtube_dnn"] = df
        logger.info(f"✅ Loaded YouTube DNN: {df.shape}")
    else:
        logger.warning(f"⚠️ YouTube DNN file not found: {DNN_PATH}")
    
    return recall_dfs


def normalize_scores(df):
    """对召回分数进行归一化"""
    score_cols = [col for col in df.columns if col.endswith('_score')]
    
    for col in score_cols:
        if df[col].max() > df[col].min():
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[col] = 0.0
    
    logger.info(f"✅ Normalized score columns: {score_cols}")
    return df


def weighted_fusion(recall_dfs, weights=None):
    """加权融合召回结果"""
    if weights is None:
        weights = {"cf_score": 0.3, "keyword_score": 0.3, "dnn_score": 0.4}
    
    logger.info(f"Using fusion weights: {weights}")
    
    # 合并所有召回结果
    merged_df = None
    for source, df in recall_dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=["user_id", "item_id"], how="outer")
    
    # 填充缺失值
    merged_df = merged_df.fillna(0.0)
    
    # 归一化分数
    merged_df = normalize_scores(merged_df)
    
    # 计算融合分数
    merged_df["fusion_score"] = 0.0
    for score_col, weight in weights.items():
        if score_col in merged_df.columns:
            merged_df["fusion_score"] += merged_df[score_col] * weight
    
    return merged_df


def rank_and_filter(df, top_k=TOP_K):
    """对每个用户排序并取Top-K"""
    # 按用户分组并排序
    ranked_df = df.sort_values(["user_id", "fusion_score"], ascending=[True, False])
    
    # 为每个用户取Top-K
    result_df = ranked_df.groupby("user_id").head(top_k).reset_index(drop=True)
    
    # 添加排名
    result_df["rank"] = result_df.groupby("user_id").cumcount() + 1
    
    return result_df


def main():
    logger.info("🚀 Running Fusion Recall...")
    
    # 加载召回结果
    recall_dfs = load_recall_results()
    
    if not recall_dfs:
        logger.error("❌ No recall results found!")
        return
    
    # 加权融合
    fused_df = weighted_fusion(recall_dfs)
    logger.info(f"✅ Fusion completed: {fused_df.shape}")
    
    # 排序和过滤
    result_df = rank_and_filter(fused_df)
    logger.info(f"✅ Ranking completed: {result_df.shape}")
    
    # 保存结果
    result_df.to_csv(FUSION_OUTPUT_PATH, index=False)
    logger.info(f"📦 Fusion results saved to: {FUSION_OUTPUT_PATH}")
    
    # 统计信息
    logger.info(f"📊 Total users: {result_df['user_id'].nunique()}")
    logger.info(f"📊 Total recommendations: {len(result_df)}")
    logger.info(f"📊 Average recommendations per user: {len(result_df) / result_df['user_id'].nunique():.2f}")


if __name__ == "__main__":
    main()
