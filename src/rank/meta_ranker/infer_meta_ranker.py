# src/rank/meta_ranker/infer_meta_ranker.py

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# -------- Logger --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------- Config --------
OUTPUT_DIR = "output/meta_ranker"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, "training_data.json")
GBDT_MODEL_PATH = os.path.join(OUTPUT_DIR, "gbdt_model.pkl")
LR_MODEL_PATH = os.path.join(OUTPUT_DIR, "lr_model.pkl")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "onehot_encoder.pkl")

INFER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "meta_ranker_infer.json")
TOP_K = 20


def load_training_data():
    """加载和 build_training_data 相同结构的数据"""
    records = []
    with open(TRAIN_DATA_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame([{
        "user_id": r["user_id"],
        "item_id": r["item_id"],
        "label": r["label"],
        "score_itemcf": r["scores"]["itemcf"],
        "score_keyword": r["scores"]["keyword"],
        "score_dnn": r["scores"]["youtube_dnn"],
        **r["features"]
    } for r in records])

    return df


def load_models():
    """加载训练好的模型"""
    try:
        gbm = joblib.load(GBDT_MODEL_PATH)
        lr = joblib.load(LR_MODEL_PATH)
        enc = joblib.load(ENCODER_PATH)
        logger.info("✅ Models loaded successfully")
        return gbm, lr, enc
    except FileNotFoundError as e:
        logger.error(f"❌ Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise


def validate_input_data(df):
    """验证输入数据的完整性"""
    required_cols = ["user_id", "item_id", "score_itemcf", "score_keyword", "score_dnn"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("❌ Input data is empty")
    
    logger.info(f"✅ Input data validated: {df.shape[0]} samples, {df['user_id'].nunique()} users")


def infer_from_recall_data(recall_data_path, user_profile_path, top_k=20):
    """
    从新的召回结果进行推理
    
    Args:
        recall_data_path: 召回结果文件路径
        user_profile_path: 用户画像文件路径
        top_k: 返回的推荐数量
    """
    logger.info(f"🚀 Running inference from recall data: {recall_data_path}")
    
    try:
        # 1. 加载召回结果
        recall_df = pd.read_csv(recall_data_path)
        logger.info(f"✅ Loaded recall data: {recall_df.shape}")
        
        # 2. 加载用户画像
        from src.data.load_user_profile import load_user_profile_feature
        profile_df = load_user_profile_feature(user_profile_path)
        logger.info(f"✅ Loaded user profile: {profile_df.shape}")
        
        # 3. 合并数据
        merged_df = recall_df.merge(profile_df, on="user_id", how="left")
        
        # 4. 处理缺失值（使用与训练时相同的策略）
        from src.rank.meta_ranker.build_training_data import handle_missing_features
        merged_df = handle_missing_features(merged_df)
        
        # 5. 加载模型
        gbm, lr, enc = load_models()
        
        # 6. 准备特征
        feature_cols = [c for c in merged_df.columns if c not in ["user_id", "item_id"]]
        X = merged_df[feature_cols]
        
        # 7. 预测
        leaf_idx = gbm.predict(X, pred_leaf=True)
        leaf_idx = np.array(leaf_idx, dtype=np.int32)
        X_enc = enc.transform(leaf_idx)
        final_scores = lr.predict_proba(X_enc)[:, 1]
        
        # 8. 生成推荐结果
        results = []
        for row, score in zip(merged_df.itertuples(index=False), final_scores):
            record = {
                "user_id": row.user_id,
                "item_id": row.item_id,
                "final_score": float(score),
                "scores": {
                    "itemcf": float(getattr(row, 'score_itemcf', 0.0)),
                    "keyword": float(getattr(row, 'score_keyword', 0.0)),
                    "youtube_dnn": float(getattr(row, 'score_dnn', 0.0))
                }
            }
            results.append(record)
        
        # 9. 按用户分组排序
        output = []
        df_results = pd.DataFrame(results)
        for uid, group in df_results.groupby("user_id"):
            top_items = group.sort_values("final_score", ascending=False).head(top_k)
            output.extend(top_items.to_dict(orient="records"))
        
        logger.info(f"📊 Generated {len(output)} recommendations for {df_results['user_id'].nunique()} users")
        return output
        
    except Exception as e:
        logger.error(f"❌ Inference from recall data failed: {e}")
        raise


def main():
    logger.info("🚀 Running inference with Meta-Ranker (GBDT+LR)")

    try:
        # 1. 加载训练数据结构（只是为了获取特征列结构）
        # 注意：实际推理时应该使用新的召回结果，这里仅用于演示
        df = load_training_data()
        feature_cols = [c for c in df.columns if c not in ["user_id", "item_id", "label"]]
        
        # 验证输入数据
        validate_input_data(df)

        # 2. 加载模型
        gbm, lr, enc = load_models()

        # 3. 构造推理数据
        X = df[feature_cols]

        # 4. GBDT 叶子索引 → One-hot
        leaf_idx = gbm.predict(X, pred_leaf=True)
        leaf_idx = np.array(leaf_idx, dtype=np.int32)
        X_enc = enc.transform(leaf_idx)

        # 5. LR 输出最终分数
        final_scores = lr.predict_proba(X_enc)[:, 1]

        # 6. 组装 JSON 输出
        results = []
        for row, score in zip(df.itertuples(index=False), final_scores):
            record = {
                "user_id": row.user_id,
                "item_id": row.item_id,
                "final_score": float(score),
                "scores": {
                    "itemcf": float(row.score_itemcf),
                    "keyword": float(row.score_keyword),
                    "youtube_dnn": float(row.score_dnn)
                },
                "features": {
                    "age_range": row.age_range,
                    "gender": row.gender,
                    "city": row.city,
                    "cluster_id": row.cluster_id,
                    "recency": row.recency,
                    "frequency": row.frequency,
                    "actions_per_active_day_30d": row.actions_per_active_day_30d,
                    "monetary": row.monetary,
                    "rfm_score": row.rfm_score
                }
            }
            results.append(record)

        # 按 user_id 分组 + 排序取 Top-K
        output = []
        df_results = pd.DataFrame(results)
        for uid, group in df_results.groupby("user_id"):
            top_items = group.sort_values("final_score", ascending=False).head(TOP_K)
            output.extend(top_items.to_dict(orient="records"))

        # 保存 JSON
        with open(INFER_OUTPUT_PATH, "w") as f:
            for r in output:
                f.write(json.dumps(r) + "\n")

        logger.info(f"📦 Inference results saved to {INFER_OUTPUT_PATH}")
        logger.info(f"📊 Processed {len(output)} recommendations for {df_results['user_id'].nunique()} users")

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
