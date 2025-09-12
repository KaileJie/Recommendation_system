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


def main():
    logger.info("🚀 Running inference with Meta-Ranker (GBDT+LR)")

    # 1. 加载训练数据结构（只是为了拿 feature 列，不需要 label）
    df = load_training_data()
    feature_cols = [c for c in df.columns if c not in ["user_id", "item_id", "label"]]

    # 2. 加载模型
    gbm = joblib.load(GBDT_MODEL_PATH)   # ✅ 直接加载 LGBMClassifier
    lr = joblib.load(LR_MODEL_PATH)      # ✅ 加载 LR
    enc = joblib.load(ENCODER_PATH)      # ✅ 加载 OneHotEncoder

    logger.info("✅ Models loaded")

    # 3. 构造推理数据
    X = df[feature_cols]

    # 4. GBDT 叶子索引 → One-hot
    leaf_idx = gbm.predict(X, pred_leaf=True)   # ✅ LGBMClassifier 支持
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


if __name__ == "__main__":
    main()
