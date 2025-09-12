import os
import json
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import pickle

# -------- Logger --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------- Config --------
OUTPUT_DIR = "output/meta_ranker"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, "training_data.json")
TOP_K = 20


def load_training_data():
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


def evaluate_ranking(y_true, y_pred, user_ids, item_ids, k=TOP_K):
    """计算 Recall@K 和 Precision@K"""
    df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "label": y_true,
        "score": y_pred
    })

    results = []
    for uid, group in df.groupby("user_id"):
        gt_items = set(group[group["label"] == 1]["item_id"])
        if not gt_items:
            continue
        rec_items = group.sort_values("score", ascending=False).head(k)["item_id"].tolist()
        hit = len(set(rec_items) & gt_items)
        recall = hit / len(gt_items)
        precision = hit / k
        results.append((recall, precision))

    if results:
        avg_recall = sum(r for r, _ in results) / len(results)
        avg_precision = sum(p for _, p in results) / len(results)
    else:
        avg_recall, avg_precision = 0.0, 0.0

    return avg_recall, avg_precision


def main():
    logger.info("🚀 Training Meta-Ranker (GBDT + LR)")

    df = load_training_data()
    logger.info(f"✅ Loaded training data: {df.shape}")

    features = [c for c in df.columns if c not in ["user_id", "item_id", "label"]]
    X = df[features]
    y = df["label"]

    # 拆分训练/验证集
    X_train, X_val, y_train, y_val, train_df, val_df = train_test_split(
        X, y, df[["user_id", "item_id"]], test_size=0.2, random_state=42
    )

    # -------- LightGBM --------
    gbm = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=64,
        random_state=42
    )
    gbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    # -------- GBDT+LR --------
    # 提取 GBDT 的叶子索引
    X_train_gbdt = gbm.predict(X_train, pred_leaf=True)
    X_val_gbdt = gbm.predict(X_val, pred_leaf=True)

    # One-hot 编码
    enc = OneHotEncoder(handle_unknown="ignore")
    X_train_enc = enc.fit_transform(X_train_gbdt)
    X_val_enc = enc.transform(X_val_gbdt)

    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_train_enc, y_train)

    # -------- Evaluation --------
    # LightGBM
    gbm_val_pred = gbm.predict_proba(X_val)[:, 1]
    gbm_auc = roc_auc_score(y_val, gbm_val_pred)
    gbm_recall, gbm_precision = evaluate_ranking(y_val, gbm_val_pred, val_df["user_id"], val_df["item_id"])

    # GBDT+LR
    lr_val_pred = lr.predict_proba(X_val_enc)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_val_pred)
    lr_recall, lr_precision = evaluate_ranking(y_val, lr_val_pred, val_df["user_id"], val_df["item_id"])

    logger.info(f"📊 LightGBM - AUC: {gbm_auc:.4f}, Recall@{TOP_K}: {gbm_recall:.4f}, Precision@{TOP_K}: {gbm_precision:.4f}")
    logger.info(f"📊 GBDT+LR  - AUC: {lr_auc:.4f}, Recall@{TOP_K}: {lr_recall:.4f}, Precision@{TOP_K}: {lr_precision:.4f}")

    # -------- Save --------
    joblib.dump(gbm, os.path.join(OUTPUT_DIR, "gbdt_model.pkl"))   # ✅ 保存 LGBMClassifier
    joblib.dump(lr, os.path.join(OUTPUT_DIR, "lr_model.pkl"))      # ✅ 保存 LR
    joblib.dump(enc, os.path.join(OUTPUT_DIR, "onehot_encoder.pkl"))  # ✅ 保存 OneHotEncoder

    logger.info("📦 Models saved to output/meta_ranker")



if __name__ == "__main__":
    main()
