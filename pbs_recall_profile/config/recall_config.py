# config/recall_config.py

# ===== Redis =====
REDIS = {
    "HOST": "localhost",
    "PORT": 6379,
    "DB": 0,
    # 倒排索引 key 前缀：最终 key 形如 pbs:index:<keyword>
    "INDEX_PREFIX": "pbs:index",
    # 可选：键过期（秒）。不想过期就设为 None
    "EXPIRE": None,
}
STRATEGIES = ("tfidf", "textrank", "final")  # 受支持的策略

# ===== MongoDB =====
MONGO = {
    "URI": "mongodb://localhost:27017",
    "DATABASE": "pbs_news",
    "COLLECTION": "economy_articles",
}

# ===== Keyword Fusion / Model Hyperparams =====
FUSION = {
    "TOP_K": 20,           # 生成候选关键词数量
    "ALPHA": 0.6,          # TF-IDF 权重（1-alpha 给 TextRank）
    "NORMALIZE": "minmax", # "minmax" | "zscore"
    "METHOD": "weighted",  # "weighted" | "rrf"
    "RRF_K": 60.0,
}

# 模型产物（如需，当前 TF-IDF 会在首次 fit 时保存/加载）
ARTIFACTS = {
    "TFIDF_MODEL_PATH": "artifacts/tfidf_model.pkl",
}


