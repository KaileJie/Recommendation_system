# model/tfidf_model.py
import os
import joblib
import numpy as np
from typing import List, Tuple, Iterable, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from config.recall_config import ARTIFACTS

DEFAULT_MODEL_PATH = ARTIFACTS["TFIDF_MODEL_PATH"]


class TFIDFKeywordExtractor:
    def __init__(self, max_features: int = 1000, stop_words: Optional[str] = "english"):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
        self.feature_names: List[str] = []
        self._tfidf_matrix = None  # 仅用于调试/分析；不参与推理逻辑

    def fit(self, documents: Iterable[str]) -> None:
        """拟合 TF-IDF 模型并缓存特征名"""
        self._tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

    def extract(self, text: str, top_k: int = 10, with_scores: bool = True) -> Union[List[Tuple[str, float]], List[str]]:
        """统一接口：返回 [(keyword, score)]，score 为该文档内的 TF-IDF 权重"""
        if not hasattr(self.vectorizer, "vocabulary_") or self.vectorizer.vocabulary_ is None:
            raise ValueError("TF-IDF model is not fitted yet. Call fit(corpus) or load(model_path).")

        tfidf_vector = self.vectorizer.transform([text])
        row = tfidf_vector.toarray().ravel()
        if row.size == 0:
            return []

        top_idx = np.argsort(row)[::-1][:top_k]

        pairs: List[Tuple[str, float]] = []
        for i in top_idx:
            score = float(row[i])
            if score <= 0:
                continue
            if 0 <= i < len(self.feature_names):
                kw = self.feature_names[i]
                if kw:
                    pairs.append((kw, score))

        return pairs if with_scores else [kw for kw, _ in pairs]

    def transform(self, text: str, topk: int = 10) -> List[str]:
        """向后兼容：仅返回关键词列表（不含分数）"""
        pairs = self.extract(text, top_k=topk, with_scores=True)
        return [kw for kw, _ in pairs]

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        """仅保存 vectorizer；加载时会自动恢复 feature_names"""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        joblib.dump(self.vectorizer, path)

    def load(self, path: str = DEFAULT_MODEL_PATH) -> None:
        """加载已训练的 vectorizer，并同步 feature_names"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        self.vectorizer = joblib.load(path)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()


# ---------- 工具函数 ----------

def load_fitted_extractor(
    corpus: Optional[Iterable[str]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    max_features: int = 1000,
    force_retrain: bool = False,
    stop_words: Optional[str] = "english",
) -> TFIDFKeywordExtractor:
    """加载训练好的 TF-IDF 提取器；若不存在且提供了语料，则训练并保存"""
    extractor = TFIDFKeywordExtractor(max_features=max_features, stop_words=stop_words)

    if force_retrain or (corpus and not os.path.exists(model_path)):
        if not corpus:
            raise ValueError("Corpus is required for training the TF-IDF model.")
        extractor.fit(corpus)
        extractor.save(model_path)
        print(f"TF-IDF model trained and saved to {model_path}")
    else:
        extractor.load(model_path)
        print(f"TF-IDF model loaded from {model_path}")

    return extractor
