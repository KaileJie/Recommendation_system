# model/textrank_model.py
"""
TextRank-based keyword extractor module
统一接口：extract(text, top_k=10, with_scores=True) -> [(keyword, score)]
"""

import re
from typing import List, Tuple, Optional
from config.recall_config import ARTIFACTS

try:
    from gensim.summarization import keywords as textrank_keywords
except Exception as e:
    textrank_keywords = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


class TextRankKeywordExtractor:
    def __init__(self, ratio: float = 0.2, lemmatize: bool = True, deacc: bool = True):
        self.ratio = ratio
        self.lemmatize = lemmatize
        self.deacc = deacc

    def extract(self, text: str, top_k: int = 10, with_scores: bool = True) -> List[Tuple[str, float]]:
        if _IMPORT_ERR is not None or textrank_keywords is None:
            raise ImportError(f"gensim.summarization.keywords not available: {_IMPORT_ERR}")

        if not isinstance(text, str) or not text.strip():
            return []

        clean = self._clean_text(text)

        try:
            pairs = textrank_keywords(
                clean,
                words=top_k,
                split=True,
                scores=True,
                lemmatize=self.lemmatize,
                deacc=self.deacc,
            )
            out: List[Tuple[str, float]] = []
            for kw, sc in pairs:
                kw = (kw or "").strip()
                if kw:
                    try:
                        out.append((kw, float(sc)))
                    except Exception:
                        continue

            return out if with_scores else [kw for kw, _ in out]

        except Exception as e:
            print(f"[TextRank] Keyword extraction failed: {e}")
            return []

    def extract_keywords(self, text: str, topn: int = 10) -> List[str]:
        pairs = self.extract(text, top_k=topn, with_scores=True)
        return [kw for kw, _ in pairs]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return text.strip()


# --------- 全局工具函数（带缓存） ---------
_textrank_extractor: Optional[TextRankKeywordExtractor] = None

def extract_textrank_keywords(text: str, topn: int = 10) -> List[str]:
    global _textrank_extractor
    if _textrank_extractor is None:
        _textrank_extractor = TextRankKeywordExtractor()
    return _textrank_extractor.extract_keywords(text, topn=topn)
