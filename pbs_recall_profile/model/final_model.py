# model/final_model.py
# Fusion of TF-IDF and TextRank keyword scores
# Output: [(keyword, final_score, tfidf_score, textrank_score)]

from typing import List, Tuple, Dict, Iterable, Optional
import math
import numpy as np


def _to_dict(pairs: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in pairs:
        if not k:
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if math.isclose(hi, lo, rel_tol=1e-12, abs_tol=1e-12):
        # 全相等时给常数，避免除零
        return {k: 1.0 for k in scores}
    rng = hi - lo
    return {k: (v - lo) / rng for k, v in scores.items()}


def _zscore_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    mu = float(vals.mean())
    std = float(vals.std(ddof=0))
    if math.isclose(std, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return {k: 0.0 for k in scores}
    return {k: (float(v) - mu) / std for k, v in scores.items()}


def _normalize(scores: Dict[str, float], method: str = "minmax") -> Dict[str, float]:
    method = (method or "minmax").lower()
    if method == "zscore":
        # z 分数可能为负，便于 rank，但用于加权时也 OK
        return _zscore_norm(scores)
    return _minmax_norm(scores)


def _topk(d: Dict[str, float], k: int) -> Dict[str, float]:
    if k <= 0 or not d:
        return {}
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:k])


def _rrf_score(rank: int, k: float = 60.0) -> float:
    # Reciprocal Rank Fusion component
    # score = 1 / (k + rank)
    return 1.0 / (k + rank)


class FinalKeywordExtractor:
    """
    Combine TF-IDF & TextRank by:
      - method='weighted' -> final = alpha * norm(tfidf) + (1 - alpha) * norm(textrank)
      - method='rrf'      -> final = RRF(rank_tfidf) + RRF(rank_textrank)

    normalize: 'minmax' (default) or 'zscore'
    """

    def __init__(
        self,
        tfidf_extractor,
        textrank_extractor,
        alpha: float = 0.6,
        normalize: str = "minmax",
        method: str = "weighted",
        rrf_k: float = 60.0,
    ):
        self.tfidf = tfidf_extractor
        self.tr = textrank_extractor
        self.alpha = float(alpha)
        self.normalize = normalize
        self.method = method
        self.rrf_k = float(rrf_k)

    def extract(
        self, text: str, top_k: int = 20, with_scores: bool = True
    ) -> List[Tuple[str, float, float, float]]:
        # 1) 分别提取原始分数
        tfidf_pairs = self.tfidf.extract(text, top_k=top_k, with_scores=True)  # [(kw, s)]
        tr_pairs = self.tr.extract(text, top_k=top_k, with_scores=True)

        tfidf_raw = _to_dict(tfidf_pairs)
        tr_raw = _to_dict(tr_pairs)

        if self.method.lower() == "rrf":
            # 2a) RRF 使用名次，不用归一化
            tfidf_rank = {kw: r + 1 for r, (kw, _) in enumerate(sorted(tfidf_raw.items(), key=lambda x: x[1], reverse=True))}
            tr_rank = {kw: r + 1 for r, (kw, _) in enumerate(sorted(tr_raw.items(), key=lambda x: x[1], reverse=True))}
            all_keys = set(tfidf_raw) | set(tr_raw)
            fused: Dict[str, float] = {}
            for k in all_keys:
                r1 = tfidf_rank.get(k, len(tfidf_rank) + len(all_keys))  # 缺失给个大 rank
                r2 = tr_rank.get(k, len(tr_rank) + len(all_keys))
                fused[k] = _rrf_score(r1, self.rrf_k) + _rrf_score(r2, self.rrf_k)
        else:
            # 2b) Weighted Sum on normalized scores
            tfidf_n = _normalize(tfidf_raw, self.normalize)
            tr_n = _normalize(tr_raw, self.normalize)
            all_keys = set(tfidf_n) | set(tr_n)
            fused = {}
            a = self.alpha
            for k in all_keys:
                fused[k] = a * tfidf_n.get(k, 0.0) + (1 - a) * tr_n.get(k, 0.0)

        # 3) 截断 & 排序
        fused = _topk(fused, top_k)

        # 4) 输出 A 格式：[(kw, final, tfidf_raw, tr_raw)]
        out: List[Tuple[str, float, float, float]] = []
        for k, f in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            out.append((k, float(f), float(tfidf_raw.get(k, 0.0)), float(tr_raw.get(k, 0.0))))

        return out if with_scores else [(k, s) for k, s, *_ in out]

    # 可选：动态改策略/参数
    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def set_method(self, method: str) -> None:
        self.method = method

    def set_normalize(self, normalize: str) -> None:
        self.normalize = normalize
