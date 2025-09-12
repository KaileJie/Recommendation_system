import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy.sparse import csr_matrix
import logging
import faiss

logger = logging.getLogger(__name__)

def build_faiss_index(S: csr_matrix, nlist: int = 1000) -> faiss.IndexIVFFlat:
    """
    构建FAISS索引用于快速相似度搜索
    
    Args:
        S: 物品相似度矩阵
        nlist: 聚类中心数量
    
    Returns:
        FAISS索引
    """
    logger.info("[build_faiss_index] Building FAISS index...")
    
    # 将相似度矩阵转换为密集矩阵
    S_dense = S.toarray()
    num_items = S_dense.shape[0]
    
    # 调整聚类中心数量：不能超过物品数量
    nlist = min(nlist, num_items)
    logger.info(f"[build_faiss_index] Adjusted nlist to {nlist} (original: {nlist}, items: {num_items})")
    
    # 创建FAISS索引
    dimension = S_dense.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)  # 内积索引
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # 训练索引
    index.train(S_dense.astype('float32'))
    
    # 添加向量到索引
    index.add(S_dense.astype('float32'))
    
    logger.info(f"[build_faiss_index] Index built with {index.ntotal} vectors")
    return index


# 主推荐函数：使用FAISS进行快速推荐
def recommend_for_users_batch(
    UI: csr_matrix,
    S: csr_matrix,
    u2idx: Dict[int, int],
    idx2it: Dict[int, int],
    user_ids: Optional[List[int]] = None,
    n_recs: int = 20,
    batch_size: int = 1024,
    filter_seen: bool = True,
    min_score: Optional[float] = None,
    as_dataframe: bool = True,
    source_tag: str = "itemcf",
    nlist: int = 1000,
    nprobe: int = 10
) -> Union[pd.DataFrame, Dict[int, List[int]]]:
    """
    使用FAISS进行快速推荐的协同过滤算法
    
    Args:
        UI: 用户-物品矩阵
        S: 物品相似度矩阵
        u2idx: 用户ID到索引的映射
        idx2it: 索引到物品ID的映射
        user_ids: 要推荐的用户列表
        n_recs: 每个用户推荐多少个物品
        batch_size: 每批处理多少个用户
        filter_seen: 是否过滤已看过的物品
        min_score: 最小推荐分数
        as_dataframe: 是否返回DataFrame格式
        source_tag: 推荐来源标签
        nlist: FAISS聚类中心数量（越大越精确，但越慢）
        nprobe: 搜索的聚类数量（越大越精确，但越慢）
    
    Returns:
        推荐结果 DataFrame 或字典
    """
    logger.info("[recommend_faiss] Using FAISS for fast recommendation...")
    
    if user_ids is None:
        user_ids = list(u2idx.keys())
    
    # 构建FAISS索引
    index = build_faiss_index(S, nlist)
    index.nprobe = nprobe
    
    # 反转映射
    idx2u = {v: k for k, v in u2idx.items()}
    
    results = []
    
    for start in range(0, len(user_ids), batch_size):
        batch_users = user_ids[start:start + batch_size]
        u_idxs = [u2idx[u] for u in batch_users if u in u2idx]
        
        if not u_idxs:
            continue
        
        # 获取用户历史交互向量
        user_vectors = UI[u_idxs].toarray().astype('float32')
        
        # 使用FAISS搜索相似物品
        scores, indices = index.search(user_vectors, min(n_recs * 3, S.shape[1]))
        
        for row_idx, u_idx in enumerate(u_idxs):
            user_scores = scores[row_idx]
            item_indices = indices[row_idx]
            
            # 过滤已看过的物品和无效索引
            if filter_seen:
                seen_items = set(UI[u_idx].indices)
                valid_mask = [(idx != -1 and idx not in seen_items and idx < len(idx2it)) for idx in item_indices]
                user_scores = user_scores[valid_mask]
                item_indices = item_indices[valid_mask]
            else:
                # 即使不过滤已看过的物品，也要过滤无效索引
                valid_mask = [(idx != -1 and idx < len(idx2it)) for idx in item_indices]
                user_scores = user_scores[valid_mask]
                item_indices = item_indices[valid_mask]
            
            # 过滤低分物品
            if min_score is not None:
                valid_mask = user_scores >= min_score
                user_scores = user_scores[valid_mask]
                item_indices = item_indices[valid_mask]
            
            # 取Top-K
            if len(user_scores) > n_recs:
                top_idx = np.argpartition(user_scores, -n_recs)[-n_recs:]
                user_scores = user_scores[top_idx]
                item_indices = item_indices[top_idx]
            
            # 排序并添加结果
            order = np.argsort(-user_scores)
            for rank, (item_idx, score) in enumerate(zip(item_indices[order], user_scores[order]), start=1):
                results.append((idx2u[u_idx], idx2it[item_idx], rank, float(score), source_tag))
    
    if as_dataframe:
        df = pd.DataFrame(results, columns=["user_id", "item_id", "rank", "score", "source"])
        return df.sort_values(["user_id", "rank"]).reset_index(drop=True)
    else:
        recs_dict = {}
        for uid, iid, _, score, _ in results:
            recs_dict.setdefault(uid, []).append((iid, score))
        return recs_dict
