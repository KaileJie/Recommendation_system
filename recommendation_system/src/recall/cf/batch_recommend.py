import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

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
    source_tag: str = "itemcf"
) -> Union[pd.DataFrame, Dict[int, List[int]]]:
    """
    Batch item-based CF recommendation.

    Returns:
        pd.DataFrame with columns: user_id, item_id, rank, score, source
        or dict[user_id -> list[(item_id, score)]]
    """
    logger.info("[recommend_batch] Running in batches of %d...", batch_size)

    if user_ids is None:
        user_ids = list(u2idx.keys())

    results = []

    for start in range(0, len(user_ids), batch_size):
        batch_users = user_ids[start:start + batch_size]
        u_idxs = [u2idx[u] for u in batch_users if u in u2idx]

        if not u_idxs:
            continue

        scores = UI[u_idxs].dot(S)

        if filter_seen:
            for row_idx, u_idx in enumerate(u_idxs):
                seen_items = UI[u_idx].indices
                if len(seen_items) > 0:
                    scores[row_idx, seen_items] = 0.0

        for row_idx, u_idx in enumerate(u_idxs):
            row = scores.getrow(row_idx)
            if row.nnz == 0:
                continue
            vals = row.data
            cols = row.indices

            if min_score is not None:
                mask = vals >= min_score
                vals, cols = vals[mask], cols[mask]

            if len(vals) > n_recs:
                top_idx = np.argpartition(vals, -n_recs)[-n_recs:]
                vals, cols = vals[top_idx], cols[top_idx]

            order = np.argsort(-vals)
            for rank, (c, v) in enumerate(zip(cols[order], vals[order]), start=1):
                results.append((u_idx, idx2it[c], rank, float(v), source_tag))

    if as_dataframe:
        df = pd.DataFrame(results, columns=["user_id", "item_id", "rank", "score", "source"])
        return df.sort_values(["user_id", "rank"]).reset_index(drop=True)
    else:
        recs_dict = {}
        for uid, iid, _, score, _ in results:
            recs_dict.setdefault(uid, []).append((iid, score))
        return recs_dict
