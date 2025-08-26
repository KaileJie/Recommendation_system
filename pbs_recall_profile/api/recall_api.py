# api/recall_api.py
from fastapi import FastAPI, Query, HTTPException
from typing import List, Optional

from config.recall_config import STRATEGIES
from service.recall_service import recall_by_query

app = FastAPI(title="Recall API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recall", response_model=dict)
def recall(
    query: str = Query(..., min_length=1, description="Search keywords, space-separated"),
    strategy: str = Query("final", description=f"One of {STRATEGIES}"),
    limit: Optional[int] = Query(20, ge=1, le=1000, description="Max URLs to return"),
):
    s = strategy.lower()
    if s not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unsupported strategy '{strategy}'. Allowed: {STRATEGIES}")

    try:
        urls: List[str] = recall_by_query(query, strategy=s, limit=limit)
        # 只返回 URL 列表
        return {"urls": urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
