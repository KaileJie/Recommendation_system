# pbs_spider/pipelines.py
from datetime import datetime
from itemadapter import ItemAdapter
from dao.mongo_db import MongoDAO

class MongoDBPipeline:
    def __init__(self, mongo_uri=None, mongo_db=None, mongo_collection=None):
        # 这些字段只用于日志展示；真正的连接在 MongoDAO 内部按配置完成
        self.mongo_uri = mongo_uri or "from config"
        self.mongo_db = mongo_db or "from config"
        self.mongo_collection = mongo_collection or "from config"
        self.dao = None

    @classmethod
    def from_crawler(cls, crawler):
        # 兼容旧的 settings，主要用于日志；真实连接参数由 MongoDAO 读取 config.recall_config
        return cls(
            mongo_uri=crawler.settings.get('MONGO_URI'),
            mongo_db=crawler.settings.get('MONGO_DATABASE'),
            mongo_collection=crawler.settings.get('MONGO_COLLECTION')
        )

    def open_spider(self, spider):
        spider.logger.info(f"[MongoDBPipeline] Connecting to MongoDB (via config.recall_config)")
        self.dao = MongoDAO()  # ✅ 无参初始化，读 config.recall_config.MONGO
        spider.logger.info(f"[MongoDBPipeline] Connected. DB/Collection from config.recall_config")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        doc = adapter.asdict()

        # 必要字段
        url = (doc.get("url") or "").strip()
        if not url:
            spider.logger.warning("[MongoDBPipeline] Skip item: missing 'url'")
            return item

        # 规范字段（可保留，也可删除以最小改动）
        content = doc.get("content") or ""
        if not isinstance(doc.get("total_words"), int):
            doc["total_words"] = len(content.split())

        for k in ("publish_date", "scrape_date"):
            v = doc.get(k)
            if isinstance(v, datetime):
                doc[k] = v.isoformat()

        # 写入（按 url upsert）
        try:
            result = self.dao.upsert_item({'url': url}, doc)
            if getattr(result, "upserted_id", None):
                spider.logger.info(f"[MongoDBPipeline] Inserted: {doc.get('title') or url}")
            else:
                spider.logger.info(f"[MongoDBPipeline] Updated: {doc.get('title') or url}")
        except Exception as e:
            spider.logger.error(f"[MongoDBPipeline] Upsert failed for url={url}: {e}")

        return item

    def close_spider(self, spider):
        if self.dao:
            self.dao.close()
        spider.logger.info("[MongoDBPipeline] Closed MongoDB connection.")
