# dao/mongo_db.py

import pymongo
from datetime import datetime, timedelta
from config.recall_config import MONGO  # 引入配置

class MongoDAO:
    def __init__(self):
        self.client = pymongo.MongoClient(MONGO["URI"])
        self.db = self.client[MONGO["DATABASE"]]
        self.collection = self.db[MONGO["COLLECTION"]]

    def upsert_item(self, query, update_fields):
        return self.collection.update_one(query, {"$set": update_fields}, upsert=True)

    def close(self):
        self.client.close()


class ArticleDAO:
    def __init__(self):
        self.client = pymongo.MongoClient(MONGO["URI"])
        self.collection = self.client[MONGO["DATABASE"]][MONGO["COLLECTION"]]

    def get_recent_articles(self, days=90, limit=100):
        """
        Retrieve articles published in the last `days` days from MongoDB.
        Returns a list of documents with 'url' and 'content'.
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = {"publish_date": {"$gte": cutoff_date.isoformat()}}
        projection = {"url": 1, "content": 1, "final_keywords": 1, "title": 1, "publish_date": 1}

        cursor = self.collection.find(query, projection).limit(limit)
        return [doc for doc in cursor if "url" in doc and "content" in doc]

    def update_article_keywords(self, article_id, tfidf_keywords, textrank_keywords, final_keywords):
        self.collection.update_one(
            {"_id": article_id},
            {
                "$set": {
                    "tfidf_keywords": tfidf_keywords,
                    "textrank_keywords": textrank_keywords,
                    "final_keywords": final_keywords
                }
            }
        )

    def close(self):
        self.client.close()
