# dao/redis_cache.py

import redis
from config.recall_config import REDIS


class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS["HOST"],
            port=REDIS["PORT"],
            db=REDIS["DB"],
            decode_responses=True
        )

    def save_inverted_index(self, base_key, inverted_index, expire=None):
        """
        将倒排索引保存为多个 Redis Set（每个关键词一个 key）
        """
        pipeline = self.client.pipeline()

        for keyword, article_list in inverted_index.items():
            keyword = keyword.lower()
            if not article_list:
                continue
            redis_key = f"{base_key}:{keyword}"
            pipeline.delete(redis_key)
            pipeline.sadd(redis_key, *article_list)
            if expire:
                pipeline.expire(redis_key, expire)

        pipeline.execute()

    def load_keyword_articles(self, base_key, keyword):
        """
        加载某个关键词对应的文章列表（Set）
        """
        redis_key = f"{base_key}:{keyword.lower()}"
        return self.client.smembers(redis_key)

    def delete_index_by_keyword(self, base_key, keyword):
        """
        删除某个关键词的索引
        """ 
        redis_key = f"{base_key}:{keyword.lower()}"
        self.client.delete(redis_key)

    def delete_all_index_keys(self, base_key, keywords):
        """
        批量删除一组关键词对应的倒排索引
        """ 
        pipeline = self.client.pipeline()
        for keyword in keywords:
            redis_key = f"{base_key}:{keyword.lower()}"
            pipeline.delete(redis_key)
        pipeline.execute()

    def clear_all_keys_with_prefix(self, base_key):
        """
        删除以 base_key 开头的所有 Redis key，例如：pbs:index:tfidf:*
        """
        pattern = f"{base_key}:*"
        keys = list(self.client.scan_iter(match=pattern))
        if keys:
            self.client.delete(*keys)
