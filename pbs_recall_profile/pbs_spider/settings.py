# pbs_spider/settings.py

BOT_NAME = "pbs_spider"

SPIDER_MODULES = ["pbs_spider.spiders"]
NEWSPIDER_MODULE = "pbs_spider.spiders"

# 更像浏览器的 UA（可减少被挡）
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = 1

# 稳定性与友好度
COOKIES_ENABLED = False
CONCURRENT_REQUESTS = 8

# 自适应限速（更稳更温柔）
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 5

ITEM_PIPELINES = {
    "pbs_spider.pipelines.MongoDBPipeline": 300,
}

# 仅用于日志展示；实际连接在 dao/mongo_db.py 读取 config.recall_config.MONGO
MONGO_URI = "mongodb://localhost:27017"
MONGO_DATABASE = "pbs_news"
MONGO_COLLECTION = "economy_articles"

REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
