# pbs_spider/items.py
import scrapy

class PbsEconomyItem(scrapy.Item):
    # Basic metadata
    title = scrapy.Field()
    description = scrapy.Field()
    category = scrapy.Field()
    url = scrapy.Field()
    publish_date = scrapy.Field()
    scrape_date = scrapy.Field()

    # Content analysis
    content = scrapy.Field()
    total_words = scrapy.Field()

    # Placeholder metrics (for future use)
    views = scrapy.Field()
    likes = scrapy.Field()

