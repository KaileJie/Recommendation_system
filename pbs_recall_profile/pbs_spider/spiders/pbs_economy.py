# pbs_spider/spiders/pbs_economy.py
import scrapy
import re
from datetime import datetime
from pbs_spider.items import PbsEconomyItem
from utils.text import get_text_safe, extract_clean_content, count_words


class PbsEconomySpider(scrapy.Spider):
    name = "pbs_economy"
    allowed_domains = ["pbs.org"]
    start_urls = ["https://www.pbs.org/newshour/economy"]

    def __init__(self, max_pages=10, *args, **kwargs):
        """
        Initialize the spider with a dynamic max_pages value.
        Allow `-a max_pages=5` when running the spider.
        """
        super().__init__(*args, **kwargs)
        self.max_pages = int(max_pages)
        self.page_count = 0

    def parse(self, response):
        self.page_count += 1
        self.logger.info(f"Scraping LIST PAGE {self.page_count}: {response.url}")

        if self.page_count > self.max_pages:
            self.logger.info(f"Reached max_pages={self.max_pages}. Stopping crawl.")
            return

        # extract article cards
        articles = response.css('article.card-horiz')
        self.logger.info(f"Found {len(articles)} article elements on this page")

        for article in articles:
            relative_url = article.css('a::attr(href)').get()
            if not relative_url:
                continue

            url = response.urljoin(relative_url)
            title_snippet = get_text_safe(article.css('a span::text'))
            byline = get_text_safe(article.css('p.card-horiz__byline::text'))

            yield response.follow(
                url,
                callback=self.parse_detail,
                meta={
                    'url': url,
                    'list_title': title_snippet,
                    'byline': byline
                }
            )

        # handle pagination
        next_page = response.css('a[rel="next"]::attr(href)').get()
        if next_page:
            next_page_url = response.urljoin(next_page)
            self.logger.info(f"Following NEXT PAGE: {next_page_url}")
            yield response.follow(next_page_url, callback=self.parse)

    def extract_type_from_url(self, url: str) -> str:
        """
        Extract article category from URL path (e.g., 'economy', 'politics').
        Unified to lowercase for consistency.
        """
        match = re.search(r"newshour/([^/]+)/", url)
        return match.group(1).lower() if match else "unknown"

    def parse_detail(self, response):
        self.logger.info(f"Scraping DETAIL PAGE: {response.url}")

        try:
            # 1) title
            title = get_text_safe(response.css('meta[property="og:title"]::attr(content)'))
            if not title:
                title = get_text_safe(response.css('h1.article__title::text'))

            # 2) description
            description = get_text_safe(response.css('meta[name="description"]::attr(content)'))

            # 3) url & category
            url = response.meta.get('url') or response.url
            category = self.extract_type_from_url(url)

            # 4) content（多个备选容器，取到的文本会做清洗）
            content = extract_clean_content(response, [
                'div.body-text p::text',
                'div.article-body p::text',
                'div.article__body p::text',
                'section.article-body p::text',
                'article p::text',
            ])

            # 5) total_words
            total_words = count_words(content)

            # 6) publish_date（原始字符串；pipeline 会保持为字符串或做规范化）
            publish_date = get_text_safe(
                response.css('meta[property="article:published_time"]::attr(content)') or
                response.css('meta[name="date"]::attr(content)') or
                response.css('span.date::text')
            )

            # 7) scrape_date（ISO）
            scrape_date = datetime.utcnow().isoformat()

            # 8/9) placeholder metrics
            views = 0
            likes = 0

            # Build item 
            item = PbsEconomyItem(
                title=title,
                description=description,
                category=category,
                content=content,
                url=url,
                total_words=total_words,
                publish_date=publish_date,
                scrape_date=scrape_date,
                views=views,
                likes=likes
            )

            yield item

        except Exception as e:
            self.logger.error(f"❌ Error parsing detail page {response.url}: {e}", exc_info=True)
