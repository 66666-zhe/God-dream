import re
import scrapy
from zhihu_hot.items import ZhihuHotItem


class ZhihuHotSpider(scrapy.Spider):
    name = "zhihu_hot"
    allowed_domains = ["tophub.today"]
    start_urls = ["https://tophub.today/n/mproPpoq6O"]

    custom_settings = {
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    }

    def parse(self, response):
        rows = response.xpath('//table[1]//tr')
        self.logger.info(f"Found {len(rows)} hot items")

        for idx, row in enumerate(rows, start=1):
            cells = row.xpath('.//td')
            if len(cells) < 3:
                continue

            # Extract rank
            rank_text = cells[0].xpath('string(.)').get().strip().rstrip('.')
            try:
                rank = int(rank_text)
            except (ValueError, TypeError):
                rank = idx

            # Extract title from the <a> tag
            title = cells[2].xpath('.//a/text()').get('').strip()

            # Extract link
            href = cells[2].xpath('.//a/@href').get('')
            if href and not href.startswith('http'):
                href = 'https://www.zhihu.com' + href

            # Extract heat: find text matching "N 万/亿 热度" pattern
            all_text = cells[2].xpath('string(.)').get('')
            heat_match = re.search(r'(\d+[\d.]*\s*万热度)', all_text)
            heat = heat_match.group(1) if heat_match else ""

            item = ZhihuHotItem()
            item["rank"] = rank
            item["title"] = title
            item["excerpt"] = ""
            item["heat"] = heat
            item["url"] = href
            item["answer_count"] = 0
            item["follower_count"] = 0
            yield item
