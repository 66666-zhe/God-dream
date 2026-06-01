import scrapy


class ZhihuHotItem(scrapy.Item):
    rank = scrapy.Field()
    title = scrapy.Field()
    excerpt = scrapy.Field()
    heat = scrapy.Field()
    url = scrapy.Field()
    answer_count = scrapy.Field()
    follower_count = scrapy.Field()
