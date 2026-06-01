import csv
import json
from datetime import datetime

import pymysql


class JsonPipeline:
    def open_spider(self, spider):
        self.items = []

    def process_item(self, item, spider):
        self.items.append(dict(item))
        return item

    def close_spider(self, spider):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zhihu_hot_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
        spider.logger.info(f"JSON: saved {len(self.items)} items to {filename}")


class CsvPipeline:
    def open_spider(self, spider):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"zhihu_hot_{timestamp}.csv"
        self.file = open(self.filename, "w", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=[
            "rank", "title", "excerpt", "heat", "url", "answer_count", "follower_count"
        ])
        self.writer.writeheader()

    def process_item(self, item, spider):
        self.writer.writerow(dict(item))
        return item

    def close_spider(self, spider):
        self.file.close()
        spider.logger.info(f"CSV: saved to {self.filename}")


class MySQLPipeline:
    def open_spider(self, spider):
        self.conn = pymysql.connect(
            host="192.168.86.253",
            port=3306,
            user="student",
            password="student24",
            database="student24",
            charset="utf8mb4",
        )
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS zhihu_hot (
                id INT AUTO_INCREMENT PRIMARY KEY,
                `rank` INT,
                title VARCHAR(500),
                excerpt TEXT,
                heat VARCHAR(100),
                url VARCHAR(500),
                answer_count INT,
                follower_count INT,
                crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def process_item(self, item, spider):
        self.cursor.execute(
            "INSERT INTO zhihu_hot (`rank`, title, excerpt, heat, url, answer_count, follower_count) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (item["rank"], item["title"], item["excerpt"], item["heat"],
             item["url"], item["answer_count"], item["follower_count"])
        )
        self.conn.commit()
        return item

    def close_spider(self, spider):
        self.cursor.close()
        self.conn.close()
        spider.logger.info("MySQL: data saved to zhihu_hot table")
