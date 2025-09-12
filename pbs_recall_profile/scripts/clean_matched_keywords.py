# scripts/clean_matched_keywords.py
from pymongo import MongoClient

# 连接 MongoDB
client = MongoClient("mongodb://localhost:27017")  
db = client["pbs_news"]  
collection = db["economy_articles"]  

# 查看有多少文档包含 matched_keywords
count_before = collection.count_documents({"matched_keywords": {"$exists": True}})
print(f"清理前，matched_keywords 字段数量: {count_before}")

if count_before > 0:
    # 批量删除 matched_keywords 字段
    result = collection.update_many(
        {"matched_keywords": {"$exists": True}},
        {"$unset": {"matched_keywords": ""}}
    )
    print(f"已清理文档数量: {result.modified_count}")
else:
    print("没有需要清理的 matched_keywords 字段。")

# 再次验证
count_after = collection.count_documents({"matched_keywords": {"$exists": True}})
print(f"清理后，matched_keywords 字段数量: {count_after}")
