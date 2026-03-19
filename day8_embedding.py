from dotenv import load_dotenv
import os
import numpy as np
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentences = {
    "base": "如何提高机器学习模型的准确率",
    "similar": "怎样让机器学习效果更好",
    "keyword": "机器学习准确率提高方法",
    "unrelated": "今天晚饭吃什么好"
}

base_emb = get_embedding(sentences["base"])

print("模型：智谱 embedding-3（中文优化）")
for label, text in list(sentences.items())[1:]:
    emb = get_embedding(text)
    sim = cosine_similarity(base_emb, emb)
    print(f"vs「{text}」：{sim:.4f}")

print(f"向量维度：{len(base_emb)}")