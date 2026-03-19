from dotenv import load_dotenv
import os
import numpy as np
import faiss
import time
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return np.array(response.data[0].embedding)

# ── 第二段：FAISS基础检索 ──
documents = [
    "RAG是检索增强生成技术，先检索再生成",
    "FAISS是Facebook开发的向量检索库",
    "BM25是基于词频的关键词检索算法",
    "Transformer是现代大模型的基础架构",
    "LangChain是LLM应用开发框架",
    "向量数据库用于存储和检索高维向量",
    "Prompt Engineering是设计提示词的技术",
    "Fine-tuning是在预训练模型上继续训练"
]

print("正在向量化文档（需要调用8次API）...")
doc_embeddings = np.array([get_embedding(doc) for doc in documents]).astype('float32')
print(f"文档向量矩阵形状：{doc_embeddings.shape}")

dimension = doc_embeddings.shape[1]  # 2048

# 构建FlatL2索引
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
print(f"索引中向量数量：{index.ntotal}")

def search(query, top_k=3):
    query_emb = get_embedding(query)
    query_emb = np.array([query_emb]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    
    print(f"\n问题：{query}")
    print("检索结果：")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  Top{i+1} [距离:{dist:.4f}] {documents[idx]}")

search("什么是向量检索")
search("LLM开发用什么框架")
search("怎么训练模型")

# ── 第三段：FlatL2 vs IVFFlat速度对比 ──
print("\n\n正在测试索引速度对比...")

# 用随机向量模拟大规模数据（不调API，纯测速）
np.random.seed(42)
n_vectors = 10000
large_corpus = np.random.random((n_vectors, dimension)).astype('float32')
query_vec = np.random.random((1, dimension)).astype('float32')

# FlatL2
flat_index = faiss.IndexFlatL2(dimension)
flat_index.add(large_corpus)

start = time.time()
for _ in range(100):
    flat_index.search(query_vec, 5)
flat_time = (time.time() - start) / 100

# IVFFlat
nlist = 100
ivf_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, nlist)
ivf_index.train(large_corpus)
ivf_index.add(large_corpus)
ivf_index.nprobe = 10

start = time.time()
for _ in range(100):
    ivf_index.search(query_vec, 5)
ivf_time = (time.time() - start) / 100

print(f"FlatL2 平均检索时间：{flat_time*1000:.2f}ms")
print(f"IVFFlat 平均检索时间：{ivf_time*1000:.2f}ms")
print(f"速度提升：{flat_time/ivf_time:.1f}倍")