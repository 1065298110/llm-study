from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import os
import numpy as np
import faiss
import pickle
import jieba
from zhipuai import ZhipuAI

load_dotenv()
zhipu_client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1
)

document_text = """
RAG技术详解

第一节 什么是RAG
RAG全称Retrieval Augmented Generation，即检索增强生成。它是一种将信息检索
与文本生成相结合的技术框架。RAG的核心思想是：在生成回答之前，先从知识库中
检索出与问题相关的内容，再将检索结果作为上下文提供给语言模型生成最终答案。

第二节 为什么需要RAG
大语言模型存在两个主要问题：知识截止日期和幻觉问题。知识截止日期意味着模型
不了解训练数据之后发生的事情。幻觉问题是指模型可能生成看似合理但实际错误的
内容。RAG通过引入外部知识库来解决这两个问题，让模型能够基于真实、最新的信
息生成回答，并且可以溯源引用来源。

第三节 RAG的核心组件
RAG系统主要包含三个核心组件。第一是文档处理模块，负责将原始文档进行清洗、
分块和向量化。第二是检索模块，负责根据用户问题找到最相关的文档块，通常使用
向量相似度搜索实现。第三是生成模块，将检索到的相关内容和用户问题一起输入给
大语言模型，生成最终的回答。

第四节 向量化与检索
文档向量化是RAG系统的关键步骤。通过Embedding模型将文本转换成高维向量，使
得语义相似的文本在向量空间中距离更近。检索时，将用户问题同样转换成向量，然
后通过计算余弦相似度或L2距离找到最相关的文档块。常用的向量检索工具包括
FAISS、Chroma和Milvus等。

第五节 分块策略的重要性
文档分块策略对RAG系统的效果有重要影响。chunk_size决定每个块的大小，太小
会导致上下文不完整，太大会引入噪音。chunk_overlap是相邻块之间的重叠部分，
用于避免重要信息被切断在块的边界处。对于技术文档，通常推荐chunk_size在
300到500字之间，overlap在50到100字之间。

第六节 RAG的局限性
RAG系统也存在一些局限性。检索质量直接影响生成质量，如果检索到的内容不相关，
模型可能生成错误答案。此外，RAG系统的延迟比直接调用模型更高，因为需要额外
的检索步骤。对于需要跨多个文档进行复杂推理的问题，RAG的效果也可能不理想。
"""

# ── 第一部分：建库或加载（持久化）──
INDEX_PATH = "rag_index.faiss"
CHUNKS_PATH = "rag_chunks.pkl"

def build_index(chunks):
    """向量化所有块，建立FAISS索引，保存到本地"""
    print("正在向量化文档块...")
    embeddings = np.array([get_embedding(c) for c in chunks]).astype('float32')
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # 保存索引和chunks到本地
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"索引已保存到本地：{INDEX_PATH}")
    return index, chunks

def load_or_build(document_text):
    """
    如果本地有索引就直接加载，没有就重新建
    这样重启程序就不用重新调用API了
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = splitter.split_text(document_text)

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("发现本地索引，直接加载...")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)
        print(f"加载完成，共{index.ntotal}个向量")
    else:
        print("未发现本地索引，重新建立...")
        index, chunks = build_index(chunks)

    return index, chunks

def get_embedding(text):
    response = zhipu_client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return np.array(response.data[0].embedding)

# 加载或建立索引
index, chunks = load_or_build(document_text)

# ── 第二部分：BM25检索 ──
# BM25是关键词检索，需要先对文本分词
print("\n建立BM25索引...")

# jieba分词：把中文句子切成词语列表
tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
# 例如："RAG是检索增强生成" → ["RAG", "是", "检索", "增强", "生成"]

bm25 = BM25Okapi(tokenized_chunks)
print("BM25索引建立完成")

def bm25_search(query, top_k=3):
    """BM25关键词检索"""
    # 问题也要分词
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    # 取分数最高的top_k个
    top_indices = np.argsort(scores)[::-1][:top_k]
    return list(top_indices), scores[top_indices]

def vector_search(query, top_k=3):
    """向量检索"""
    query_emb = get_embedding(query)
    query_emb = np.array([query_emb]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    return list(indices[0]), distances[0]

# ── 第三部分：RRF融合 ──
def rrf_fusion(vector_indices, bm25_indices, k=60):
    """
    RRF（Reciprocal Rank Fusion）：融合两路检索结果
    
    原理：每个文档的得分 = 1/(k + 向量排名) + 1/(k + BM25排名)
    k=60是经验值，防止排名第1的文档得分过高
    
    优势：不需要对两路分数归一化，直接用排名融合
    """
    scores = {}
    
    # 向量检索结果打分
    for rank, idx in enumerate(vector_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    
    # BM25检索结果打分
    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    
    # 按总分排序，取top3
    sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in sorted_indices[:3]]

# ── 第四部分：对比实验 ──
def compare_search(query):
    print(f"\n{'='*50}")
    print(f"问题：{query}")
    
    # 单独向量检索
    v_indices, v_distances = vector_search(query, top_k=3)
    print(f"\n向量检索结果：")
    for i, idx in enumerate(v_indices):
        print(f"  [{i+1}] {chunks[idx][:50]}...")
    
    # 单独BM25检索
    b_indices, b_scores = bm25_search(query, top_k=3)
    print(f"\nBM25检索结果：")
    for i, idx in enumerate(b_indices):
        print(f"  [{i+1}] {chunks[idx][:50]}...")
    
    # RRF融合
    fused_indices = rrf_fusion(v_indices, b_indices)
    print(f"\n混合检索（RRF融合）结果：")
    for i, idx in enumerate(fused_indices):
        print(f"  [{i+1}] {chunks[idx][:50]}...")

# 测试两个问题
compare_search("RAG的核心组件有哪些")
compare_search("FAISS检索工具")  # 这个问题BM25会更有优势（精确关键词）