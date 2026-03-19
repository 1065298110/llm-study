from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import numpy as np
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

# ── 第一部分：加载PDF ──
# 先用一个简单的文本模拟，不需要真实PDF
# 模拟一篇技术文章
fake_document = """
第一章 引言
人工智能技术近年来取得了突破性进展。大语言模型的出现改变了人机交互的方式。
本文将介绍RAG技术的核心原理和实现方法。RAG全称检索增强生成，是一种结合检索
和生成的技术框架。它能够有效解决大模型知识过时和幻觉问题。

第二章 RAG原理
RAG系统主要由三个部分组成：文档处理模块、检索模块和生成模块。
文档处理模块负责将原始文档切分成合适大小的块，并转换成向量表示。
检索模块负责根据用户问题找到最相关的文档块。
生成模块负责将检索到的内容和用户问题一起喂给大语言模型生成最终回答。

第三章 向量检索
向量检索是RAG系统的核心技术之一。它将文本转换成高维向量，通过计算向量之间
的距离来衡量语义相似度。常用的向量检索库包括FAISS、Chroma和Milvus。
FAISS由Facebook开发，支持多种索引类型，适合大规模向量检索任务。
Chroma是一个轻量级向量数据库，适合开发和小规模应用。

第四章 分块策略
文档分块是RAG系统中容易被忽视但非常重要的环节。
chunk_size决定每个块的大小，chunk_overlap决定相邻块之间的重叠部分。
重叠的目的是避免重要信息被切断在两个块的边界处。
合适的分块策略能显著提升检索质量。
"""

# ── 第二部分：对比不同分块参数 ──
def chunk_and_analyze(text, chunk_size, chunk_overlap):
    """
    text: 要分块的文本
    chunk_size: 每块的字符数
    chunk_overlap: 相邻块重叠的字符数
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 分割优先级：先按段落，再按句子，最后按字符
        # 这就是Recursive（递归）的含义
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    print(f"\nchunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"切成了 {len(chunks)} 块")
    print(f"平均每块字数：{sum(len(c) for c in chunks) / len(chunks):.0f}字")
    print(f"最短块：{min(len(c) for c in chunks)}字")
    print(f"最长块：{max(len(c) for c in chunks)}字")
    print(f"\n第一块内容预览：\n{chunks[0][:100]}...")
    
    return chunks

# 跑三组对比实验
configs = [
    (100, 0),    # 小块，无重叠
    (200, 50),   # 中块，有重叠
    (500, 100),  # 大块，有重叠
]

all_results = {}
for chunk_size, chunk_overlap in configs:
    chunks = chunk_and_analyze(fake_document, chunk_size, chunk_overlap)
    all_results[f"size{chunk_size}_overlap{chunk_overlap}"] = chunks

# ── 第三部分：重叠的作用演示 ──
print("\n\n" + "="*50)
print("演示chunk_overlap的作用")
print("="*50)

test_text = "张三是一位优秀的工程师。他擅长Python编程。他在北京工作。他喜欢打篮球。"

# 无重叠
no_overlap = RecursiveCharacterTextSplitter(
    chunk_size=20, chunk_overlap=0
).split_text(test_text)

# 有重叠
with_overlap = RecursiveCharacterTextSplitter(
    chunk_size=20, chunk_overlap=10
).split_text(test_text)

print("\n无重叠分块：")
for i, chunk in enumerate(no_overlap):
    print(f"  块{i}: {chunk}")

print("\n有重叠分块：")
for i, chunk in enumerate(with_overlap):
    print(f"  块{i}: {chunk}")

print("\n观察：有重叠时，「张三」这个主语会出现在更多块里，")
print("避免了「他擅长Python」这种句子丢失主语的问题")