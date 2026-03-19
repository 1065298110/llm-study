from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import numpy as np
import faiss
from zhipuai import ZhipuAI

load_dotenv()

zhipu_client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1
)

# ── 第一步：准备文档 ──
# 用一段较长的文本模拟真实文档
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

# ── 第二步：分块 ──
print("Step 1: 文档分块...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = splitter.split_text(document_text)
print(f"共切成 {len(chunks)} 块")

# ── 第三步：向量化所有块 ──
print("\nStep 2: 向量化文档块（需要调用API）...")
def get_embedding(text):
    response = zhipu_client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return np.array(response.data[0].embedding)

chunk_embeddings = np.array([get_embedding(chunk) for chunk in chunks]).astype('float32')
print(f"向量矩阵形状：{chunk_embeddings.shape}")

# ── 第四步：建FAISS索引 ──
print("\nStep 3: 建立FAISS索引...")
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print(f"索引中向量数量：{index.ntotal}")

# ── 第五步：检索函数 ──
def retrieve(query, top_k=3):
    """
    输入问题，返回最相关的top_k个文档块
    """
    query_emb = get_embedding(query)
    query_emb = np.array([query_emb]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks

# ── 第六步：生成函数 ──
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识库问答助手。请根据以下检索到的相关内容回答用户问题。
    
相关内容：
{context}

要求：
1. 只根据提供的相关内容回答，不要编造信息
2. 如果相关内容不足以回答问题，请说明
3. 回答要简洁准确"""),
    ("user", "{question}")
])

parser = StrOutputParser()
chain = prompt | llm | parser

def rag_answer(question):
    """
    完整的RAG流程：检索 + 生成
    """
    print(f"\n{'='*50}")
    print(f"问题：{question}")
    
    # 检索
    retrieved = retrieve(question, top_k=3)
    context = "\n\n".join(retrieved)
    
    print(f"\n检索到的相关内容（共{len(retrieved)}块）：")
    for i, chunk in enumerate(retrieved):
        print(f"  [{i+1}] {chunk[:60]}...")
    
    # 生成
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    print(f"\n最终回答：{answer}")
    return answer

# ── 第七步：测试 ──
rag_answer("RAG是什么？")
rag_answer("为什么需要RAG技术？")
rag_answer("分块策略怎么选择？")
rag_answer("RAG有什么缺点？")