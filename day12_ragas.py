from ragas import evaluate
from ragas.metrics import (
    faithfulness,         # 忠实度：回答是否基于检索内容
    answer_relevancy,    # 答案相关性：回答是否回答了问题
    context_recall,      # 上下文召回：检索内容是否覆盖了答案
    context_precision,   # 上下文精确度：检索内容是否都有用
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from dotenv import load_dotenv
import os
import numpy as np
import faiss
import pickle
from zhipuai import ZhipuAI

load_dotenv()

zhipu_client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1
)

# 复用之前的文档和索引
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

# ── 加载索引 ──
def get_embedding(text):
    response = zhipu_client.embeddings.create(
        model="embedding-3", input=text
    )
    return np.array(response.data[0].embedding)

INDEX_PATH = "rag_index.faiss"
CHUNKS_PATH = "rag_chunks.pkl"

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    print(f"索引加载完成，共{len(chunks)}块")
else:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = splitter.split_text(document_text)
    embeddings = np.array([get_embedding(c) for c in chunks]).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)

# ── RAG流程 ──
def retrieve(query, top_k=3):
    query_emb = np.array([get_embedding(query)]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0]]

prompt = ChatPromptTemplate.from_messages([
    ("system", """根据以下内容回答问题，只用提供的内容，不要编造。

相关内容：
{context}"""),
    ("user", "{question}")
])
chain = prompt | llm | StrOutputParser()

def rag_answer(question):
    retrieved = retrieve(question)
    context = "\n\n".join(retrieved)
    answer = chain.invoke({"context": context, "question": question})
    return answer, retrieved

# ── 构建测试集 ──
# RAGAS需要四个字段：
# question: 问题
# answer: RAG系统的回答
# contexts: 检索到的文档块列表
# ground_truth: 标准答案（人工写的）
print("\n构建测试集...")

test_cases = [
    {
        "question": "RAG是什么？",
        "ground_truth": "RAG全称Retrieval Augmented Generation，即检索增强生成，是一种将信息检索与文本生成相结合的技术框架。"
    },
    {
        "question": "RAG解决了大模型的哪些问题？",
        "ground_truth": "RAG解决了大语言模型的知识截止日期问题和幻觉问题。"
    },
    {
        "question": "RAG系统由哪些核心组件构成？",
        "ground_truth": "RAG系统主要包含文档处理模块、检索模块和生成模块三个核心组件。"
    },
    {
        "question": "chunk_overlap的作用是什么？",
        "ground_truth": "chunk_overlap是相邻块之间的重叠部分，用于避免重要信息被切断在块的边界处。"
    },
    {
        "question": "RAG有什么局限性？",
        "ground_truth": "RAG的局限性包括：检索质量直接影响生成质量、系统延迟比直接调用模型更高、对于需要跨多个文档进行复杂推理的问题效果不理想。"
    },
]

# 跑RAG系统获取回答和检索结果
dataset_dict = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

for case in test_cases:
    print(f"处理问题：{case['question']}")
    answer, retrieved = rag_answer(case["question"])
    dataset_dict["question"].append(case["question"])
    dataset_dict["answer"].append(answer)
    dataset_dict["contexts"].append(retrieved)
    dataset_dict["ground_truth"].append(case["ground_truth"])

dataset = Dataset.from_dict(dataset_dict)

# ── 配置RAGAS使用智谱 ──
ragas_llm = LangchainLLMWrapper(llm)

embedding_model = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# ── 运行评估 ──
print("\n开始RAGAS评估（需要调用多次API，稍等...）")
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("\n" + "="*50)
print("RAGAS评估结果：")
print("="*50)
df = result.to_pandas()
print(f"Faithfulness（忠实度）：   {df['faithfulness'].mean():.4f}")
print(f"Answer Relevancy（相关性）：{df['answer_relevancy'].mean():.4f}")
print(f"Context Recall（召回率）：  {df['context_recall'].mean():.4f}")
print(f"Context Precision（精确度）：{df['context_precision'].mean():.4f}")
print("\n各问题详细结果：")
print(df.to_string())