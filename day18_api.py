# day18_api.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import os
import tempfile
import asyncio

load_dotenv()

app = FastAPI(title="RAG API", description="基于混合检索的RAG问答系统")

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 全局存储索引（生产环境应该用数据库，这里简化）
vectorstore = None


# ===== 接口一：上传文档，建索引 =====
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    
    # 把上传的文件保存到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # 加载和分块
    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    # 建向量索引
    # 智谱embedding限制每批最多64条，分批处理
    batch_size = 50
    if len(chunks) <= batch_size:
       vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore = FAISS.from_documents(chunks[:batch_size], embeddings)
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            vectorstore.add_documents(batch)
    
    # 清理临时文件
    os.unlink(tmp_path)
    
    return {
        "status": "success",
        "filename": file.filename,
        "chunks": len(chunks)
    }


# ===== 接口二：普通问答 =====
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/query")
async def query(request: QueryRequest):
    if vectorstore is None:
        return {"error": "请先上传文档"}
    
    # 检索
    docs = vectorstore.similarity_search(request.question, k=request.top_k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 生成
    prompt = f"""基于以下内容回答问题，回答要简洁准确。
如果内容中没有相关信息，请直接说"文档中未找到相关信息"。

内容：
{context}

问题：{request.question}"""
    
    response = llm.invoke(prompt)
    
    return {
        "answer": response.content,
        "source_chunks": len(docs)
    }


# ===== 接口三：流式输出问答 =====
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    if vectorstore is None:
        async def error_gen():
            yield "请先上传文档"
        return StreamingResponse(error_gen(), media_type="text/plain")
    
    docs = vectorstore.similarity_search(request.question, k=request.top_k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""基于以下内容回答问题，回答要简洁准确。
如果内容中没有相关信息，请直接说"文档中未找到相关信息"。

内容：
{context}

问题：{request.question}"""

    async def generate():
        async for chunk in llm.astream(prompt):
            if chunk.content:
                yield chunk.content
    
    return StreamingResponse(generate(), media_type="text/plain")


# ===== 健康检查 =====
@app.get("/health")
async def health():
    return {"status": "ok", "index_loaded": vectorstore is not None}