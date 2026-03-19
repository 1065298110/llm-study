# Week 3 知识总结

---

## 1. Agent 核心概念

### 什么是 Agent
让 LLM 从"回答问题"变成"自主完成多步任务"。
普通 LLM 调用是一次性的：输入 → 输出。
Agent 是循环的：思考 → 行动 → 观察结果 → 再思考 → 直到完成。

### ReAct 框架
业内最主流的 Agent 框架，解决两个本质问题：
- LLM 知识有截止日期，需要工具访问实时/外部信息
- 复杂任务无法一步完成，需要根据上一步结果决定下一步

ReAct 的循环结构：
```
Thought: 我需要做什么
Action: 调用某个工具
Action Input: 工具的输入参数
Observation: 工具返回的结果
（循环，直到）
Final Answer: 输出最终结果
```

### Tool 的核心要点
- `@tool` 装饰器把普通函数变成 Agent 可调用的工具
- **docstring 是给 LLM 看的**，模型靠它判断要不要用这个工具、怎么传参
- docstring 写得越清楚，模型选对工具的概率越高
- 生产环境需要对传入参数做清洗，模型有时会把格式关键词混入参数

### 实验发现
跑 `verbose=True` 时可以在终端看到完整的 Thought/Action/Observation 日志，
这是理解 ReAct 运行机制最直接的方式。

遇到的真实 bug：模型传参时把 `Observation` 关键词拼入了参数字符串
（`"Anthropic\nObservation"`），导致工具匹配失败。
根本原因是 ReAct prompt 格式解析的版本兼容问题。
生产环境解决方案：改用 LangGraph 或 OpenAI Function Calling。

---

## 2. Memory 机制

### 为什么需要 Memory
LLM 每次调用都是无状态的，不记得上一轮说了什么。
Memory 把历史对话以某种形式保存，塞进下一次调用的 Prompt 里。

### 两种 Memory 对比

| 类型 | 原理 | token消耗 | 适用场景 |
|------|------|----------|---------|
| ConversationBufferMemory | 原文全部保存 | 随轮数线性增长 | 短对话（<10轮） |
| ConversationSummaryMemory | 每轮压缩成摘要 | 基本恒定 | 长对话，RAG问答系统 |

### 实验结果
BufferMemory 3轮对话后 history 已经很长，原文完整保留。
SummaryMemory 4轮对话后 history 只剩一段摘要，
关键信息（密码学背景、RAG方向、华为目标）全部保留，token 大幅压缩。

### 代价
SummaryMemory 每轮需要额外一次 LLM 调用来生成摘要，
有额外延迟和 API 费用，但长对话场景下是值得的。

### 注意
`ConversationSummaryBufferMemory`（按 token 数触发压缩）
在智谱 API 下不可用，因为智谱模型没有实现 `get_num_tokens_from_messages()`。
替代方案：直接用 `ConversationSummaryMemory`（每轮都压缩）。

---

## 3. FastAPI 服务化

### 为什么要做服务化
把 Python 脚本包装成 REST API，才能被前端、其他服务调用，
才算真正"部署"了一个系统，而不只是本地跑的脚本。

### 核心概念

**路由装饰器**：`@app.get()` / `@app.post()` 定义接口路径和方法

**请求体**：用 Pydantic `BaseModel` 定义，自动做类型校验
```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
```

**文件上传**：`UploadFile` + `File(...)` 接收二进制文件

**流式输出**：`StreamingResponse` + `async generator` 实现逐 token 返回
```python
async def generate():
    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield chunk.content
return StreamingResponse(generate(), media_type="text/plain")
```

**Swagger UI**：访问 `/docs` 自动生成交互式接口文档，不用写前端就能测试

### 本次实现的三个接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/upload` | POST | 上传 PDF，分块建 FAISS 索引 |
| `/query` | POST | 普通问答，返回完整 JSON |
| `/query/stream` | POST | 流式问答，Transfer-Encoding: chunked |
| `/health` | GET | 健康检查，确认服务和索引状态 |

### 遇到的问题
智谱 embedding API 单次最多接受 64 条输入，
超出时报 `input数组最大不得超过64条`。
解决方案：分批处理，每批 50 条，用 `vectorstore.add_documents()` 追加。

---

## 4. Week 3 跳过的内容

| 内容 | 原因 | 后续补充时机 |
|------|------|------------|
| Docker 化 | 未安装 Docker Desktop | 第二阶段项目完成后补 |
| 异步压测（locust） | 优先级低于项目实战 | 有需要再补 |

---

## 5. 三周基础阶段总结

三周建立的完整技术栈：
```
Week 1：LLM基础
  Transformer架构 → API调用 → Prompt Engineering → HuggingFace → LangChain LCEL

Week 2：RAG全链路
  Embedding原理 → FAISS向量库 → 文档分块 → 完整RAG Pipeline
  → 混合检索(BM25+向量+RRF) → RAGAS评估框架

Week 3：工程能力
  Agent(ReAct框架) → Tool调用 → Memory机制 → FastAPI服务化
```

RAGAS 基线数据（来自 Week 2 实验）：
- Faithfulness：1.0
- Answer Relevancy：0.60（待优化）
- Context Recall：1.0
- Context Precision：1.0

---

## 6. 下阶段目标（Week 4 起）

进入第二阶段项目攻坚：
**面向长文档的多策略RAG检索增强问答系统**

核心优化方向：
- BGE-Reranker 二阶段精排
- 层级索引（摘要层+细粒度层）
- HyDE 查询改写
- Answer Relevancy 从 0.60 提升到 0.75+
- 完整的 5组配置 × 4指标 RAGAS 对比矩阵