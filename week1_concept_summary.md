# Week 1 知识总结

---

## 1. Transformer 架构

Transformer 是现代所有大模型的基础架构，由 Encoder 和 Decoder 两部分组成，但实际使用中分成三类模型：

| 类型 | 代表模型 | 工作方式 | 适合任务 |
|------|---------|---------|---------|
| Encoder-only | BERT、BGE-M3 | 双向阅读整句话，输出每个词的理解向量 | Embedding、分类、信息抽取 |
| Decoder-only | GPT2、GLM、Qwen | 从左到右逐词生成，只看左边上下文 | 对话、文本生成、代码生成 |
| Encoder-Decoder | T5、BART | Encoder理解输入，Decoder生成输出 | 翻译、摘要、改写 |

**你调用的 glm-4-flash 是 Decoder-only。你做Embedding实验用的 all-MiniLM-L6-v2 是 Encoder-only。**

Decoder 的核心机制（3B1B视频的内容）：
- 把文字切成 Token
- 每个 Token 转成向量（Embedding）
- 经过多层 Attention，每个词向量融合上下文信息
- 最后预测下一个词的概率分布
- 按概率采样，输出下一个词

---

## 2. 核心概念速查

### Token
文字的最小处理单位，不等于一个字。中文大约每个字是1个token，英文大约每3-4个字母是1个token。`"今天天气很好"` 大概是6个token。模型处理的是token序列，不是原始文字。

### Embedding（嵌入）
把文字转成高维数字向量的过程。语义相近的词/句子，在向量空间里距离也相近。

你的实验结果：
```
「今天天气很好」vs「今天阳光明媚」→ 相似度 0.7629  （语义相近）
「今天天气很好」vs「我喜欢吃苹果」→ 相似度 0.4961  （语义无关）
「今天天气很好」vs「机器学习是AI的子领域」→ 相似度 0.3749  （语义无关）
```
这是RAG检索的核心原理——用向量相似度找语义相关的文档，而不是关键词匹配。

### Context Window（上下文窗口）
模型每次能「看到」的最大token数量。比如glm-4-flash支持128k token，超出这个长度的内容模型就看不到了。这是RAG要解决的核心问题之一——文档太长放不进去，所以要先检索出相关片段再喂给模型。

### Temperature
控制模型输出的随机程度。

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0 | 每次输出几乎完全一样 | RAG问答、代码生成 |
| 0.7 | 有变化但合理 | 一般对话 |
| 1.5+ | 很随机，可能出现奇怪表达 | 创意写作（慎用） |

你的实验验证了这一点：temperature=0时三次回答几乎一模一样，temperature=1.5时开始出现随机变化。

### Top-p（核采样）
每次选下一个词时，只从「概率累加到p」的候选词里随机选。比如Top-p=0.9，就是把所有候选词按概率从高到低排列，累加到90%就停，只在这个范围内抽取。p越小候选词越少越保守，p越大越随机。和temperature作用类似，实际使用一般只调其中一个。

### System Prompt
对话开始前给模型的「角色设定和行为约束」，相当于给模型划定一个大范围。比如：`"你是一个专业的Python工程师，只回答编程相关问题"`。之后所有user的问题都在这个约束下回答。

### User Prompt
用户实际输入的问题，在System Prompt划定的范围内有效。

---

## 3. Prompt Engineering 三种方式

### Zero-shot
直接问，不给任何示例。适合没有固定格式要求的场景。

```
输入：判断这句话的情感是正面还是负面：'这家餐厅服务太差了'
输出：负面（但格式不一定可控）
```

### Few-shot
先给几个「输入→输出」的例子，再问正式问题。适合需要控制输出格式的场景。

```
例子：'这部电影太精彩了' → 正面
例子：'快递速度慢得离谱' → 负面
现在判断：'这家餐厅服务太差了' → （模型严格按格式输出）负面
```

**和Zero-shot的核心区别：** Few-shot的输出格式更可控，因为模型从例子里学到了「只输出正面或负面」。

### Chain-of-Thought（CoT，思维链）
让模型先推理再回答。适合有逻辑步骤、有歧义、有转折的复杂问题。

```
问：判断「虽然等了很久，但菜的味道确实不错」的情感，请先分析再给结论
模型：这句话有转折，前半句「等了很久」是负面，但「但」字后面「味道不错」是正面，
     整体来看正面评价占主导 → 正面
```

不加CoT时模型可能只抓到「等了很久」判断为负面，加了CoT之后推理过程可见，结论更准确。

---

## 4. HuggingFace 生态

### 是什么
全球最大的开源模型社区，提供：
- **Model Hub**：数十万个开源模型，可以直接下载使用
- **Transformers库**：统一的模型加载和推理接口
- **Datasets库**：开源数据集
- **Spaces**：免费部署AI应用（你的项目最终会部署在这里）

### Pipeline 接口
把「加载模型→tokenize→推理→decode」全部封装好，一行代码跑通一个任务：

```python
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("Artificial intelligence is", max_new_tokens=50)
```

### Embedding 模型
`all-MiniLM-L6-v2` 是 Encoder-only 模型，专门把句子转成384维向量。后面项目里会换成中文效果更好的 `BGE-M3`（1024维）。

---

## 5. LangChain 基础

### 是什么
专门为LLM应用开发设计的框架，把常用操作封装好，让你不用每次都手写API调用。

### 三个核心组件

**PromptTemplate：** 把提示词模板化，用占位符复用
```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("user", "{question}")
])
# 调用时传入变量值，模板自动填充
```

**LLM：** 模型调用的封装，智谱/阿里/OpenAI都用同一套接口
```python
llm = ChatOpenAI(model="glm-4-flash", api_key=..., base_url=...)
response = llm.invoke("问题")
# 返回 AIMessage 对象，.content 取文字内容
```

**OutputParser：** 把LLM返回的 AIMessage 对象转成纯字符串
```python
parser = StrOutputParser()
result = parser.invoke(response)  # 现在是字符串了
```

### LCEL 管道语法（重点）
用 `|` 把组件串联成流水线，数据从左到右依次流过：

```python
chain = prompt | llm | parser
#        ↑        ↑      ↑
#     填充变量  调用模型  提取文字

result = chain.invoke({"role": "专家", "question": "解释RAG"})
```

**为什么重要：** 后面RAG的完整链就是在这个基础上加一个检索器：
```python
chain = retriever | prompt | llm | parser
```
结构不变，只是在前面插了一个「先检索相关文档」的步骤。

---

## 6. 开发环境

```
Anaconda 虚拟环境：llm-study（Python 3.10）
编辑器：VSCode，解释器选 llm-study
API：智谱 glm-4-flash（免费）+ 阿里百炼 qwen-plus（免费额度）
.env 文件：存放 API Key，不传 GitHub
```

---

## 7. 下周预告（Week 2）

下周进入RAG全链路，会用到这周所有的概念：
- Embedding → 把文档转成向量存起来
- 向量相似度 → 检索最相关的文档片段
- LangChain → 把检索+生成串成完整流水线
- Prompt Engineering → 设计好的System Prompt让回答更准确
