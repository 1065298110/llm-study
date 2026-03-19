# day17_memory.py

from langchain_openai import ChatOpenAI
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory
)
from langchain_classic.chains import ConversationChain
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ===== 任务一：BufferMemory，看原始记忆是什么样的 =====
print("=" * 50)
print("任务一：ConversationBufferMemory")
print("=" * 50)

buffer_memory = ConversationBufferMemory()
chain1 = ConversationChain(llm=llm, memory=buffer_memory, verbose=True)

chain1.predict(input="我叫张三，我是一名密码学研究生。")
chain1.predict(input="我正在学习大模型应用开发。")
chain1.predict(input="你还记得我叫什么吗？我在学什么？")

# 查看原始记忆内容
print("\n当前记忆内容：")
print(buffer_memory.load_memory_variables({}))


# ===== 任务二：SummaryBufferMemory，超出限制自动压缩 =====
print("\n" + "=" * 50)
print("任务二：ConversationSummaryBufferMemory")
print("=" * 50)
from langchain_classic.memory import ConversationSummaryMemory

# 直接用 SummaryMemory，不做 token 计数，每轮都压缩
summary_memory = ConversationSummaryMemory(llm=llm)
chain2 = ConversationChain(llm=llm, memory=summary_memory, verbose=False)

chain2.predict(input="我叫张三，我是密码学研究生，目前在学大模型。")
print("第1轮后记忆：", summary_memory.load_memory_variables({})["history"])

chain2.predict(input="我最喜欢的技术方向是RAG系统，特别是混合检索。")
print("第2轮后记忆：", summary_memory.load_memory_variables({})["history"])

chain2.predict(input="我的目标是三个月内找到大模型相关实习，目标公司是华为西研所。")
print("第3轮后记忆：", summary_memory.load_memory_variables({})["history"])

chain2.predict(input="你能帮我总结一下我的情况吗？")
print("第4轮后记忆：", summary_memory.load_memory_variables({})["history"])

# 查看压缩后的记忆
print("\n压缩后的记忆内容：")
print(summary_memory.load_memory_variables({}))