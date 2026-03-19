# from langchain_core.prompts import ChatPromptTemplate

# # PromptTemplate的作用：把提示词模板化，用变量占位
# # 好处是同一个模板可以复用，只改变量值
# template = ChatPromptTemplate.from_messages([
#     ("system", "你是一个{role}，请用专业的语气回答问题。"),
#     ("user", "{question}")
# ])

# # 把模板渲染成具体的消息列表
# messages = template.invoke({
#     "role": "Python工程师",
#     "question": "什么是装饰器？"
# })

# print("渲染后的消息：")
# for m in messages.messages:
#     print(f"  [{m.type}]: {m.content}")

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # 用智谱的OpenAI兼容接口接入LangChain
# llm = ChatOpenAI(
#     model="glm-4-flash",
#     api_key=os.getenv("ZHIPUAI_API_KEY"),
#     base_url="https://open.bigmodel.cn/api/paas/v4/",
#     temperature=0.1
# )

# # OutputParser的作用：把LLM返回的复杂对象转成纯字符串
# parser = StrOutputParser()

# # 单独测试LLM
# response = llm.invoke("用一句话解释什么是RAG")
# print(f"LLM原始返回类型：{type(response)}")
# print(f"LLM原始返回内容：{response.content}")

# # 用parser处理
# result = parser.invoke(response)
# print(f"\nParser处理后类型：{type(result)}")
# print(f"Parser处理后内容：{result}")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}。"),
    ("user", "{question}")
])

parser = StrOutputParser()

# ── LCEL管道：用 | 把三个组件串联起来 ──
chain = prompt | llm | parser
#        ↑        ↑      ↑
#     模板渲染   LLM推理  提取文字

# 调用整个链，只需要传变量
result = chain.invoke({
    "role": "数据结构专家",
    "question": "用一句话解释二叉树"
})
print(f"链式调用结果：{result}")

# 换个角色，复用同一条链
result2 = chain.invoke({
    "role": "厨师",
    "question": "用一句话解释二叉树"
})
print(f"\n换角色后结果：{result2}")