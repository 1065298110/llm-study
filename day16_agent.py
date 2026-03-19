#有bug
# day16_agent.py

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 工具列表
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# ReAct 标准 Prompt（这个格式是固定的，不能随便改）
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,  # 加这行：解析失败时让模型重试而不是崩溃
    handle_tool_errors=True,     # 加这行：工具报错时也能优雅处理
)

# 跑起来
result = agent_executor.invoke({"input": "2024年巴黎奥运会中国获得了多少枚金牌？"})
print("\n最终答案：", result["output"])


# ===== 任务二：自定义 Tool =====
from langchain.tools import tool

@tool
def get_ai_company_info(company_name: str) -> str:
    """查询AI公司的基本信息。输入公司名称，返回该公司的主要产品和成立时间。只需传入公司名称本身，例如：Anthropic"""
    company_db = {
        "anthropic": "成立于2021年，主要产品是Claude系列大模型，专注于AI安全研究。",
        "openai": "成立于2015年，主要产品是GPT系列和ChatGPT，目前估值超过1000亿美元。",
        "智谱ai": "成立于2019年，源自清华大学，主要产品是GLM系列模型和ChatGLM。",
        "月之暗面": "成立于2023年，主要产品是Kimi，以超长上下文窗口著称。",
    }
    key = company_name.lower().strip()
    print(f"DEBUG 查询的key: [{repr(key)}]")  # 加这行

    return company_db.get(key, f"未找到关于 {company_name} 的信息。")

# 重建 agent，加入自定义工具
tools2 = [search_tool, get_ai_company_info]
agent2 = create_react_agent(llm, tools2, prompt)
agent_executor2 = AgentExecutor(
    agent=agent2,
    tools=tools2,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    handle_tool_errors=True,
)

result2 = agent_executor2.invoke({"input": "帮我查一下Anthropic这个公司的信息"})
print("\n最终答案：", result2["output"])