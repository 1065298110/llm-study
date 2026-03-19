from dotenv import load_dotenv
import os
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

def ask(messages, label):
    """
    封装一个通用的提问函数，避免重复写client.chat.completions.create
    label是标签，用来标记这次实验是哪种方式
    """
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        temperature=0.1   # 今天做对比实验，temperature固定低值，排除随机干扰
    )
    result = response.choices[0].message.content.strip()
    print(f"\n【{label}】\n{result}\n")
    return result

# ── Zero-shot：直接问，不给任何示例 ──
ask([
    {"role": "user", "content": "判断这句话的情感是正面还是负面：'这家餐厅的服务态度太差了'"}
], label="Zero-shot")
# ── Few-shot：给几个例子，让模型学习格式和规律 ──
ask([
    {"role": "user", "content": """判断情感是正面还是负面，只回答"正面"或"负面"。

例子：
输入：'这部电影太精彩了' → 正面
输入：'快递速度慢得离谱' → 负面
输入：'今天天气不错' → 正面

现在判断：
输入：'这家餐厅的服务态度太差了'"""}
], label="Few-shot")
# ── Chain-of-Thought：让模型先推理再回答 ──
ask([
    {"role": "user", "content": """判断这句话的情感，请先分析原因，再给出结论。

句子：'虽然等了很久，但菜的味道确实不错'"""}
], label="Chain-of-Thought")

# 对比：同一句话用Zero-shot
ask([
    {"role": "user", "content": "判断这句话的情感是正面还是负面：'虽然等了很久，但菜的味道确实不错'"}
], label="Zero-shot对比（复杂句子）")