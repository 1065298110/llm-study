from dotenv import load_dotenv
import os
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

question = "给我推荐一个周末活动"

for temp in [0.0, 0.7, 1.5]:
    print(f"\n{'='*50}")
    print(f"temperature = {temp}")
    print('='*50)
    
    for i in range(3):
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": question}],
            temperature=temp
        )
        result = response.choices[0].message.content.strip()
        print(f"\n第{i+1}次: {result[:80]}...")  # 只打印前80字