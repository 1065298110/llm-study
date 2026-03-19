from transformers import pipeline

# # pipeline是HuggingFace封装好的高级接口
# # 第一次运行会自动从网上下载模型，需要等一会儿
# generator = pipeline("text-generation", model="gpt2")

# result = generator(
#     "Artificial intelligence is",
#     max_new_tokens=50,   # 最多生成50个新token
#     num_return_sequences=2   # 生成2个不同的结果
# )

# for i, r in enumerate(result):
#     print(f"\n结果{i+1}: {r['generated_text']}")
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载一个专门做embedding的模型
# 第一次会下载，比gpt2稍大
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "今天天气很好",
    "今天阳光明媚",       # 语义相近
    "我喜欢吃苹果",       # 语义无关
    "机器学习是AI的子领域"  # 语义无关
]

# 把所有句子转成向量
embeddings = model.encode(sentences)

print(f"\n向量维度：{embeddings.shape}")
# 输出类似：(4, 384) 表示4个句子，每个384维

# 计算余弦相似度
def cosine_similarity(a, b):
    """
    余弦相似度：两个向量的夹角余弦值
    值域[-1,1]，越接近1说明语义越相似
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

base = embeddings[0]  # "今天天气很好"作为基准

print(f"\n以「今天天气很好」为基准：")
for i in range(1, len(sentences)):
    sim = cosine_similarity(base, embeddings[i])
    print(f"vs「{sentences[i]}」：相似度 = {sim:.4f}")