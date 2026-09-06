import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载 .env 文件中的环境变量（API Key 不再硬编码，防止泄露）
load_dotenv()

# 2. 创建客户端，连接硅基流动
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.siliconflow.cn/v1"
)

# 2. 调用大模型
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",  # 用哪个模型
    messages=[
        {"role": "system", "content": "你是一个友好的AI助手，回答简洁明了。"},
        {"role": "user", "content": "你好，请用一句话介绍一下你自己。"}
    ],
    temperature=0.7,  # 创造性，0=很稳定，1=很随机
    max_tokens=500    # 回答最多多少字
)

# 3. 打印回答
print("AI 的回答：")
print(response.choices[0].message.content)
