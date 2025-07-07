from openai import OpenAI
import os
import transformers
import torch

client = OpenAI(api_key="sk-ce297dad0b374efe942c1ff951b758dc",base_url="https://api.deepseek.com")

messages = [
    {"role": "system", "content": "你是一个会深思熟虑的AI助手。"},
    {"role": "user", "content": "你好，我是邱锡鹏老师的学生。"}
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    max_tokens=1024,  # 最大生成token数
    temperature=0.7,  # 控制生成随机性，值越高越随机
    top_p=0.8,  # 核采样参数，保留累积概率前80%的token
    stream=False,  # 是否启用流式输出
)
text = response.choices[0].message.content
print(text)