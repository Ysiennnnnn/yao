# import os
# import openai

#填入你的OPENAI_API_KEY
# openai.api_key = "sk-proj-6ADc5YtNJ2IDIvf68SGDT3BlbkFJVbIl1GSafn6V0JC8MO8h"

# import openai
# from openai import OpenAI

# client = OpenAI(
#     base_url="https://chat1.plus7.plus/v1",
#     api_key = "sk-proj-xcOlbWog2lHMNZwFQJ36T3BlbkFJ50RBslgZrB5IpnopZSPo",
# )


import openai

# 设置你的OpenAI API密钥
openai.api_key = 'sk-proj-xcOlbWog2lHMNZwFQJ36T3BlbkFJ50RBslgZrB5IpnopZSPo'

# 调用GPT-3.5 Turbo模型
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me how to make lasagna."}
    ]
)

# 打印模型的回答
print(response['choices'][0]['message']['content'])
