import json
import os
from http import HTTPStatus
import openai

# 设置API密钥
openai.api_key = 'sk-proj-xcOlbWog2lHMNZwFQJ36T3BlbkFJ50RBslgZrB5IpnopZSPo'

# 数据集文件路径
data_file_path = '/root/yao/retrieval/eval_data_use/popqa_longtail.json'
# 结果保存文件路径
results_file_path = '/root/yao/retrieval/eval_data_predict_gpt_base/popqa_longtail.json'

# 读取数据集文件
with open(data_file_path, 'r') as file:
    data = json.load(file)

# 定义生成的消息模板
messages_template = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': ''}
]

# 获取已处理的ID列表
processed_ids = set()
if os.path.exists(results_file_path):
    with open(results_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 确保行不为空
                processed_data = json.loads(line)
                processed_ids.add(processed_data['id'])

for item in data:
    item_id = item['id']
    if item_id in processed_ids:
        continue  # 跳过已处理的项

    question = item['question']
    s_wiki_title = item.get('s_wiki_title', '')
    ctxs = item.get('ctxs', [])
    
    # 构建上下文内容
    context_texts = "\n".join([f"Context {i+1}: {ctx['text']}" for i, ctx in enumerate(ctxs[:2])])
    
    # 构建prompt
    prompt = f"Question: {question}\nTitle: {s_wiki_title}\n{context_texts}\nPlease list all possible answers, The result may be a word or phrase, do not return to the thinking process."

    # 设置用户消息内容
    messages_template[1]['content'] = prompt
    
    # 调用GPT API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_template
    )
    
    if response:
        result = response['choices'][0]['message']['content']
        result_data = {'id': item_id, 'result': result}
        
        # 追加保存结果到文件
        with open(results_file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            file.flush()
        
        # 将已处理的ID添加到已处理列表中
        processed_ids.add(item_id)
    else:
        print('Failed to get a response for item id:', item_id)
