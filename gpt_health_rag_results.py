import json
import os
from http import HTTPStatus
import openai

# 设置OpenAI API密钥
openai.api_key = 'sk-proj-xcOlbWog2lHMNZwFQJ36T3BlbkFJ50RBslgZrB5IpnopZSPo'

# 文件路径
data_file_path = '/root/yao/retrieval/eval_data_use/health_claims_processed.json'
results_file_path = '/root/yao/retrieval/eval_data_predict_gpt_rag/health_claims_processed.json'
index_file_path = '/root/yao/retrieval/index_build_save/arc_challenge_processed_index.json'
train_data_path = '/root/yao/retrieval/contriever_data/train.json'

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

# 检查并读取index文件
index_data = {}
if os.path.exists(index_file_path):
    with open(index_file_path, 'r', encoding='utf-8') as file:
        index_data = json.load(file)

# 读取train数据文件
train_data = {}
if os.path.exists(train_data_path):
    with open(train_data_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)
        
indexxx = 0
for item in data:
    indexxx+=1
    item_id = item['id']
    if item_id in processed_ids:
        continue  # 跳过已处理的项

    question = item['question']
    ctxs_texts = "\n".join([ctx['text'] for ctx in item['ctxs'][:1]])
    prompt = f"Question: {question}\nContext: {ctxs_texts}\nPlease choose directly from true and false, do not return to the thought process, just true or false."

    # 处理index数据
    if item_id in index_data:
        a_id = index_data[item_id]['a_id']
        b_ids = index_data[item_id]['b_id']
        for b_id in b_ids:
            if b_id in train_data:
                instruction = train_data[b_id]['instruction']
                prompt += f" {instruction}"
    
    if indexxx == 1:
        print(prompt)

    # 设置用户消息内容
    messages_template[1]['content'] = prompt
    
    # 调用GPT API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_template
    )
    
    if response['choices']:
        result = response['choices'][0]['message']['content']
        result_data = {'id': item_id, 'result': result}
        
        # 追加保存结果到文件
        with open(results_file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            file.flush()
        
        # 将已处理的ID添加到已处理列表中
        processed_ids.add(item_id)
    else:
        print(f'Error processing item id: {item_id}')
