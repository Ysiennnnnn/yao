import json
import os
from http import HTTPStatus
import openai

# 设置API密钥
openai.api_key = 'sk-proj-xcOlbWog2lHMNZwFQJ36T3BlbkFJ50RBslgZrB5IpnopZSPo'

# 文件路径
data_file_path = '/root/yao/retrieval/eval_data_use/arc_challenge_processed.json'
results_file_path = '/root/yao/retrieval/eval_data_predict_gpt_rag/arc_challenge_processed.json'
index_file_path = '/root/yao/retrieval/index_build_save/arc_challenge_processed_index.json'
train_file_path = '/root/yao/retrieval/contriever_data/train.json'

# 读取数据集文件
with open(data_file_path, 'r') as file:
    data = json.load(file)

# 获取已处理的ID列表
processed_ids = set()
if os.path.exists(results_file_path):
    with open(results_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                processed_data = json.loads(line)
                processed_ids.add(processed_data['id'])
indexx=0
# 读取index文件
index_data = {}
if os.path.exists(index_file_path):
    with open(index_file_path, 'r', encoding='utf-8') as file:
        index_data = json.load(file)

# 读取train文件
train_data = {}
if os.path.exists(train_file_path):
    with open(train_file_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)

for item in data:
    indexx+=1
    item_id = item['id']
    if item_id in processed_ids:
        continue  # 跳过已处理的项

    question = item['question']
    choices = item['choices']['text']
    labels = item['choices']['label']

    # 构建prompt
    choices_str = "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices))])
    ctxs_texts = "\n".join([ctx['text'] for ctx in item['ctxs'][:1]])

    prompt = f"Question: {question}\nChoices:\n{choices_str}\nContext:\n{ctxs_texts}\nPlease choose directly from A B C D. Do not return to the thought process, just the letter."

    # 追加index和train数据
    if item_id in index_data:
        a_id = index_data[item_id]['a_id']
        b_ids = index_data[item_id]['b_id']
        for b_id in b_ids:
            train_item = next((x for x in train_data if x['id'] == b_id), None)
            if train_item:
                instruction = train_item['instruction']
                output = train_item['output']
                prompt = f"{instruction}\n{output}\n{prompt}"
    if indexx==1:
        print(prompt)

    # 定义生成的消息模板
    messages_template = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    # 调用GPT模型
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
        print('Request failed')
