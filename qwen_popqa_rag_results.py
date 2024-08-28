import json
import os
from http import HTTPStatus
import dashscope

# 设置API密钥
dashscope.api_key = 'sk-64de2eee2c81419587d35b5fb6b5394f'

# 文件路径
data_file_path = '/root/yao/retrieval/eval_data_use/popqa_longtail.json'
results_file_path = '/root/yao/retrieval/eval_data_predict_qwen_rag/popqa_longtail.json'
arc_index_path = '/root/yao/retrieval/index_build_save/arc_challenge_processed_index.json'
train_data_path = '/root/yao/retrieval/contriever_data/train.json'

# 读取数据集文件
with open(data_file_path, 'r') as file:
    data = json.load(file)

# 读取已处理的结果
processed_ids = set()
if os.path.exists(results_file_path):
    with open(results_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                processed_data = json.loads(line)
                processed_ids.add(processed_data['id'])

# 检查arc_challenge_processed_index.json文件是否存在
if os.path.exists(arc_index_path):
    with open(arc_index_path, 'r') as file:
        arc_index_data = json.load(file)
else:
    arc_index_data = {}

# 读取train.json文件
if os.path.exists(train_data_path):
    with open(train_data_path, 'r') as file:
        train_data = json.load(file)
else:
    train_data = []

# 定义生成的消息模板
messages_template = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': ''}
]

# 构建ID到train数据的映射
train_data_dict = {item['id']: item for item in train_data}

for item in data:
    item_id = item['id']
    if item_id in processed_ids:
        continue

    question = item['question']
    s_wiki_title = item.get('s_wiki_title', '')
    ctxs = item.get('ctxs', [])
    
    # 构建上下文内容
    context_texts = "\n".join([f"Context {i+1}: {ctx['text']}" for i, ctx in enumerate(ctxs[:2])])
    
    # 添加arc_challenge_processed_index.json中的内容
    if item_id in arc_index_data:
        a_id = arc_index_data[item_id]['a_id']
        b_ids = arc_index_data[item_id]['b_id']
        additional_contexts = []
        for b_id in b_ids:
            if b_id in train_data_dict:
                additional_contexts.append(train_data_dict[b_id]['instruction'])
        
        additional_context_texts = "\n".join(additional_contexts)
        context_texts = f"{context_texts}\n{additional_context_texts}"

    # 构建prompt
    prompt = f"Question: {question}\nTitle: {s_wiki_title}\nContext: {context_texts}\nPlease list all possible answers. The result may be a word or phrase, do not return to the thinking process."
    # print(prompt)
    # 设置用户消息内容
    messages_template[1]['content'] = prompt
    
    # 调用qwen API
    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages_template,
        result_format='message'
    )
    
    if response.status_code == HTTPStatus.OK:
        result = response.output.choices[0]['message']['content']
        result_data = {'id': item_id, 'result': result}
        
        # 追加保存结果到文件
        with open(results_file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            file.flush()
        
        # 将已处理的ID添加到已处理列表中
        processed_ids.add(item_id)
    else:
        print(f'Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}')
