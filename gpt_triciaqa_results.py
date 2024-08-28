import json
import os
from http import HTTPStatus
import openai

# 设置API密钥
openai.api_key = 'sk-proj-xcOlbWog2lHMNZwFQJ36T3BlbkFJ50RBslgZrB5IpnopZSPo'

# 数据集文件路径
data_file_path = '/root/yao/retrieval/eval_data_use/triviaqa_test.json'
# 结果保存文件路径
results_file_path = '/root/yao/retrieval/eval_data_predict_gpt_rag/triviaqa_test.json'
# 处理后的索引文件路径
processed_index_path = '/root/yao/retrieval/index_build_save/arc_challenge_processed_index.json'
# 训练数据文件路径
train_data_path = '/root/yao/retrieval/contriever_data/train.json'

# 读取数据集文件
with open(data_file_path, 'r') as file:
    data = json.load(file)

# 获取已处理的ID列表
processed_ids = set()
if os.path.exists(results_file_path):
    with open(results_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 确保行不为空
                processed_data = json.loads(line)
                processed_ids.add(processed_data['id'])

# 读取处理后的索引文件
if os.path.exists(processed_index_path):
    with open(processed_index_path, 'r', encoding='utf-8') as file:
        processed_index = json.load(file)
else:
    processed_index = {}

# 读取训练数据文件
with open(train_data_path, 'r', encoding='utf-8') as file:
    train_data = {item['id']: item for item in json.load(file)}

for item in data:
    item_id = item['id']
    if item_id in processed_ids:
        continue  # 跳过已处理的项

    question = item['question']
    ctxs = item.get('ctxs', [])
    
    # 构建上下文内容
    context_texts = "\n".join([f"Context {i+1}: {ctx['text']}" for i, ctx in enumerate(ctxs[:1])])

    # 添加默认的prompt
    prompt = f"Question: {question}\nContext: {context_texts}"

    # 查找arc_challenge_processed_index.json中的a_id和b_id
    if item_id in processed_index:
        a_id = processed_index[item_id]['a_id']
        b_ids = processed_index[item_id].get('b_id', [])
        
        # 根据b_ids从train.json中获取instruction和output
        additional_contexts = []
        for b_id in b_ids:
            if b_id in train_data:
                train_item = train_data[b_id]
                instruction = train_item['instruction']
                output = train_item['output']
                additional_contexts.append(f"Instruction: {instruction}\nOutput: {output}")

        if additional_contexts:
            additional_context_text = "\n\n".join(additional_contexts)
            prompt += f"\n\nAdditional Contexts:\n{additional_context_text}"

    # 完成prompt的构建
    prompt += "\nPlease list all possible answers. The result may be a word or phrase. Do not return the thinking process."

    # 设置用户消息内容
    messages_template = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    
    # 调用GPT-3.5 Turbo模型
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
        print(f"Failed to get response for item ID {item_id}.")
