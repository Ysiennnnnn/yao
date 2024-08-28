import json
import os
from http import HTTPStatus
import dashscope

# 设置API密钥
dashscope.api_key = 'sk-64de2eee2c81419587d35b5fb6b5394f'

# 数据集文件路径
data_file_path = '/root/yao/retrieval/eval_data_use/arc_challenge_processed.json'
# 结果保存文件路径
results_file_path = '/root/yao/retrieval/eval_data_predict/arc_challenge_processed.json'

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
    choices = item['choices']['text']
    labels = item['choices']['label']
    
    # 构建prompt
    choices_str = "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices))])
    prompt = f"Question: {question}\nChoices:\n{choices_str}\nPlease choose directly from A B C D. Do not return to the thought process, just the letter."
    
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
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
