import json
import re

# 读取 a.json 文件
with open('/root/yao/retrieval/eval_data_predict_qwen_rag/health_claims_processed.json', 'r') as a_file:
    a_data = [json.loads(line) for line in a_file]

# 读取 b.json 文件
with open('/root/yao/retrieval/eval_data_use/health_claims_processed.json', 'r') as b_file:
    b_data = json.load(b_file)

# 处理 a.json 中的 result 字段
def process_result(result):
    # 只保留 true 或 false，并忽略大小写
    result = re.search(r'\b(true|false)\b', result, re.IGNORECASE)
    if result:
        return 1 if result.group(0).lower() == 'true' else 0
    else:
        return None

# 建立 b.json 中的 id 到数据的映射
b_data_map = {item['id']: item for item in b_data}

# 初始化计数器
correct_predictions = 0
total_predictions = 0
incorrect_results = []

# 进行比对并计算准确率
for a_item in a_data:
    a_id = a_item['id']
    processed_result = process_result(a_item['result'])
    
    if processed_result is not None and a_id in b_data_map:
        b_item = b_data_map[a_id]
        b_answers = [1 if answer.lower() == 'true' else 0 for answer in b_item['answers']]
        
        if processed_result in b_answers:
            correct_predictions += 1
        else:
            incorrect_results.append({
                'id': a_id,
                'expected': b_answers,
                'predicted': processed_result
            })
        total_predictions += 1

# 计算准确率
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

# 保存预测错误的结果到 c.json
with open('/root/yao/retrieval/error_qwen_rag_analysis_1/health_claims_processed_qwen_base.json', 'w') as c_file:
    json.dump(incorrect_results, c_file, indent=4)

print(f'Accuracy: {accuracy:.2%}')
