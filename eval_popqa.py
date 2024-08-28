import json

# 读取a.json的数据
with open('/root/yao/retrieval/eval_data_predict_qwen_rag/popqa_longtail.json', 'r', encoding='utf-8') as file_a:
    a_data = [json.loads(line) for line in file_a]

# 读取b.json的数据
with open('/root/yao/retrieval/eval_data_use/popqa_longtail.json', 'r', encoding='utf-8') as file_b:
    b_data = json.load(file_b)

# 创建一个字典，用于快速查找b.json中的数据
b_dict = {item['id']: item for item in b_data}

# 计数器
correct_count = 0
incorrect_count = 0

# 保存错误预测的结果
incorrect_predictions = []

# 遍历a.json的数据进行匹配
for a_item in a_data:
    a_id = a_item['id']
    a_result = a_item['result'].lower()  # 转为小写以便匹配
    if a_id in b_dict:
        b_item = b_dict[a_id]
        answers = [answer.lower() for answer in b_item['answers']]  # 转为小写以便匹配
        
        # 检查a_result是否包含在answers中
        if any(answer in a_result for answer in answers):
            correct_count += 1
        else:
            incorrect_count += 1
            incorrect_predictions.append(a_item)
    else:
        incorrect_count += 1
        incorrect_predictions.append(a_item)

# 计算准确率
total_count = correct_count + incorrect_count
accuracy = correct_count / total_count if total_count > 0 else 0

# 打印准确率
print(f"Accuracy: {accuracy:.2%}")

# 将错误预测的结果保存到c.json
with open('/root/yao/retrieval/error_qwen_rag_analysis_1/popqa_longtail_qwen_base.json', 'w', encoding='utf-8') as file_c:
    json.dump(incorrect_predictions, file_c, ensure_ascii=False, indent=4)
