import json

# 读取 a.json 文件
with open('/root/yao/retrieval/eval_data_predict_qwen_rag/triviaqa_test.json', 'r', encoding='utf-8') as file:
    a_data = [json.loads(line) for line in file]

# 读取 b.json 文件
with open('/root/yao/retrieval/eval_data_use/triviaqa_test.json', 'r', encoding='utf-8') as file:
    b_data = json.load(file)

# 创建一个字典来快速查找 b_data 中的数据
b_dict = {item['id']: item for item in b_data}

# 用于存储预测错误的结果
incorrect_predictions = []

# 统计正确和错误的数量
correct_count = 0
incorrect_count = 0

# 遍历 a.json 的数据进行匹配
for a_item in a_data:
    a_id = a_item['id']
    a_result = a_item['result'].lower()
    
    if a_id in b_dict:
        b_answers = [answer.lower() for answer in b_dict[a_id]['answers']]
        
        # 检查 a_result 是否匹配 b_answers 中的任意一个
        if any(answer in a_result for answer in b_answers):
            correct_count += 1
        else:
            incorrect_predictions.append(a_item)
            incorrect_count += 1
            
            
total_predictions = correct_count + incorrect_count
accuracy = correct_count / total_predictions if total_predictions > 0 else 0

print(f"处理完成，错误结果已保存到 c.json")
print(f"准确率: {accuracy:.2%}")

# 将错误的结果保存到 c.json
with open('/root/yao/retrieval/error_qwen_rag_analysis_1/triviaqa_test_qwen_base.json', 'w', encoding='utf-8') as file:
    json.dump(incorrect_predictions, file, ensure_ascii=False, indent=4)

print("处理完成，错误结果已保存到 c.json")
