import json

# 读取a.json文件
with open('/root/yao/retrieval/eval_data_predict_gpt_rag3/arc_challenge_processed.json', 'r') as file:
    a_data = [json.loads(line) for line in file]

# 读取b.json文件
with open('/root/yao/retrieval/eval_data_use/arc_challenge_processed.json', 'r') as file:
    b_data = json.load(file)

# 创建b.json数据的字典索引
b_dict = {item['id']: item for item in b_data}

# 处理a.json的result字段
for item in a_data:
    item['result'] = item['result'].split('.')[0]  # 去掉选项后面的符号或字符

# 统计准确度并记录错误结果
correct_count = 0
total_count = len(a_data)
errors = []

for item in a_data:
    a_id = item['id']
    a_result = item['result']
    
    if a_id in b_dict:
        b_answerKey = b_dict[a_id]['answerKey']
        if a_result == b_answerKey:
            correct_count += 1
        else:
            errors.append({
                "id": a_id,
                "predicted": a_result,
                "correct": b_answerKey
            })

# 计算准确度
accuracy = correct_count / total_count

# 输出准确度
print(f'Accuracy: {accuracy:.2%}')

# 将错误结果写入c.json文件
with open('/root/yao/retrieval/error_gpt_rag_analysis_3/arc_challenge_processed.json', 'w') as file:
    json.dump(errors, file, indent=4)

# 保存结果
print(f'Errors saved to c.json')
