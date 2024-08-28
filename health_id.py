import json
import uuid

# 读取数据集文件
data_file_path = '/root/yao/retrieval/eval_data_use/health_claims_processed.json'
with open(data_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 为每条数据添加唯一的ID
for item in data:
    item['id'] = f"health_{uuid.uuid4().hex}"

# 将修改后的数据保存回文件
output_file_path = '/root/yao/retrieval/eval_data_use/health_claims_processed.json'
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

