import json
import torch
from transformers import AutoTokenizer, AutoModel

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco')

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 计算句子嵌入
def get_embeddings(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

# 平均池化
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

# 计算余弦相似度
def cosine_similarity(embeddings1, embeddings2):
    return torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

# 主函数
def main():
    # 读取数据
    a_data = read_json('/root/yao/retrieval/eval_data_use/arc_challenge_processed.json')  
    b_data = read_json('/root/yao/retrieval/contriever_data/train.json') 

    results = []

    # 对于a_data中的每个问题
    for a_item in a_data:
        question = a_item["question"]
        
        # 计算问题的嵌入
        question_embedding = get_embeddings([question])

        b_scores = []

        # 对于b_data中的每个instruction
        for b_item in b_data:
            instruction = b_item["instruction"]
            
            # 计算instruction的嵌入
            instruction_embedding = get_embeddings([instruction])
            
            # 计算相似度
            score = cosine_similarity(question_embedding, instruction_embedding).item()
            
            b_scores.append((b_item["id"], score))
        
        # 选择相似度最高的10个
        top_10 = sorted(b_scores, key=lambda x: x[1], reverse=True)[:10]
        
        for b_id, score in top_10:
            results.append({
                "a_id": a_item["id"],
                "b_id": b_id,
                "similarity_score": score
            })
    
    # 保存结果到c.json
    with open('/root/yao/retrieval/index_build_save/arc_challenge_processed_index.json', 'w', encoding='utf-8') as outfile:  # 替换为实际路径
        json.dump(results, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
