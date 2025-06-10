import json
import random

# 配置路径和参数
input_file = 'TimeQA_Easy.json'
sample_size = 1000
output_file = f'TimeQA_Easy_subset_{sample_size}.json'
random_seed = 42  # 可更换为其他数值实现不同采样

# 读取原始数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]


# 检查样本数量是否足够
if len(data) < sample_size:
    raise ValueError(f"数据集中仅有 {len(data)} 条样本，无法抽取 {sample_size} 条。")

# 随机抽样
random.seed(random_seed)
subset = random.sample(data, sample_size)

# 保存新子集
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print(f"✅ 已成功从 {input_file} 中抽取 {sample_size} 条样本，并保存至 {output_file}")
