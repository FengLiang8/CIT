import json

# 路径配置
input_json = "train_data_200.json"
prompt_template_path = "CI_prompt_train.txt"
output_json = "trl_formatted_data.json"

# 读取提示模板
with open(prompt_template_path, "r", encoding="utf-8") as f:
    prompt_template = f.read().strip()

# 读取原始数据
with open(input_json, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 构造新格式数据
formatted_data = []
for example in raw_data:
    prompt = prompt_template.format(
        question=example["question"],
        answer=example["answer"],
        contexts="\n".join(example["contexts"])
    )
    response = example["generated"]

    formatted_data.append({
        "prompt": prompt,
        "response": response
    })

# 保存为新 JSON 文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"已保存为 {output_json}，共 {len(formatted_data)} 条样本")
