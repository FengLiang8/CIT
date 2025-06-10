import json

def extract_matching_entries(original_file, question_list_file, output_file):
    # ✅ 读取多行 JSON 对象（非标准 JSON 文件）
    with open(original_file, "r", encoding="utf-8") as f:
        original_data = [json.loads(line) for line in f if line.strip()]

    # 构建索引：question -> 原始条目
    question_to_entry = {item["question"]: item for item in original_data}

    # 读取目标问题列表
    with open(question_list_file, "r", encoding="utf-8") as f:
        selected_questions = json.load(f)

    # 查找匹配条目
    matched = []
    not_found = []

    for item in selected_questions:
        question = item["question"]
        entry = question_to_entry.get(question)
        if entry:
            matched.append(entry)
        else:
            not_found.append(question)

    # 保存输出
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)

    print(f"✅ Found {len(matched)} matching entries. Saved to {output_file}")
    if not_found:
        print(f"⚠️  {len(not_found)} questions not found:")
        for q in not_found:
            print(" -", q)

# Example usage:
if __name__ == "__main__":
    extract_matching_entries(
        original_file="TimeQA_Easy.json",
        question_list_file="S300t100.json",
        output_file="N300t100.json"
    )
