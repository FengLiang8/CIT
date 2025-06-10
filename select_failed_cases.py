import json

def select_em0_f1low(input_file, output_file, max_samples=100):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove summary if present
    data = [d for d in data if isinstance(d, dict) and "question" in d]

    # 1. EM = 0 条目
    em0_samples = [item for item in data if item.get("exact_match") == 0]

    # 2. 如果不满 max_samples，按 F1 升序补齐
    if len(em0_samples) < max_samples:
        em1_or_other = [item for item in data if item not in em0_samples]
        em1_or_other.sort(key=lambda x: x.get("f1", 1.0))  # 默认 f1 为 1.0
        needed = max_samples - len(em0_samples)
        em0_samples.extend(em1_or_other[:needed])

    # 3. 只提取 question 和 ground_truth
    result = [{"question": item["question"], "ground_truth": item["ground_truth"]} for item in em0_samples[:max_samples]]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Selected {len(result)} samples saved to {output_file}")


# Example usage:
if __name__ == "__main__":
    select_em0_f1low("base_result_300.json", "S300t100.json")
