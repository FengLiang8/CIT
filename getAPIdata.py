import json
import random

# ========== Configuration ==========
full_path = "TimeQA_Easy.json"
subset_path = "TimeQA_Easy_subset_100.json"
output_path = "TimeQA_API_training_200.json"
seed = 42  # for reproducibility

# ========== Load Data ==========

with open(full_path, "r", encoding="utf-8") as f:
    full_data = [json.loads(line) for line in f if line.strip()]

with open(subset_path, "r", encoding="utf-8") as f:
    subset_data = json.load(f)

# ========== Identify Subset Questions ==========
subset_questions = set(item["question"] for item in subset_data)

# ========== Filter Out Subset ==========
remaining_data = [item for item in full_data if item["question"] not in subset_questions]

print(f"Remaining after exclusion: {len(remaining_data)}")

# ========== Randomly Sample 200 ==========
random.seed(seed)
sampled_data = random.sample(remaining_data, 200)

# ========== Save ==========
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved 200 non-overlapping samples to: {output_path}")
