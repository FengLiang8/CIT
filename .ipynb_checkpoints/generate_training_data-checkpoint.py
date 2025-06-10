import os
import json
import time
from openai import OpenAI
from tqdm import tqdm

# ======== Configuration ========
API_KEY = os.environ.get("DEEPSEEK_API_KEY")  # Set your key via environment variable
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

DATA_PATH = "TimeQA_API_training_200.json"               # Your input file with question/context/answer
PROMPT_TEMPLATE_PATH = "CI_prompt_train.txt"  # Your prompt template with placeholders
OUTPUT_PATH = "train_data_200.json"
SLEEP_SECONDS = 1.2                          # Delay to avoid rate limits

# ======== Load Model Client ========
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ======== Load Prompt Template ========
with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    prompt_template = f.read()

# ======== Load Dataset ========
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# ======== Generate for Each Sample ========
results = []
for i, item in enumerate(tqdm(dataset, desc="Generating")):
    question = item["question"]
    contexts = "\n".join(f"({s})" for s in item["TG"])  # Assuming "TG" is a list
    answer = item["answer"][0].strip()  # Assuming answer is a list like ["July 4"]

    # Fill prompt
    full_prompt = prompt_template.format(question=question, contexts=contexts, answer=answer)

    try:
        # API call
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            stream=False
        )
        reply = response.choices[0].message.content.strip()

        results.append({
            "id": i + 1,
            "question": question,
            "contexts": item["TG"],
            "answer": answer,
            "generated": reply
        })

    except Exception as e:
        print(f"[Error at {i}] {e}")
        continue

    time.sleep(SLEEP_SECONDS)

# ======== Save Output ========
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Generation completed. Output saved to {OUTPUT_PATH}")
