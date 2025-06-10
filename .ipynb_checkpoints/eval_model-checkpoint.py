import json
import re
import string
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

# ========== Normalize text for EM/F1 ==========
def normalize_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return " ".join(text.strip().split())

def compute_em(gt_answer, model_output):
    return int(normalize_text(gt_answer) in normalize_text(model_output))

def compute_f1(pred, gt):
    pred_tokens = normalize_text(pred).split()
    gt_tokens = normalize_text(gt).split()
    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# ========== Load Model ==========
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).cuda().eval()
    return model, tokenizer

# ========== Build Prompt from Template ==========
def build_prompt(question, TG, prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Format TG into context
    if isinstance(TG, list):
        contexts = "\n".join(f"({entry})" for entry in TG)
    elif isinstance(TG, str):
        contexts = TG.strip()
    else:
        raise ValueError("TG must be a list or a string.")

    values = {
        "question": question,
        "contexts": contexts
    }

    return prompt_template.format_map(values)

# ========== Generate Model Output + Token Tracking ==========
def generate(model, tokenizer, prompt, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )

    output_tokens = outputs.sequences.shape[1] - input_tokens
    response_ids = outputs.sequences[0][input_tokens:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response, input_tokens, output_tokens

# ========== Main Evaluation ==========
def evaluate(model, tokenizer, dataset_path, prompt_path, output_file, max_tokens):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    predictions = []
    total_em = total_f1 = 0
    total_input_tokens = total_output_tokens = 0

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        question = sample['question']
        TG = sample['TG']
        gt_answer = sample['answer'][0].strip()

        prompt = build_prompt(question, TG, prompt_path)
        model_output, input_tokens, output_tokens = generate(model, tokenizer, prompt, max_tokens)

        em = compute_em(gt_answer, model_output)
        f1 = compute_f1(model_output, gt_answer)

        total_em += em
        total_f1 += f1
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        predictions.append({
            "id": i + 1,
            "question": question,
            "ground_truth": gt_answer,
            "prompt": prompt,
            "raw_output": model_output,
            "exact_match": em,
            "f1": round(f1, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        })

    total = len(predictions)
    summary = {
        "total": total,
        "average_em": round(total_em / total * 100, 2),
        "average_f1": round(total_f1 / total * 100, 2),
        "average_input_tokens": round(total_input_tokens / total, 2),
        "average_output_tokens": round(total_output_tokens / total, 2),
        "average_total_tokens": round((total_input_tokens + total_output_tokens) / total, 2)
    }

    predictions.append({"summary": summary})

    print("\n=== Evaluation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on QA Dataset using TG and question only.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt template (must contain {question} and {contexts})")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens to generate")
    args = parser.parse_args()

    logging.set_verbosity_error()
    model, tokenizer = load_model(args.model)
    evaluate(model, tokenizer, args.data, args.prompt, args.output, args.max_tokens)

if __name__ == "__main__":
    main()
