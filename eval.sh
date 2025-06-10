python eval_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data TimeQA_Easy_subset_1000.json \
  --prompt base_prompt.txt \
  --output base_result_1000.json \
  --max_tokens 1024

python eval_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data TimeQA_Easy_subset_1000.json \
  --prompt CI_prompt.txt \
  --output CI_result_1000.json \
  --max_tokens 1024

python eval_sft_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data TimeQA_Easy_subset_1000.json \
  --prompt base_prompt.txt \
  --output sft_base_result_1000.json \
  --sft_model_dir sft_results/checkpoint-1000
  --max_tokens 1024

python eval_sft_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data TimeQA_Easy_subset_1000.json \
  --prompt CI_prompt.txt \
  --output sft_CI_result_1000.json \
  --sft_model_dir sft_results/checkpoint-1000
  --max_tokens 1024