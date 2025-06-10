from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig
import json

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


with open("trl_formatted_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for example in data:
    example["completion"] = example.pop("response")

dataset = Dataset.from_list(data)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

training_args = TrainingArguments(
    output_dir="./sft_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=100,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()
