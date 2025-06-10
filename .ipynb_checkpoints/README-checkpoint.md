# CIT 项目文件说明

本项目包含一系列用于处理、训练和评估大型语言模型（LLM）的脚本和数据文件，特别关注基于时间上下文的问答（TimeQA）。

## 主要脚本和文件功能

### 数据处理与生成

- **`getAPIdata.py`**: 从完整数据集中排除一个子集，并从剩余数据中随机抽样200条，用于API训练数据准备。
- **`sub_timeqa_easy.py`**: 从 `TimeQA_Easy.json` 文件中随机抽取指定数量的样本，生成数据子集。
- **`generate_training_data.py`**: 使用 DeepSeek API 和提供的提示模板 (`CI_prompt_train.txt`)，根据输入数据 (`TimeQA_API_training_200.json`) 生成训练用的问答对，并将结果保存到 `train_data_200.json`。
- **`trl_formatted_data.py`**: 将 `train_data_200.json` 中的数据转换为 TRL (Transformer Reinforcement Learning) 库SFTTrainer所接受的格式，并保存为 `trl_formatted_data.json`。它使用 `CI_prompt_train.txt` 来构建prompt。
- **`extract_full_entries.py`**: 从一个原始的多行JSON文件（例如 `TimeQA_Easy.json`）中，根据一个问题列表JSON文件（例如 `S300t100.json`），提取出包含这些问题的完整条目，并保存到新的JSON文件（例如 `N300t100.json`）。

### 模型训练

- **`CI_train.py`**: 使用 `trl_formatted_data.json` 中的数据，通过SFTTrainer (Supervised Fine-tuning Trainer) 对 Qwen2.5-0.5B-Instruct 模型进行LoRA微调。训练参数和PEFT配置在此脚本中定义，训练好的模型保存在 `./sft_model` 目录下。
- **`trl_test.py`**: 一个测试脚本，使用 `trl-lib/Capybara` 数据集对 Qwen2.5-0.5B-Instruct 模型进行SFTTrainer的LoRA微调测试。主要用于验证训练流程。

### 模型评估

- **`eval_model.py`**: 评估基础LLM在问答数据集上的表现。它加载指定的Hugging Face模型，使用提供的数据集JSON文件和提示模板文件，生成回答并计算Exact Match (EM) 和 F1分数。结果保存在指定的输出JSON文件中。
- **`eval_sft_model.py`**: 评估经过SFT（Supervised Fine-tuning）微调后的LLM在问答数据集上的表现。它加载基础模型和PEFT LoRA权重，合并模型后进行评估，其余功能与 `eval_model.py` 类似。
- **`check_json.py`**: 验证模型输出的JSON文件。它检查输出中 `<answer>` 标签内的内容是否与真实答案匹配，并打印前100个样本的准确率、`<answer>` 标签覆盖率和标签内准确率等统计信息。
- **`select_failed_cases.py`**: 从评估结果JSON文件中筛选出表现不佳的样本（例如EM=0或F1较低的样本），提取问题和真实答案，保存为新的JSON文件，用于进一步分析或数据增强。
- **`eval.sh`**: 一个shell脚本，用于批量运行模型评估。它调用 `eval_model.py` 和 `eval_sft_model.py` 对不同的模型（基础模型和SFT模型）和提示（基础提示和CI提示）在 `TimeQA_Easy_subset_1000.json` 数据集上进行评估，并将结果保存到 `results` 目录下的相应JSON文件中。

### 提示文件

- **`base_prompt.txt`**: 一个基础的提示模板，包含问题和上下文的占位符，用于模型推理。
- **`CI_prompt.txt`**: 一个更复杂的提示模板，指导模型使用链式思考（Chain of Thought, CoT）和反思（Reflection）来回答问题。它要求模型在 `<causal_reasoning>` 标签内进行逐步推理，在 `<reflection>` 标签内进行反思，并在 `<answer>` 标签内给出最终答案。
- **`CI_prompt_train.txt`**: 与 `CI_prompt.txt` 类似，但额外包含一个 `{answer}` 占位符，用于在生成训练数据时，让模型学习如何根据已知答案进行链式思考和反思的推理过程。

### 数据文件 (部分示例)

- **`TimeQA_API_training_200.json`**: 包含200条用于API训练的问答数据。
- **`TimeQA_Easy_subset_10.json` / `TimeQA_Easy_subset_100.json` / `TimeQA_Easy_subset_1000.json`**: `TimeQA_Easy.json` 的不同大小的子集。
- **`train_data_200.json`**: 由 `generate_training_data.py` 生成的包含模型生成推理过程的训练数据。
- **`trl_formatted_data.json`**: 经过 `trl_formatted_data.py` 处理后，符合TRL库SFTTrainer输入格式的训练数据。
- **`sft_CI_result_100.json` / `sft_base_result_100.json`**: SFT模型在不同提示下的评估结果示例。

### 其他

- **`.gitignore`**: 指定Git版本控制忽略的文件，例如 `TimeQA_Easy.json` (通常是较大的原始数据集)。
- **`train_dataset.py`**: 一个简单的脚本，演示如何使用OpenAI API (DeepSeek) 进行聊天补全请求。看起来像是一个早期的测试或示例代码，可能与主要的训练流程不直接相关。
- **`README.md`**: 本文件，提供项目结构的概览和文件功能说明。

## 目录结构

- **`.ipynb_checkpoints/`**: Jupyter Notebook的检查点文件目录。
- **`results/`**: 存放模型评估结果的目录。
  - **`.ipynb_checkpoints/`**: Jupyter Notebook的检查点文件目录 (在results内)。
  - `CI_result_10.json`, `CI_result_100.json`, `base_result_10.json`, `base_result_100.json`, `base_result_1000.json`, `sft_CI_result_100.json`, `sft_base_result_100.json`: 不同条件下模型评估结果的JSON文件。
- **`sft_model/`**: 存放SFT模型训练检查点 (checkpoints) 的目录。
  - `checkpoint-xxxx/`: 各个训练步骤保存的模型权重和配置文件。

## 使用说明

1.  **数据准备**: 
    -   使用 `sub_timeqa_easy.py` 从原始数据创建子集。
    -   使用 `getAPIdata.py` 准备用于API调用的数据。
    -   运行 `generate_training_data.py` (需要配置API密钥) 生成包含推理过程的训练数据。
    -   运行 `trl_formatted_data.py` 将数据转换为SFT训练格式。
2.  **模型训练**: 
    -   运行 `CI_train.py` 进行SFT微调。
3.  **模型评估**: 
    -   使用 `eval.sh` 脚本批量评估模型，或单独运行 `eval_model.py` / `eval_sft_model.py`。
    -   使用 `check_json.py` 检查评估结果的格式和准确性。
    -   使用 `select_failed_cases.py` 分析评估失败的案例。

确保根据需要修改脚本中的文件路径和模型名称。