# CIT Project File Description

This project includes a collection of scripts and data files used for processing, training, and evaluating large language models (LLMs), with a particular focus on time-contextual question answering (TimeQA).

## Main Scripts and File Functions

### Data Processing and Generation

* **`getAPIdata.py`**: Excludes a subset from the full dataset and randomly samples 200 entries from the remaining data to prepare for API training.
* **`sub_timeqa_easy.py`**: Randomly samples a specified number of examples from `TimeQA_Easy.json` to create a data subset.
* **`generate_training_data.py`**: Uses the DeepSeek API and a provided prompt template (`CI_prompt_train.txt`) to generate question-answer pairs based on the input data (`TimeQA_API_training_200.json`), saving the result to `train_data_200.json`.
* **`trl_formatted_data.py`**: Converts `train_data_200.json` into a format compatible with the TRL (Transformer Reinforcement Learning) library's SFTTrainer, saving it as `trl_formatted_data.json`. It builds prompts using `CI_prompt_train.txt`.
* **`extract_full_entries.py`**: Extracts full entries from a multi-line JSON file (e.g., `TimeQA_Easy.json`) based on a JSON file of questions (e.g., `S300t100.json`), saving them to a new JSON file (e.g., `N300t100.json`).

### Model Training

* **`CI_train.py`**: Performs LoRA fine-tuning on the Qwen2.5-0.5B-Instruct model using data from `trl_formatted_data.json` with SFTTrainer (Supervised Fine-tuning Trainer). Training parameters and PEFT configurations are defined within the script, and the fine-tuned model is saved in the `./sft_model` directory.
* **`trl_test.py`**: A test script for LoRA fine-tuning using the SFTTrainer on the `trl-lib/Capybara` dataset. Mainly used to validate the training workflow.

### Model Evaluation

* **`eval_model.py`**: Evaluates the performance of a base LLM on a QA dataset. It loads a specified Hugging Face model, uses the provided dataset and prompt template, generates answers, and computes Exact Match (EM) and F1 scores. Results are saved to a specified output JSON file.
* **`eval_sft_model.py`**: Evaluates an SFT (Supervised Fine-tuned) model’s performance on a QA dataset. It loads the base model and PEFT LoRA weights, merges them, and performs evaluation. Other functions are similar to `eval_model.py`.
* **`check_json.py`**: Validates the output JSON from model inference. It checks whether the contents within the `<answer>` tag match the true answers and prints statistics such as accuracy, `<answer>` tag coverage, and tag-internal accuracy for the first 100 samples.
* **`select_failed_cases.py`**: Extracts poorly performing samples (e.g., EM=0 or low F1 scores) from evaluation results, saving the question and true answers to a new JSON file for further analysis or data augmentation.
* **`eval.sh`**: A shell script to batch-run model evaluations. It invokes `eval_model.py` and `eval_sft_model.py` to evaluate both base and SFT models using different prompts on the `TimeQA_Easy_subset_1000.json` dataset, saving results to the `results` directory.

### Prompt Files

* **`base_prompt.txt`**: A basic prompt template containing placeholders for questions and context, used for inference.
* **`CI_prompt.txt`**: A more advanced prompt template guiding the model to use Chain of Thought (CoT) and Reflection for answering. It instructs the model to reason step-by-step in the `<causal_reasoning>` tag, reflect in the `<reflection>` tag, and give the final answer in the `<answer>` tag.
* **`CI_prompt_train.txt`**: Similar to `CI_prompt.txt`, but includes a `{answer}` placeholder to teach the model how to reason using CoT and Reflection given the correct answer during training.

### Data Files (Sample)

* **`TimeQA_API_training_200.json`**: Contains 200 QA pairs used for API training.
* **`TimeQA_Easy_subset_10.json` / `TimeQA_Easy_subset_100.json` / `TimeQA_Easy_subset_1000.json`**: Different-sized subsets of `TimeQA_Easy.json`.
* **`train_data_200.json`**: Training data with model-generated reasoning, created by `generate_training_data.py`.
* **`trl_formatted_data.json`**: Training data formatted for TRL SFTTrainer, created by `trl_formatted_data.py`.
* **`sft_CI_result_100.json` / `sft_base_result_100.json`**: Sample evaluation results of SFT models under different prompts.

### Others

* **`.gitignore`**: Specifies files to be ignored by Git version control, such as `TimeQA_Easy.json` (usually a large raw dataset).
* **`train_dataset.py`**: A simple script demonstrating how to make chat completion requests using the OpenAI (DeepSeek) API. Likely an early prototype or test script not directly tied to the main training pipeline.
* **`README.md`**: This file, providing an overview of the project structure and file functionalities.

## Directory Structure

* **`.ipynb_checkpoints/`**: Jupyter Notebook checkpoint directory.
* **`results/`**: Stores model evaluation results.

  * **`.ipynb_checkpoints/`**: Notebook checkpoints inside the `results` directory.
  * Files like `CI_result_10.json`, `CI_result_100.json`, `base_result_10.json`, etc., contain evaluation results under various conditions.
* **`sft_model/`**: Stores checkpoint files from SFT training.

  * `checkpoint-xxxx/`: Checkpoints for different training steps.

## Usage Instructions

1. **Data Preparation**:

   * Use `sub_timeqa_easy.py` to create subsets from raw data.
   * Use `getAPIdata.py` to prepare data for API calls.
   * Run `generate_training_data.py` (requires API key setup) to generate training data with reasoning.
   * Run `trl_formatted_data.py` to convert data for SFT training format.
2. **Model Training**:

   * Run `CI_train.py` to fine-tune the model using SFT.
3. **Model Evaluation**:

   * Use `eval.sh` to evaluate models in batch, or run `eval_model.py` / `eval_sft_model.py` individually.
   * Use `check_json.py` to validate the format and accuracy of evaluation results.
   * Use `select_failed_cases.py` to analyze failed cases.

Ensure you modify file paths and model names in scripts according to your specific setup.

---

Let me know if you'd like this in a formatted document or if you want help customizing it for publication (e.g., GitHub).

