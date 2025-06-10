import json
import re
import argparse

def validate_answers(input_file: str) -> None:
    """
    Validate model outputs in JSON by checking <answer> content against ground truth.
    Prints statistics for the first 100 actual samples only.
    """
    try:
        # Load JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Only evaluate the first 100 real samples (exclude summary if exists)
        actual_data = data[:100]

        total_items = len(actual_data)
        total_correct = 0
        total_with_answer_tag = 0
        correct_with_answer_tag = 0

        for item in actual_data:
            raw_output = item.get('raw_output', '')
            ground_truth = item.get('ground_truth', '').strip().lower()

            match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL | re.IGNORECASE)
            if match:
                total_with_answer_tag += 1
                extracted_answer = match.group(1).strip().lower()
                extracted_answer = re.sub(r'\s+', ' ', extracted_answer)

                is_correct = extracted_answer == ground_truth
                if is_correct:
                    total_correct += 1
                    correct_with_answer_tag += 1
            else:
                is_correct = False

        # Compute stats
        accuracy = total_correct / total_items if total_items > 0 else 0
        answer_tag_coverage = total_with_answer_tag / total_items if total_items > 0 else 0
        answer_tag_accuracy = correct_with_answer_tag / total_with_answer_tag if total_with_answer_tag > 0 else 0

        # Print results
        print("\n=== Evaluation Results (first 100 samples only) ===")
        print(f"Total samples evaluated: {total_items}")
        print(f"Correct answers (exact match): {total_correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"<answer> tag coverage: {answer_tag_coverage:.2%}")
        print(f"Accuracy within <answer> tag: {answer_tag_accuracy:.2%}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {input_file}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# ========== Argument Parser ==========
parser = argparse.ArgumentParser(description='Evaluate model outputs in JSON file')
parser.add_argument('--input', required=True, help='Path to input JSON file')
args = parser.parse_args()

validate_answers(args.input)
