import json
import re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "checkpoints/llava"

# MUST MATCH your categories in result.json
CATEGORY_NAMES = [
    "Accordion",
    "Action Controls",
    "Cards",
    "Carousels",
    "Chat Message",
    "Chatbot Interface",
    "Information Stamps",
    "Message Reactions",
    "Persistent Menu",
    "Quick Replies",
    "Typing Indicator",
    "Window Controls"
]

def extract_categories(text):
    """Extract all category names found in a text."""
    found = []
    for cat in CATEGORY_NAMES:
        if cat.lower() in text.lower():
            found.append(cat)
    return set(found)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    test_data = load_jsonl("data/annotations/test.jsonl")

    # Tracking metrics
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    overall_correct = 0
    overall_total = 0

    print("Evaluating...")

    for item in test_data:
        question = item["conversations"][0]["value"]
        gt_answer = item["conversations"][1]["value"]

        inputs = tokenizer(f"USER: {question}\nASSISTANT:", return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=60)
        pred_text = tokenizer.decode(out[0], skip_special_tokens=True)

        gt_set = extract_categories(gt_answer)
        pred_set = extract_categories(pred_text)

        # Update per-category accuracy
        for cat in gt_set:
            category_total[cat] += 1
            if cat in pred_set:
                category_correct[cat] += 1

        # Update overall accuracy
        if gt_set == pred_set:
            overall_correct += 1
        overall_total += 1

    # Print results
    print("\n==== OVERALL ACCURACY ====")
    print(f"{overall_correct}/{overall_total} = {overall_correct/overall_total*100:.2f}%")

    print("\n==== PER-CATEGORY ACCURACY ====")
    for cat in CATEGORY_NAMES:
        if category_total[cat] > 0:
            acc = category_correct[cat] / category_total[cat] * 100
            print(f"{cat:20s} : {category_correct[cat]}/{category_total[cat]} = {acc:.2f}%")
        else:
            print(f"{cat:20s} : No test samples")

    print("\nDone.")
