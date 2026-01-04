import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

MODEL_NAME = "liuhaotian/llava-v1.5-7b"

def load_dataset(jsonl_path):
    samples = []
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            user_msg = obj["conversations"][0]["value"]
            assistant_msg = obj["conversations"][1]["value"]

            text = f"USER: {user_msg}\nASSISTANT: {assistant_msg}"
            samples.append({"text": text})
    return samples


class LLAVADataset:
    def __init__(self, jsonl_path, tokenizer):
        self.data = load_dataset(jsonl_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.data[i]["text"],
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt"
        )
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in enc.items()}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    train_dataset = LLAVADataset("data/annotations/train.jsonl", tokenizer)
    val_dataset = LLAVADataset("data/annotations/validation.jsonl", tokenizer)

    training_args = TrainingArguments(
        output_dir="checkpoints/llava",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
