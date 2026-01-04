import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer
import torch

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

def load_dataset(jsonl_path, image_dir):
    samples = []
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            # The user message now contains an <image> token, which is the standard for LLaVA
            user_msg = obj["conversations"][0]["value"]
            assistant_msg = obj["conversations"][1]["value"]
            image_file = obj.get("image")

            # Construct the prompt to be processed by the Llava processor
            prompt = f"USER: {user_msg}\nASSISTANT: {assistant_msg}"
            
            if image_file:
                samples.append({
                    "image_file": os.path.join(image_dir, image_file),
                    "prompt": prompt
                })
    return samples


class LLAVADataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, image_dir, processor):
        self.samples = load_dataset(jsonl_path, image_dir)
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        prompt = sample["prompt"]
        image_file = sample["image_file"]
        
        try:
            image = Image.open(image_file).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found {image_file}, skipping.")
            # Return the next valid item
            return self.__getitem__((i + 1) % len(self))
            
        # The processor handles tokenization of the prompt and preprocessing of the image.
        # It will correctly format the inputs for the multi-modal Llava model.
        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True, truncation=True)
        
        # Squeeze batch dimensions
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # The labels are the input_ids, which the model will use for language modeling loss.
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs


if __name__ == "__main__":
    # The processor bundles the tokenizer and image processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # Load the correct model for conditional generation
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

    train_dataset = LLAVADataset("data/annotations/train.jsonl", "data/images", processor)
    val_dataset = LLAVADataset("data/annotations/validation.jsonl", "data/images", processor)

    training_args = TrainingArguments(
        output_dir="checkpoints/llava",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        fp16=True, # fp16 is recommended for LLaVA
        remove_unused_columns=False, # Important for multi-modal inputs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
