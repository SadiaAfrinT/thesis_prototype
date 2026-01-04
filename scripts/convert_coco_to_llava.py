import json
from collections import defaultdict
import os

# Correct paths for your project
IMAGE_ROOT = "data/images"
COCO_JSON_PATH = "data/raw/result.json"
OUTPUT_JSONL_PATH = "data/annotations/llava_dataset.jsonl"

def clean_filename(raw_filename):
    """
    Fixes Label Studio filenames like:
      'a09df23c-Air_India_1.png'
    →   'Air_India_1.png'
    """
    raw_filename = os.path.basename(raw_filename)

    # If filename contains a hash prefix like "a09df23c-"
    if "-" in raw_filename:
        return raw_filename.split("-", 1)[1]  # keep only the real filename
    return raw_filename


def load_coco(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def convert(coco, output_path=OUTPUT_JSONL_PATH):
    # Map category_id → name
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Map image_id → image info
    images = {img["id"]: img for img in coco["images"]}

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as out:
        for image_id, img in images.items():

            # Fix the filename (remove Label Studio prefix)
            cleaned_filename = clean_filename(img["file_name"])

            annotations = anns_by_image.get(image_id, [])

            # Natural language element list
            description = "This interface contains the following UI elements:\n"
            for ann in annotations:
                cat_name = cat_map[ann["category_id"]]
                bbox = ann["bbox"]
                description += f"- {cat_name} at {bbox}\n"

            # Build LLaVA training sample
            item = {
                "image": cleaned_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": "Describe all UI components in this interface with their positions."
                    },
                    {
                        "from": "assistant",
                        "value": description.strip()
                    }
                ]
            }

            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved LLaVA dataset to: {output_path}")


if __name__ == "__main__":
    print("Loading COCO dataset:", COCO_JSON_PATH)
    coco = load_coco(COCO_JSON_PATH)
    convert(coco)
    print("Done.")
