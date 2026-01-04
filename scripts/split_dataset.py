
import json
import random
import os

def split_data(input_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    input_file = 'data/annotations/llava_dataset.jsonl'
    train_file = 'data/annotations/train.jsonl'
    val_file = 'data/annotations/validation.jsonl'
    test_file = 'data/annotations/test.jsonl'
    
    split_data(input_file, train_file, val_file, test_file)

    print(f"Data split into {train_file}, {val_file}, and {test_file}")
