import json
import os
from pathlib import Path

def fix_metadata_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        data = json.loads(line.strip())
        if isinstance(data['ground_truth'], dict):
            data['ground_truth'] = json.dumps(data['ground_truth'])
        fixed_lines.append(json.dumps(data))

    with open(filepath, 'w') as f:
        for line in fixed_lines:
            f.write(line + '\n')

# Assuming the dataset is at /workspace/production-data/Donut_format
dataset_path = '/workspace/production-data/Donut_format'

for split in ['train', 'validation', 'test']:
    metadata_path = os.path.join(dataset_path, split, 'metadata.jsonl')
    if os.path.exists(metadata_path):
        print(f"Fixing {metadata_path}")
        fix_metadata_file(metadata_path)
    else:
        print(f"{metadata_path} not found")

print("Metadata files fixed.")