import json
import os
from pathlib import Path

def cleanup_images(dataset_path, split):
    metadata_path = os.path.join(dataset_path, split, 'metadata.jsonl')
    folder_path = os.path.join(dataset_path, split)

    # Read metadata and collect file_names
    referenced_files = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                referenced_files.add(data['file_name'])

    # List all image files in the folder
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if Path(f).suffix.lower() in image_extensions]

    # Delete images not referenced
    deleted = []
    for img in image_files:
        if img not in referenced_files:
            os.remove(os.path.join(folder_path, img))
            deleted.append(img)

    print(f"Deleted {len(deleted)} unreferenced images in {split}: {deleted}")

# Assuming the dataset is at /workspace/production-data/Donut_format
dataset_path = '/workspace/production-data/Donut_format'

for split in ['train', 'validation', 'test']:
    cleanup_images(dataset_path, split)