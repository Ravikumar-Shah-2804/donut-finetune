#!/usr/bin/env python3
"""
Script to pre-build Donut vocabulary with special tokens dynamically extracted from JSONL files.
This initializes the XLMRobertaTokenizer from asian-bart-ecjk and adds essential special tokens
based on unique keys found in the gt_parse objects within the JSONL metadata files.
"""

import json
import os
import glob
from transformers import XLMRobertaTokenizer

def extract_keys_from_jsonl(file_path):
    """
    Extract all unique keys from the gt_parse objects in a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        set: Set of unique keys found in gt_parse.
    """
    keys = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'ground_truth' in data:
                    gt_str = data['ground_truth']
                    gt = json.loads(gt_str)
                    if 'gt_parse' in gt:
                        keys.update(gt['gt_parse'].keys())
            except (json.JSONDecodeError, KeyError):
                # Skip invalid lines or missing keys
                continue
    return keys

def build_vocabulary(jsonl_root_dir, output_dir="./prebuilt_vocab"):
    """
    Build and save the Donut tokenizer with special tokens dynamically extracted from JSONL files.

    Args:
        jsonl_root_dir (str): Root directory to scan for JSONL files.
        output_dir (str): Directory to save the tokenizer.
    """
    # Find all .jsonl files recursively
    jsonl_files = glob.glob(os.path.join(jsonl_root_dir, '**', '*.jsonl'), recursive=True)

    if not jsonl_files:
        print(f"No JSONL files found in {jsonl_root_dir}")
        return

    all_keys = set()
    for file_path in jsonl_files:
        keys = extract_keys_from_jsonl(file_path)
        all_keys.update(keys)

    if not all_keys:
        print("No keys extracted from JSONL files")
        return

    # Initialize the base tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")

    # Add essential special tokens
    special_tokens = ["<sep/>"]  # Used for representing lists in JSON

    # Add special tokens for each unique key found
    for key in sorted(all_keys):
        special_tokens.extend([f"<s_{key}>", f"</s_{key}>"])

    # Add the special tokens to the tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Extracted {len(all_keys)} unique keys from {len(jsonl_files)} JSONL files")
    print(f"Added {len(special_tokens)} special tokens:")
    for token in special_tokens:
        print(f"  {token}")

if __name__ == "__main__":
    # Scan from the current directory for JSONL files
    build_vocabulary(".")