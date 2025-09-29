import json
import time
import os
from pathlib import Path
from PIL import Image
import torch
import pytorch_lightning as pl
from lightning_module import DonutModelPLModule
from sconf import Config
import wandb
import weave

wandb.login(key='fa09a72f9dc3063f756c3f60300c431ca19d7218')
wandb.init(project='ravikumarshah-vebuin/minitron-donut')
weave.init(project_name='ravikumarshah-vebuin/minitron-donut')
from donut.donut.model import DonutModel

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

# Load the config
config_path = "result/train_custom_receipt/20250929_104457/config.yaml"
config = Config(config_path)

# Load the model
model_dir = "result/train_custom_receipt/20250929_104457"
model_pl = DonutModelPLModule(config)
model_pl.model = DonutModel.from_pretrained(model_dir)
model_pl.eval()
model_pl.to(torch.device('cpu'))
model = model_pl

# Test folder
test_folder = Path("vista-dat/Donut_format/validation")
metadata_file = test_folder / "metadata.jsonl"

# Load metadata
ground_truths = {}
with open(metadata_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        ground_truths[data['file_name']] = json.loads(data['ground_truth'])['gt_parse']

# Results
results = []
total_time = 0
num_images = 0

file_names = sorted([f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png')) and f in ground_truths])
for i, file_name in enumerate(file_names[:10]):
    image_path = test_folder / file_name
    image = Image.open(image_path)
    
    # Inference
    start_time = time.time()
    prompt = "<s_Donut_format>"
    result = model.model.inference(image=image, prompt=prompt)
    end_time = time.time()
    
    inference_time = end_time - start_time
    total_time += inference_time
    num_images += 1
    
    predicted = result["predictions"][0]
    gt = ground_truths[file_name]
    
    edit_dist = edit_distance(json.dumps(predicted, sort_keys=True), json.dumps(gt, sort_keys=True))
    wandb.log({'predicted_json': json.dumps(predicted)})
    if i < 5:
        wandb.log({'input_image': wandb.Image(image)})
    
    # Simple accuracy: exact match on JSON
    accuracy = 1 if predicted == gt else 0
    
    results.append({
        'file_name': file_name,
        'predicted': predicted,
        'ground_truth': gt,
        'inference_time': inference_time,
        'accuracy': accuracy,
        'edit_distance': edit_dist
    })

# Compute overall metrics
if num_images > 0:
    avg_latency = total_time / num_images
    overall_accuracy = sum(r['accuracy'] for r in results) / num_images
    avg_edit_distance = sum(r['edit_distance'] for r in results) / num_images
else:
    avg_latency = 0
    overall_accuracy = 0
    avg_edit_distance = 0

wandb.log({'avg_edit_distance': avg_edit_distance})

# Output results
output_file = "evaluation_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        'results': results,
        'summary': {
            'total_images': num_images,
            'average_latency': avg_latency,
            'overall_accuracy': overall_accuracy,
            'avg_edit_distance': avg_edit_distance
        }
    }, f, ensure_ascii=False, indent=4)

print(f"Evaluation complete. Results saved to {output_file}")
print(f"Total images: {num_images}")
print(f"Average latency: {avg_latency:.4f} seconds")
print(f"Overall accuracy: {overall_accuracy:.4f}")
print(f"Average edit distance: {avg_edit_distance:.4f}")
