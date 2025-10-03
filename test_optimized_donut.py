import time
import os
from PIL import Image
from donut_model_optimized import DonutModel, DonutConfig

# Check if cache files exist
cache_files = ['cache/swin_base_patch4_window12_384.pth', 'cache/asian-bart-ecjk.pth']
for cache_file in cache_files:
    if os.path.exists(cache_file):
        print(f"Cache file {cache_file} exists.")
    else:
        print(f"Cache file {cache_file} does not exist.")

# Initialize model with default DonutConfig
config = DonutConfig()
start_time = time.time()
model = DonutModel(config)
load_time = time.time() - start_time
print(f"Model loading time: {load_time:.2f} seconds")

# Print device being used
print(f"Device: {model.device}")

# Confirm models are loaded successfully
print("Models loaded successfully.")

# Attempt simple inference test if image is available
image_path = 'donut_test/validation/32204031.jpg'
if os.path.exists(image_path):
    image = Image.open(image_path)
    prompt = "<s_receipt>"  # Assuming a receipt parsing prompt; adjust if needed
    try:
        result = model.inference(image=image, prompt=prompt)
        print("Inference test successful. Result:", result)
    except Exception as e:
        print(f"Inference test failed: {e}")
else:
    print("Model is ready for inference.")