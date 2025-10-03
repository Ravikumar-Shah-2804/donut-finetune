# test_gpu.py
import torch

print("Attempting to use CUDA...")
try:
    if torch.cuda.is_available():
        print("SUCCESS: CUDA is available!")
        device = torch.device("cuda")
        x = torch.randn(3, 3).to(device)
        print("SUCCESS: Successfully created a tensor on the GPU.")
        print("Tensor on GPU:")
        print(x)
    else:
        print("FAILURE: CUDA is NOT available.")
except Exception as e:
    print(f"An error occurred: {e}")