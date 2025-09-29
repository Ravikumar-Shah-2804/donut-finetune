import streamlit as st
from PIL import Image
import torch
import pytorch_lightning as pl
import wandb
import weave
import time

wandb.login(key='fa09a72f9dc3063f756c3f60300c431ca19d7218')
wandb.init(project='ravikumarshah-vebuin/minitron-donut')
weave.init(project_name='ravikumarshah-vebuin/minitron-donut')
from donut.lightning_module import DonutModelPLModule

# Load the model from checkpoint
checkpoint_path = "result/train_custom_receipt/20250929_104457/epoch=29-step=2640.ckpt"
model = DonutModelPLModule.load_from_checkpoint(checkpoint_path)
model.eval()
model.to(torch.device('gpu'))  # Use CPU for inference

# Streamlit app
st.title("Receipt OCR with Fine-tuned Donut Model")

st.write("Upload a receipt image to extract the JSON output.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image (resize to 256x256 and convert to tensor is handled in prepare_input)
        # Run inference
        prompt = "<s_Donut_format>"
        result = model.model.inference(image=image, prompt=prompt)

        # Extract and display JSON
        json_output = result["predictions"][0]
        st.subheader("Extracted JSON Output:")
        st.json(json_output)

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.write("Please ensure the uploaded file is a valid image and try again.")

total_latency = time.time() - start_time
wandb.log({'end_to_end_latency': total_latency})