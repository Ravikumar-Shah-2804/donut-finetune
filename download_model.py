from transformers import DonutProcessor, VisionEncoderDecoderModel

# Download and cache the Donut processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Download and cache the Donut model
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

print("Donut model 'naver-clova-ix/donut-base' has been downloaded and cached locally.")