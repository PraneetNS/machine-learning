from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("image.jpg")
inputs = processor(text=["a dog", "a cat"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits = outputs.logits_per_image
print(logits)