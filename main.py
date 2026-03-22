import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests

# ------------------ CONFIG ------------------
model_id = "llava-hf/llava-1.5-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ------------------ LOAD MODEL ------------------
print("Loading model...")

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    max_memory={0: "7GiB", "cpu": "20GiB"}
)

processor = AutoProcessor.from_pretrained(model_id)

print("Model loaded ✅")

from PIL import Image
import requests

# ------------------ LOAD IMAGE ------------------
image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# ------------------ PROMPT ------------------
prompt = "What is shown in this image? Be specific."

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    },
]

# ------------------ PROCESS INPUT ------------------
inputs = processor(
    text=processor.apply_chat_template(conversation, add_generation_prompt=True),
    images=image,
    return_tensors="pt"
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# ------------------ GENERATE ------------------
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )

# ------------------ DECODE ------------------
result = processor.decode(output[0], skip_special_tokens=True)

print("\n🧠 Model Output:\n")
print(result)

# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")