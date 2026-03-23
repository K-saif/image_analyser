import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image

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

# ------------------ LOAD IMAGE ------------------
image_path = r"C:\Users\khans\OneDrive\Documents\image_analyser\runs\detect\predict3\5.jpg"
image = Image.open(image_path).convert("RGB")

# ------------------ CHAT LOOP ------------------
print("\n💬 Ask questions about the image (type 'exit' to quit)\n")

while True:
    prompt = input("You: ")

    if prompt.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # conversation with image token
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # process input
    inputs = processor(
        text=processor.apply_chat_template(conversation, add_generation_prompt=True),
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # inference
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

    # decode
    result = processor.decode(output[0], skip_special_tokens=True)

    # clean response (remove USER/ASSISTANT if present)
    if "ASSISTANT:" in result:
        result = result.split("ASSISTANT:")[-1].strip()

    print(f"\n🤖: {result}\n")

