import torch
from transformers import AutoTokenizer
from config import SLMConfig, SLMModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# carrega o modelo do checkpoint
model = SLMModel.from_pretrained("ldsv29/slm-midtraining")
model.eval()
model.to(device)

# tokenizer GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompts = [
    """prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"""
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    print(f"Prompt: {prompt}")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("-" * 60)