#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ---- CLI ----
parser = argparse.ArgumentParser(description="Llama 3 quick test with good defaults")
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B", help="HF model id")
parser.add_argument("--prompt", default="What are the top 3 benefits of using PyTorch for deep learning?")
parser.add_argument("--bullets", type=int, default=3, help="How many bullet points to return")
parser.add_argument("--max_new_tokens", type=int, default=120)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--repetition_penalty", type=float, default=1.2)
args = parser.parse_args()

# ---- Auth (env) ----
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
assert HF_TOKEN, "Set HUGGINGFACE_TOKEN env var (never hard-code tokens)."

# Optional: avoid torchvision import for text-only
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# ---- Torch perf knobs (5090) ----
# Use TF32 for faster matmul on post-Ampere GPUs; pick 'ieee' for strict accuracy
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.set_float32_matmul_precision("high")  # new API hint; harmless if future-changed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {args.model} …")
tok = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN)

# Provide Llama 3 chat template if missing (older transformers)
if not getattr(tok, "chat_template", None):
    tok.chat_template = (
        "<|begin_of_text|>{% for m in messages %}"
        "{% if m['role']=='system' %}{{ '<|start_header_id|>system<|end_header_id|>\\n\\n'+m['content']+'<|eot_id|>' }}"
        "{% elif m['role']=='user' %}{{ '<|start_header_id|>user<|end_header_id|>\\n\\n'+m['content']+'<|eot_id|>' }}"
        "{% elif m['role']=='assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n'+m['content']+'<|eot_id|>' }}{% endif %}"
        "{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    )

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
    attn_implementation="sdpa",  # avoids flash-attn/xformers hard deps
)
model.eval()

# Ensure pad token exists
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.eos_token_id

# ---- Build concise prompt ----
messages = [
    {"role": "system", "content": f"You are a concise expert. Answer in exactly {args.bullets} bullet points."},
    {"role": "user", "content": args.prompt},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

# ---- Decoding config (less repetition, concise) ----
gen_cfg = GenerationConfig(
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    repetition_penalty=args.repetition_penalty,
    no_repeat_ngram_size=4,
    eos_token_id=tok.eos_token_id,
)

with torch.inference_mode():
    out = model.generate(**inputs, generation_config=gen_cfg)

# Slice off prompt tokens
gen = out[0, inputs.input_ids.shape[-1]:]
text = tok.decode(gen, skip_special_tokens=True)

print("\n--- Model Response ---\n" + text)
