#!/usr/bin/env python3
# python scripts/chat.py \
#   --base meta-llama/Meta-Llama-3-8B \
#   --system "You are concise and accurate. Answer directly." \
#   --no-sample \
#   --temperature 0 \
#   --top-p 1.0 \
#   --top-k 0 \
#   --repetition-penalty 1.2 \
#   --max-new-tokens 200


import os, sys, argparse, signal, json, torch
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
)
from huggingface_hub import snapshot_download

# ---------------- CLI ----------------
p = argparse.ArgumentParser(description="Llama 3 chat (cache-first, RTX 5090 friendly)")
p.add_argument("--base", default="meta-llama/Meta-Llama-3-8B",
               help="Base model repo id OR local path. If repo id, resolves to local cache first.")
p.add_argument("--adapter", default="", help="Path to LoRA/QLoRA adapter dir (optional)")
p.add_argument("--system", default="You are a helpful AI assistant.", help="System prompt")
p.add_argument("--max-new-tokens", type=int, default=512)
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--top-p", type=float, default=0.9)
p.add_argument("--top-k", type=int, default=50)
p.add_argument("--repetition-penalty", type=float, default=1.1)
p.add_argument("--no-sample", action="store_true", help="Disable sampling (greedy decode)")
args = p.parse_args()

# ---------------- Auth / env ----------------
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
assert HF_TOKEN, "Set HUGGINGFACE_TOKEN env var (never hard-code tokens)."
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # text-only, skip vision deps

# ---------------- Torch knobs (great on 5090) ----------------
# New API for TF32 control (fast matmul on Hopper/Ampere+)
torch.backends.cuda.matmul.fp32_precision = "tf32"

# ---------------- Resolve model path (cache-first) ----------------
def resolve_model_path(maybe_repo_or_path: str, token: str) -> str:
    if os.path.isdir(maybe_repo_or_path):
        return os.path.abspath(maybe_repo_or_path)
    # Try local cache first (offline-friendly)
    try:
        return snapshot_download(maybe_repo_or_path, local_files_only=True, token=token)
    except Exception:
        # Fallback to download if not cached
        return snapshot_download(maybe_repo_or_path, local_files_only=False, token=token)

print(f"Resolving base model: {args.base} …", flush=True)
base_path = resolve_model_path(args.base, HF_TOKEN)

# ---------------- Load tokenizer/model ----------------
print(f"Loading from: {base_path}", flush=True)
tok = AutoTokenizer.from_pretrained(base_path, token=HF_TOKEN, use_fast=True)

# Inject Llama 3 chat template if missing (older transformers)
if not getattr(tok, "chat_template", None):
    tok.chat_template = (
        "<|begin_of_text|>{% for m in messages %}"
        "{% if m['role']=='system' %}{{ '<|start_header_id|>system<|end_header_id|>\\n\\n' + m['content'] + '<|eot_id|>' }}"
        "{% elif m['role']=='user' %}{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + m['content'] + '<|eot_id|>' }}"
        "{% elif m['role']=='assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + m['content'] + '<|eot_id|>' }}"
        "{% endif %}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    )

model = AutoModelForCausalLM.from_pretrained(
    base_path,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",  # no FlashAttn/xformers dependency
)
model.eval()

# Optional: attach LoRA/QLoRA adapter
if args.adapter:
    from peft import PeftModel
    adapter_path = os.path.abspath(args.adapter)
    print(f"Attaching adapter: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    try:
        model = model.merge_and_unload()  # best perf at inference
    except Exception as e:
        print(f"(warning) merge_and_unload failed: {e}; continuing without merge")

# Ensure pad token
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.eos_token_id

# ---------------- Conversation state ----------------
messages = [{"role": "system", "content": args.system}]
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

def build_inputs():
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt").to(model.device)

def gen_cfg():
    return GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=4,
        eos_token_id=tok.eos_token_id,
    )

def save_history(path=None):
    if not path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = f"chat_{ts}.jsonl"
    with open(path, "w") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved chat history to {path}")

def reset(sys_prompt=None):
    sys_prompt = sys_prompt or args.system
    messages.clear()
    messages.append({"role": "system", "content": sys_prompt})
    print("(conversation reset)")

def _sigint(_s, _f):
    print("\n(Interrupted)")  # keep partial output, return to prompt
signal.signal(signal.SIGINT, _sigint)

print("Commands: /reset, /save [file], /quit")
while True:
    try:
        user = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not user:
        continue

    # Commands
    if user.lower() in {"/quit", "/exit"}:
        print("Goodbye!")
        break
    if user.lower().startswith("/save"):
        parts = user.split(maxsplit=1)
        save_history(parts[1] if len(parts) == 2 else None)
        continue
    if user.lower() == "/reset":
        reset()
        continue

    # Normal turn
    messages.append({"role": "user", "content": user})
    inputs = build_inputs()
    cfg = gen_cfg()

    print("\nAssistant: ", end="", flush=True)
    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=cfg, streamer=streamer)

    # Store reply in history (streamer already printed; don't print again)
    gen_tokens = out[0, inputs.input_ids.shape[-1]:]
    reply_text = tok.decode(gen_tokens, skip_special_tokens=True)
    messages.append({"role": "assistant", "content": reply_text})
