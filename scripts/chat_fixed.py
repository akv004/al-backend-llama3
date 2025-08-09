#!/usr/bin/env python3
import os, sys, argparse, signal, json, torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from huggingface_hub import snapshot_download

# -------- CLI --------
p = argparse.ArgumentParser(description="Llama 3 chat (deterministic, cache-first, RTX 5090 friendly)")
p.add_argument("--base", default="meta-llama/Meta-Llama-3-8B",
               help="HF repo id or local path. If repo id, resolves to local cache first.")
p.add_argument("--adapter", default="", help="Path to LoRA/QLoRA adapter dir (optional)")
p.add_argument("--system", default="You are concise and accurate. Answer directly.")
p.add_argument("--max-new-tokens", type=int, default=200)
args = p.parse_args()

# -------- Auth / env --------
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
assert HF_TOKEN, "Set HUGGINGFACE_TOKEN env var."
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # text-only

# -------- Fast matmul on Hopper --------
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.manual_seed(42)

# -------- Resolve model (cache first) --------
def resolve_model_path(maybe_repo_or_path: str, token: str) -> str:
    if os.path.isdir(maybe_repo_or_path):
        return os.path.abspath(maybe_repo_or_path)
    try:
        return snapshot_download(maybe_repo_or_path, local_files_only=True, token=token)
    except Exception:
        return snapshot_download(maybe_repo_or_path, local_files_only=False, token=token)

print(f"Resolving base model: {args.base} …", flush=True)
base_path = resolve_model_path(args.base, HF_TOKEN)

# -------- Load tokenizer & model --------
print(f"Loading from: {base_path}", flush=True)
tok = AutoTokenizer.from_pretrained(base_path, token=HF_TOKEN, use_fast=True)

# Inject Llama-3 chat template if missing
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
    attn_implementation="sdpa",
)
model.eval()

# Optional: attach LoRA/QLoRA
if args.adapter:
    from peft import PeftModel
    adapter_path = os.path.abspath(args.adapter)
    print(f"Attaching adapter: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    try:
        model = model.merge_and_unload()
    except Exception as e:
        print(f"(warning) merge_and_unload failed: {e}; continuing without merge")

# Tokens
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.eos_token_id

# Stop on both </s> and <|eot_id|>
stop_ids = [tok.eos_token_id]
try:
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != -1:
        stop_ids = list(set(stop_ids + [eot_id]))
except Exception:
    pass

# -------- Conversation state --------
messages = [{"role": "system", "content": args.system}]

def build_inputs():
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt").to(model.device)

def gen_cfg():
    # deterministic decoding (no sampling), strict anti-repeat
    return GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,  # Lowered from 1.2 to be less aggressive
        # no_repeat_ngram_size=6,  # Disabled, can be too restrictive with greedy search
        eos_token_id=stop_ids,   # multiple stop tokens
        pad_token_id=tok.pad_token_id,
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
    print("\n(Interrupted)")
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

    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=cfg)

    # Capture reply (no streaming; single clean print)
    gen_tokens = out[0, inputs.input_ids.shape[-1]:]
    reply_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
    print(f"\nAssistant: {reply_text}")
    messages.append({"role": "assistant", "content": reply_text})
