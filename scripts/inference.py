# File: scripts/inference.py

import torch
import instructor
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 1. Define the same Pydantic schema used in training ---
class UserInfo(BaseModel):
    name: str
    age: int
    email: str

# --- 2. Load the fine-tuned model and tokenizer ---
# The base model used during fine-tuning
base_model_name = "meta-llama/Meta-Llama-3-8B"

# The path to your saved adapter weights
adapter_path = "../models/llama-3-8b-json-extractor"

print("--- Loading base model ---")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("--- Loading tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print(f"--- Loading PEFT adapter from: {adapter_path} ---")
model = PeftModel.from_pretrained(base_model, adapter_path)

# --- 3. Patch the model with Instructor ---
# This enables the structured JSON output functionality
client = instructor.patch(
    client=model,
    tokenizer=tokenizer,
    mode=instructor.Mode.TOOLS,
)

# --- 4. Run Inference ---
# Create a new, unseen text input
new_text_input = "Please create a profile for Tom Anderson. He is 52 years old and his email is tom.a@corporation.com."

print(f"\n--- Running inference on new text: ---\n'{new_text_input}'")

# Use the patched client to get a structured response
response = client.chat.completions.create(
    model=base_model_name, # It's using the fine-tuned model under the hood
    response_model=UserInfo,
    messages=[
        {"role": "user", "content": new_text_input},
    ],
)

print("\n--- Extracted JSON Data: ---")
print(response.model_dump_json(indent=2))