# File: scripts/train.py

import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer

# Import configurations from our new config file
from config import ModelConfig, DataConfig, TrainingConfig, PeftConfig

# --- 1. Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def format_prompt(example):
    """Formats a single example into a prompt for the model."""
    return {
        DataConfig.dataset_text_field: f"""### Instruction:
Extract user information from the text and format it as a JSON object.

### Input:
{example['input']}

### Response:
{example['output']}
"""
    }


def train():
    """Main function to run the training process."""

    # --- 2. Load Model and Tokenizer ---
    logging.info(f"Loading base model: {ModelConfig.base_model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        ModelConfig.base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 3. Load and Prepare Dataset ---
    logging.info(f"Loading and formatting dataset from: {DataConfig.dataset_path}")
    dataset = load_dataset("json", data_files=DataConfig.dataset_path, split="train")
    formatted_dataset = dataset.map(format_prompt)

    # --- 4. Initialize Trainer ---
    logging.info("Initializing SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=PeftConfig.lora_config,
        dataset_text_field=DataConfig.dataset_text_field,
        max_seq_length=DataConfig.max_seq_length,
        tokenizer=tokenizer,
        args=TrainingConfig.training_args,
        packing=False,
    )

    # --- 5. Start Training ---
    logging.info("Starting model training...")
    trainer.train()
    logging.info("Training finished.")

    # --- 6. Save the Final Model ---
    logging.info(f"Saving fine-tuned model to: {ModelConfig.new_model_path}")
    trainer.save_model(ModelConfig.new_model_path)
    logging.info("✅ Model saved successfully.")


if __name__ == "__main__":
    train()