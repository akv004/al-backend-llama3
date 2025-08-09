# File: scripts/config.py

from dataclasses import dataclass, field
from transformers import TrainingArguments
from peft import LoraConfig

@dataclass
class ModelConfig:
    """Configuration for the model."""
    base_model_name: str = "meta-llama/Meta-Llama-3-8B"
    new_model_path: str = "../models/llama-3-8b-json-extractor"

@dataclass
class DataConfig:
    """Configuration for the data."""
    dataset_path: str = "../data/training_dataset.json"
    dataset_text_field: str = "text"
    max_seq_length: int = 1024

@dataclass
class TrainingConfig:
    """Configuration for training arguments."""
    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="../models/training_output",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="tensorboard",
        )
    )

@dataclass
class PeftConfig:
    """Configuration for PEFT (LoRA)."""
    lora_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )