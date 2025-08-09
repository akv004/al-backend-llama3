# AI Backend with Llama 3 for Structured JSON Output

This project is a proof-of-concept for creating a flexible AI-powered backend. The goal is to fine-tune a **Large Language Model (LLM)** to understand unstructured text and reliably return **structured JSON** data that can be used by any front-end application.

## Current Task: User Information Extraction

The current model is being fine-tuned to perform a specific task:
-   **Input:** A block of natural language text containing a user's name, age, and email.
-   **Output:** A clean, validated JSON object containing the extracted `name`, `age`, and `email`.

## Project Structure

```bash
AI_Backend_Llama3/
├── data/
│   └── training_dataset.json
├── models/
│   ├── llama-3-8b-json-extractor/
│   └── training_output/
├── scripts/
│   ├── init.py
│   ├── config.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── inference.py
│   └── test_base_model.py
└── requirements.txt
```

# Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd AI_Backend_Llama3
    ```

2.  **Create Conda Environment**
    ```bash
    conda create -n llama3-project python=3.10 -y
    conda activate llama3-project
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face Login**
    You will need a Hugging Face token with access to Llama 3.
    ```bash
    huggingface-cli login
    ```

## Workflow / Usage

Follow these steps in order to prepare the data, train the model, and test it.

### 1. Prepare the Dataset
This script uses a Pydantic model to generate a small, structured training dataset.
```bash
python scripts/prepare_data.py
```

### 2. Train the Model
This script loads the base Llama 3 8B model and fine-tunes it on the dataset created in the previous step using QLoRA.

```bash
python scripts/train.py
```

After training, the new model adapter will be saved in the `models/llama-3-8b-json-extractor` directory.

### 3. Test the Fine-Tuned Model (Inference)
This script loads your fine-tuned model and tests its ability to extract JSON from a new, unseen sentence.

```bash
python scripts/inference.py
```

### Configuration
All hyperparameters, model names, and file paths can be easily modified in the `scripts/config.py` file.

---

## Troubleshooting

### 🛠 RTX 5090 (sm_120) + PyTorch “no kernel image is available” Fix

#### The Issue
When running PyTorch code on an **NVIDIA GeForce RTX 5090** (Hopper Next architecture, compute capability `sm_120`), we hit:

```
RuntimeError: no kernel image is available for execution on the device
```

**Why it happened:**
- The default PyTorch stable wheels did not yet include compiled kernels for `sm_120`.
- Some third-party libs (e.g., `xformers`, `FlashAttention`) were pinned to older PyTorch versions and didn’t know about `sm_120` yet.
- Hugging Face `transformers` default tried to use FlashAttention 2 by default, which failed without the package.

#### The Fix

**1. Use a PyTorch nightly with CUDA 12.8 (cu128) that includes `sm_120` support**
```bash
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**2. Remove or adjust conflicting packages**
- Removed `xformers` that was hard-pinning an older torch:
  ```bash
  pip uninstall -y xformers
  ```
- Set `attn_implementation="sdpa"` when loading the model to avoid FlashAttention 2 dependency for now.
- Optional: 
  ```bash
  export TRANSFORMERS_NO_TORCHVISION=1
  ```
  to skip image libs when working text-only.

**3. Update Hugging Face libraries**
```bash
pip install -U "transformers>=4.44" accelerate safetensors sentencepiece
```

**4. Safe token handling**
Instead of hard-coding Hugging Face tokens in scripts, store in an env var:
```bash
export HUGGINGFACE_TOKEN=hf_xxx...
```
Load in Python:
```python
import os
hf_token = os.environ["HUGGINGFACE_TOKEN"]
```

**5. Adjust script defaults for stable output**
- Use `bfloat16` for best performance on 5090:
  ```python
  torch_dtype=torch.bfloat16
  ```
- Enable TF32 matmul for faster training/inference:
  ```python
  torch.backends.cuda.matmul.fp32_precision = "tf32"
  torch.set_float32_matmul_precision("high")
  ```
- Apply `GenerationConfig` with:
  ```python
  repetition_penalty=1.2
  no_repeat_ngram_size=4
  ```
  and explicit bullet-point instructions to avoid repeated text.

#### Result
After these steps:
- ✅ The model loads successfully on RTX 5090 with no CUDA kernel errors.
- ✅ Inference runs at full GPU speed with clean, concise output.
- ✅ Environment is ready for fine-tuning and inference on Hopper Next cards.
