import subprocess
import os

# ------------------- ⚙️ CONFIGURATION -------------------
# Using zsh as the designated shell.

PROJECT_PATH = os.path.expanduser("~/projects/PycharmProjects/AI_Backend_Llama3")
CONDA_ENV = "llama3-finetune"
SHELL = "zsh"  # <--- CORRECTED FOR YOU

# ------------------- 🚀 SCRIPT LOGIC -------------------

# Command sequence to be run inside zsh
commands = (
    f"conda activate {CONDA_ENV} && "
    f"cd -P '{PROJECT_PATH}' && "
    f"exec {SHELL}"
)

# The final command that launches Tilix with a zsh shell
tilix_command = ["tilix", "-e", f"{SHELL} -c \"{commands}\""]

print(f"▶️  Executing: {' '.join(tilix_command)}")

try:
    subprocess.Popen(tilix_command)
    print("✅ Tilix session launched!")
except FileNotFoundError:
    print("❌ Error: 'tilix' command not found.")
    print("   Please ensure Tilix is installed and accessible in your system's PATH.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")