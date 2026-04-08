#!/bin/bash
# setup_runpod.sh
# Script to install dependencies for running Qwen3.5 locally on a RunPod instance

set -e

echo "Starting environment setup for Qwen3.5 local evaluation..."

# 1. Update pip
echo "Updating pip..."
python3 -m pip install --upgrade pip

# 2. Install PyTorch ecosystem (CUDA 12.1)
echo "Installing PyTorch ecosystem..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install Unsloth and specialized PEFT/Quantization libraries (¡NUEVO Y CRÍTICO!)
echo "Installing Unsloth and its dependencies..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft bitsandbytes

# 4. Install core HuggingFace libraries
echo "Installing transformers and core NLP libraries..."
pip install "transformers>=4.40.0" datasets tiktoken accelerate

# 5. Install flash-attn for faster inference (highly recommended for Qwen3.5)
echo "Attempting to install flash-attn (this may take a few minutes if it needs to compile)..."
pip install flash-attn --no-build-isolation || echo "Warning: flash-attn installation failed. Inference will fall back to standard attention, which is slower but still works."

echo "✅ Environment setup complete! You are ready to run Qwen3.5 locally."
echo ""
echo "🔥 IMPORTANT: If you are using a Private Model, you must either:"
echo "   1. Run 'hf login' to save your token."
echo "   2. Export your token before running: 'export HF_TOKEN=your_token_here'"