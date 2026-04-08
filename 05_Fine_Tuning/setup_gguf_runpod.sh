#!/bin/bash

# ==============================================================================
# QUIPU AI - GGUF Packaging Setup (RUNPOD PYTORCH TEMPLATE)
# ==============================================================================

echo "🚀 Iniciando instalación de dependencias para empaquetado GGUF..."

# 1. Actualizar pip y herramientas base
echo "🛠️ Actualizando herramientas base..."
pip install --upgrade pip ninja wheel

# 2. Instalar xformers y Flash Attention
echo "📦 Instalando xformers y dependencias de aceleración..."
pip install --no-deps xformers
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="12.0"
pip install flash-attn --no-build-isolation

# 3. Instalar Unsloth
echo "🦄 Instalando Unsloth y su Zoo..."
pip install unsloth-zoo "unsloth @ git+https://github.com/unslothai/unsloth.git"

# 4. Instalar dependencias extra para HF Hub y modelado
echo "📦 Instalando resto de dependencias requeridas..."
pip install transformers==4.57.6 huggingface_hub

echo "✅ Instalación completada."
echo "Puedes ejecutar el empaquetado con:"
echo "python 05_Fine_Tuning/package_gguf.py --help"
