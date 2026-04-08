#!/bin/bash

# ==============================================================================
# QUIPU AI - RTX 5090 (RUNPOD PYTORCH 2.8 TEMPLATE)
# ==============================================================================

if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_API_KEY" ]; then
    echo "❌ ERROR: Faltan las API Keys."
    echo "Uso: HF_TOKEN='tu_token' WANDB_API_KEY='tu_key' bash run_pipeline.sh"
    exit 1
fi

SESSION_NAME="quipu_sft"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  La sesión '$SESSION_NAME' ya existe. Acoplándome..."
    sleep 2
    tmux attach -t $SESSION_NAME
    exit 0
fi

echo "🚀 Iniciando orquestador Quipu en sesión de Tmux: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# --- PASO 1: Optimizaciones de Atención ---
tmux send-keys -t $SESSION_NAME "echo '🛠️ Actualizando herramientas base...'; pip install --upgrade pip ninja wheel" C-m

tmux send-keys -t $SESSION_NAME "echo 'Instalando xformers (Fallback)...'; pip install --no-deps xformers" C-m

# 🚨 Compilamos Flash-Attn mucho más rápido porque el CUDA nativo ya es el 12.8
tmux send-keys -t $SESSION_NAME "echo 'Compilando Flash Attention 2 nativo...'; export MAX_JOBS=4; export TORCH_CUDA_ARCH_LIST=\"12.0\"; pip install flash-attn --no-build-isolation" C-m

# --- PASO 2: Frameworks de Entrenamiento ---
tmux send-keys -t $SESSION_NAME "echo '🦄 Instalando Unsloth y su Zoo...'; pip install unsloth-zoo \"unsloth @ git+https://github.com/unslothai/unsloth.git\"" C-m

tmux send-keys -t $SESSION_NAME "echo '📦 Instalando resto de dependencias...'; grep -v 'unsloth' 05_Fine_Tuning/requirements_training.txt > temp_reqs.txt && pip install -r temp_reqs.txt && rm temp_reqs.txt" C-m

# Forzamos la rama estable de transformers
tmux send-keys -t $SESSION_NAME "pip install transformers==4.57.6 httpx wandb huggingface_hub" C-m

# --- PASO 3: Logins No Interactivos ---
tmux send-keys -t $SESSION_NAME "echo '🔑 Iniciando sesión en W&B...'" C-m
tmux send-keys -t $SESSION_NAME "export WANDB_API_KEY=$WANDB_API_KEY" C-m
tmux send-keys -t $SESSION_NAME "wandb login" C-m

tmux send-keys -t $SESSION_NAME "echo '🔑 Iniciando sesión en Hugging Face...'" C-m
tmux send-keys -t $SESSION_NAME "huggingface-cli login --token $HF_TOKEN --add-to-git-credential" C-m

# --- PASO 4: Ejecución ---
tmux send-keys -t $SESSION_NAME "echo '🧪 Corriendo Dataset Formatter...'; python 05_Fine_Tuning/dataset_formatter.py" C-m
tmux send-keys -t $SESSION_NAME "echo '🔥 LANZANDO BAKEOFF PIPELINE...'; python 05_Fine_Tuning/bakeoff_pipeline.py --epochs 3" C-m

echo "✅ Todo configurado. Entrando a la sesión de Tmux..."
sleep 2
tmux attach -t $SESSION_NAME