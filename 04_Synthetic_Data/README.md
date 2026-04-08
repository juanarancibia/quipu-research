# 04_Synthetic_Data — Synthetic Data Generation Module

Generación de datos sintéticos para balancear y expandir el `golden_dataset.jsonl`.

## Strategies

### Strategy 1: Output → Input (Reverse Generation) ✅ Implemented

Toma outputs ya validados del dataset y genera inputs diversos en español argentino que producirían esas mismas transacciones.

```bash
# Ver el plan de generación sin ejecutar
python generate_reverse.py --dry-run

# Generar con auto-detección de modelo
python generate_reverse.py

# Generar con modelo específico (OpenRouter)
python generate_reverse.py --model "openrouter/openai/gpt-4o-mini"

# Limitar a N seeds, con N non-transactional, y appendear directo
python generate_reverse.py --limit 10 --non-tx 15 --append

# Todas las opciones
python generate_reverse.py \
  --model "openrouter/openai/gpt-4o-mini" \
  --limit 20 \
  --variants 3 \
  --non-tx 10 \
  --temperature 0.8 \
  --source-tag "synthetic_reverse_v2" \
  --append
```

### Strategy 2: Input → Output (Forward Generation) 🔜 Future

Genera inputs nuevos con un LLM y los etiqueta con un modelo DSPy optimizado al ~99%.

## Setup

```bash
cd 04_Synthetic_Data
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure `.env` with API keys:
```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...
```

## Module Structure

| File | Description |
|------|-------------|
| `config.py` | Auto-detects LLM provider from env vars |
| `balance_analyzer.py` | Identifies dataset gaps by category/feature |
| `generate_reverse.py` | Main CLI for Strategy 1 |
| `prompts/reverse_generation.py` | Prompt templates (Argentine WhatsApp style) |
| `validators/quality_checks.py` | Schema + amount validation |

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | auto | LiteLLM model string |
| `--limit` | all | Max seed entries to process |
| `--variants` | 3 | Input variants per seed |
| `--non-tx` | 0 | Non-transactional entries |
| `--temperature` | 0.8 | LLM creativity |
| `--dry-run` | - | Show plan only |
| `--append` | - | Append directly to golden_dataset.jsonl |
| `--source-tag` | synthetic_reverse | Metadata source tag |
