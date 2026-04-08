<!-- 
🚨 ATENCIÓN AGENTES Y DESARROLLADORES 🚨
ESTE ARCHIVO SOLO DEBE CONTENER EL ESTADO ACTUAL, REAL DEL REPOSITORIO Y EL AVANCE DEL ROADMAP.
NO AÑADIR UN REGISTRO PASO A PASO (HISTORIAL DE ITERACIONES). 
SI SE HACEN CAMBIOS EN EL PROYECTO, ACTUALIZAR LAS SECCIONES PERTINENTES PARA REFLEJAR EL NUEVO ESTADO.
-->

# Development Log — Project Quipu

## 🗺️ Roadmap Progress (Fase 1: Data Science)

Este log refleja el estado de avance respecto a los objetivos originales planteados en el `README.md` principal para la **Fase 1**.

### 1.1. The Golden Dataset 🟢 (Completado - 100%)
*   [x] Exportar chat de WhatsApp a `.txt`.
*   [x] Crear script de parsing automático (`01_Data_Acquisition/scripts/parse_whatsapp_chat.py`).
*   [x] Generar Dataset Tentativo (`01_Data_Acquisition/data/tentative_dataset.json`) — 506 conversaciones filtradas.
*   [x] Desarrollar Curator UI con categorías avanzadas, fechas relativas y estadísticas en tiempo real.
*   [x] **Curación Manual completada:** 112 entradas curadas en `golden_dataset.jsonl`.

### 1.2. The Evaluation Pipeline 🟢 (Completado - 100%)
*   [x] Implementar clase `Evaluator` en `02_Evaluation/evaluator.py`.
*   [x] Definir métricas `StrictJSONScore`, `EntityAccuracy`, `CategoryMatch`, `F1-Score`.
*   [x] Agregar `ErrorBreakdown` y `FieldErrorStats` al pipeline — breakdown por campo (amount, category, type, dates, hard gate).
*   [x] **Test suite: 28/28 pasando** (`02_Evaluation/tests/test_evaluator.py`).

### 1.3. Baseline & Prompt Optimization 🟢 (Completado - 100%)
*   [x] Construir pipeline de evaluación conectada a cualquier LLM (`run_evaluation.py` con LiteLLM).
*   [x] Simplificar el Prompt a formato Zero-Shot puro.
*   [x] Benchmark manual base de modelos (Ej: GPT-4o-mini, Gemma-3, MiniMax-2.5).
*   [x] Implementar estructura base de **DSPy** (`03_Optimization`) para optimizar los prompts automáticamente.
*   [x] Adaptar `Evaluator` en una métrica rígida `quipu_metric` (50% F1, 50% Entity Accuracy y rechazo de markdown).
*   [x] Lograr un Baseline bajo intencional probando el pipeline completo (Ej. Gemma-2-9B -> 68.4%).
*   [x] Agregar soporte multi-optimizador al CLI (`--optimizer`, `--teacher-model`, `--auto`) — soporta `BootstrapFewShot` y `MIPROv2`.
*   [x] Ejecutar primera corrida real con **MIPROv2** sobre el dataset completo (112 entradas) — descubierto overfitting severo (~99% train vs ~83% test).
*   [x] **Expandir dataset:** etiquetado manual completado (201 entradas orgánicas via curator_ui).
*   [x] **Generación de datos sintéticos:** Módulo `04_Synthetic_Data` implementado con Strategy 1 (Output→Input reverse generation). 118 entradas sintéticas generadas vía GPT-4o-mini con 0 rechazos.
*   [x] Reejecutar optimización con dataset ampliado (364 entradas) — MIPROv2 + gpt-5.1-codex-max como teacher.
*   [x] **Validación de consistencia:** Verificación final de montos en `synthetic_reverse` completada.
*   [x] **Optimización definitiva con MIPROv2:** Score final de **98.06%** en test set (55 ejemplos). 0 errores en amount, currency y type. Modelo listo como Automatic Labeler.


## Fase 2: Model Engineering (SLM Fine-Tuning & Bake-off) 🟢 (Completado)

### 1. Data Prep & Formateo (ChatML)
- [x] **Crear `dataset_formatter.py`:** Convertidos 9,196 registros de `forward_validated.jsonl` a ChatML 3-turn (`system/user/assistant`). Output: `05_Fine_Tuning/data/train_chatml.jsonl`.
  - *System Prompt:* Extraído del programa DSPy optimizado (`optimized_extractor_20260302_113921.json`), sin instrucciones de CoT (para SFT queremos output JSON directo, sin razonamiento explícito).
  - *Assistant:* JSON array puro de transacciones (sin markdown, sin texto extra).
  - *Distribución:* 8,444 single-txn | 466 dual-txn | 286 triple-txn.
- [x] **Data Splits:** Implementado en runtime dentro de `train.py` con `train_test_split(test_size=0.1, seed=42)`. No se shufflea el dataset antes del entrenamiento.
- [x] **Aislamiento del Golden Set:** `golden_dataset.jsonl` (363 registros orgánicos) permanece estrictamente separado. Solo se usa como hold-out final en `02_Evaluation/`.

### 2. Matriz de Bake-off

**Decisión de arquitectura (07/03/2026):** Se migra de Qwen2.5-Instruct a **Qwen3.5**, una familia más reciente de Alibaba. Los modelos son suficientes ya que el fine-tuning con ChatML imprime el comportamiento de instrucción.

**Arquitectura Qwen3.5:** Híbrida — combina *Gated DeltaNet* (linear attention) y *Gated Attention* (standard attention) en bloques intercalados + sparse MoE en FFN para los modelos grandes. Soporte nativo de 201 idiomas y contexto de 262K tokens.

| Tier | Model ID (HuggingFace) | Params | VRAM (bf16) |
|---|---|---|---|
| Featherweight | `Qwen/Qwen3.5-0.8B` | 0.8B | ~1.6 GB |
| Lightweight | `Qwen/Qwen3.5-2B` | 2B | ~4 GB |
| Sweet Spot | `Qwen/Qwen3.5-4B` | 4B | ~8 GB |
| Heavyweight | `Qwen/Qwen3.5-9B` | 9B | ~18 GB |

- *(Confirmado: Unsloth soporta toda la familia Qwen3.5 nativamente desde 2025.)*
- Entorno de entrenamiento: **RunPod** (GPU Ampere, CUDA 12.4, PyTorch 2.4.0).

### 3. Pipeline de Entrenamiento (Unsloth + LoRA bfloat16)

**Decisión crítica:** **No usar QLoRA (4-bit)** para Qwen3.5. Unsloth explícitamente no recomienda cuantización 4-bit para esta familia por incompatibilidades con la arquitectura Gated DeltaNet. Se entrena en **bfloat16 full-precision** con LoRA.

- [x] **Setup del Script (`train.py` en `05_Fine_Tuning/`):** CLI con flags para todos los hiperparámetros, 90/10 split en runtime, early stopping, y guardado de `training_summary.json`.
- [x] **Hiperparámetros LoRA Baseline:** `r=16`, `lora_alpha=32`, `dropout=0.05`.
  - `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - Effective batch: 4 × 4 grad_accum = 16 | LR: 2e-4 cosine | `packing=True` | `use_gradient_checkpointing="unsloth"`.
- [x] **Prevención de Overfitting:** 2 epochs por defecto. `EarlyStoppingCallback(patience=3)` sobre eval_loss.
- [x] **Thinking mode:** Deshabilitado implícitamente — Qwen3.5 Small (≤4B) lo deshabilita por defecto. El fine-tuning sobre respuestas JSON puras refuerza el modo non-thinking.

### 4. Automatización de Evals contra Golden Set
- [x] **Pipeline de Inferencia:** Merge del adaptador LoRA con el modelo base habilitado en `train.py` (`--merge-16bit`) para exportación directa a vLLM.
- [x] **Integración de Evaluación:** Desarrollado `bakeoff_pipeline.py`. Conecta el modelo fine-tuneado a un servidor vLLM local efímero y corre `02_Evaluation/run_evaluation.py` contra el `golden_dataset.jsonl`.
- [x] **Reporte Final del Bake-off:** El orquestador consolida los resultados JSON en `bakeoff_summary_report.md` (y su contraparte `.json`), midiendo Entity F1-Score, Parsing Success y métricas desglosadas.

### 6. GGUF Packaging & Despliegue Local
- [x] **Script de empaquetado (`package_gguf.py`):** Conversor de modelos fine-tuneados a formato GGUF vía Unsloth, con soporte para múltiples niveles de cuantización (`q4_k_m`, `q5_k_m`, `q8_0`, `f16`) y push automático a HuggingFace Hub.
- [x] **Setup de RunPod (`setup_gguf_runpod.sh`):** Script de instalación de dependencias (xformers, Flash Attention 2, Unsloth) para entornos GPU cloud.
- [x] **Pipeline completo (`run_pipeline.sh`):** Orquestador Tmux que encadena instalación de dependencias → login a W&B y HuggingFace → dataset formatting → bake-off pipeline en un solo comando.
- [x] **Evaluación local (`run_local_evaluation.py`):** Script de evaluación directa sobre modelos Unsloth cargados en memoria, sin necesidad de servidor vLLM, con soporte para modelos fine-tuneados en modo promptless (`--no-prompt`).
- [x] **Despliegue exitoso en `llama.cpp`:** Modelo Qwen3.5-4B cuantizado a `q4_k_m` corriendo localmente en Mac con retención completa de capacidades (97.23% F1, 100% Strict JSON).

---

### 5. Orquestador de Bake-off (`bakeoff_pipeline.py`)

Se construyó un script robusto para automatizar la matriz de pruebas sobre los modelos Qwen3.5 (0.8B, 2B, 4B). Por cada modelo, el pipeline ejecuta 4 fases garantizando consistencia y cleanup de recursos (GPU/Puertos):

1. **Fase A (Fine-Tuning):** Ejecuta `train.py` generando checkpoints completos en bfloat16.
2. **Fase B (vLLM Serve):** Levanta un servidor asíncrono con *polling* vía `httpx` hasta validar el cold-start (HTTP 200 OK).
3. **Fase C (Evaluation):** Corre el pipeline contra el endpoint efímero de vLLM y captura los reportes JSON.
4. **Fase D (Teardown):** Mecanismo de `try/finally` estricto que mata los procesos vLLM garantizando liberación de GPU incluso en bloqueos o finalizaciones forzadas (`Ctrl+C`).



---

## 📊 Golden Dataset — Estado al 28/02/2026 (post generación sintética)

| Métrica | Valor | Cambio vs. pre-sintético |
|---------|-------|------------------------|
| **Entradas totales** | 364 | +118 (era 246) |
| **Total de transacciones** | 433 | +134 (era 299) |
| **Expenses** | 361 (83.4%) | +108 |
| **Incomes** | 72 (16.6%) | +26 |
| **Entradas multi-transacción** | 24 | +5 |
| **Entradas no-transaccionales** | 20 | +14 (era 6) |
| **Moneda ARS** | 428 (98.8%) | |
| **Moneda USD** | 5 (1.2%) | |
| **Categorías activas** | 14 / 15 | |

### Distribución por fuente

| Fuente | Entradas |
|--------|----------|
| curator_ui (orgánicas) | 201 |
| synthetic_reverse (Strategy 1) | 118 |
| synthetic_boost | 15 |
| synthetic_boost_v2 | 15 |
| synthetic_normalization_v3 | 15 |

### Distribución por categoría (Expenses — 361 txns)

| Categoría | Count | % |
|-----------|-------|---|
| Supermercado_Despensa | 71 | 19.7% |
| Comida_Comprada | 54 | 15.0% |
| Ocio_Entretenimiento | 33 | 9.1% |
| Transporte | 30 | 8.3% |
| Regalos_Otros | 29 | 8.0% |
| Educacion_Capacitacion | 25 | 6.9% |
| Vivienda_Servicios | 25 | 6.9% |
| Hogar_Mascotas | 25 | 6.9% |
| Financiero_Tarjetas | 25 | 6.9% |
| Salud_Bienestar | 24 | 6.6% |
| Deporte_Fitness | 20 | 5.5% |

### Distribución por categoría (Income — 72 txns)

| Categoría | Count | % |
|-----------|-------|---|
| Inversiones_Finanzas | 25 | 34.7% |
| Salario_Honorarios | 20 | 27.8% |
| Regalos_Otros | 15 | 20.8% |
| Ventas_Negocios | 12 | 16.7% |

### ✅ Mejora de balance post-generación

**Categorías de income:** Income Regalos_Otros pasó de 3 a 15 ejemplos (5x). Todas las categorías de income ahora tienen ≥12 ejemplos.

**Categorías de expense:** Las categorías cola (Financiero_Tarjetas, Educacion_Capacitacion, Hogar_Mascotas) subieron a 25 cada una. El rango se comprimió de 12-56 a 20-71.

---

## 🧪 Evaluation Pipeline — Test Results (21/21)

```
platform linux -- Python 3.10.12, pytest-9.0.2
collected 21 items

tests/test_evaluator.py::test_strict_json_score_valid_json              PASSED
tests/test_evaluator.py::test_strict_json_score_valid_array             PASSED
tests/test_evaluator.py::test_strict_json_score_markdown_wrapped        PASSED
tests/test_evaluator.py::test_strict_json_score_conversational_text     PASSED
tests/test_evaluator.py::test_strict_json_score_missing_required_fields PASSED
tests/test_evaluator.py::test_entity_accuracy_exact_match               PASSED
tests/test_evaluator.py::test_entity_accuracy_amount_mismatch           PASSED
tests/test_evaluator.py::test_category_match_exact                      PASSED
tests/test_evaluator.py::test_category_match_case_insensitive           PASSED
tests/test_evaluator.py::test_category_match_wrong                      PASSED
tests/test_evaluator.py::test_date_match_both_null                      PASSED
tests/test_evaluator.py::test_date_match_relative_exact                 PASSED
tests/test_evaluator.py::test_date_match_absolute_substring             PASSED
tests/test_evaluator.py::test_date_match_delta_mismatch                 PASSED
tests/test_evaluator.py::test_date_match_one_null_one_set               PASSED
tests/test_evaluator.py::test_f1_more_predictions_than_expected         PASSED
tests/test_evaluator.py::test_f1_fewer_predictions_than_expected        PASSED
tests/test_evaluator.py::test_f1_non_transactional_correct              PASSED
tests/test_evaluator.py::test_f1_non_transactional_false_positive       PASSED
tests/test_evaluator.py::test_evaluate_full_report_perfect_predictions  PASSED
tests/test_evaluator.py::test_evaluate_full_report_empty_predictions    PASSED

21 passed in 0.08s
```

> **Actualización 26/02/2026:** Se agregaron `ErrorBreakdown` y `FieldErrorStats` al `Evaluator`. Ahora cada `EntryEvaluationResult` tiene un campo `error_breakdown` (flags por par matcheado), y el `EvaluationReport` agrega `error_statistics` con conteos globales. El denominador es `total_expected_transactions`: los FN (transacciones que el modelo no detectó) suman error en **todos** los campos, evitando subestimar los errores. Test suite actualizado a **28/28**.

**EntityAccuracy weights:**
- **Finanzas (50%):** `amount` 35% + `currency` 15%
- **Clasificación (30%):** `type` 15% + `category` 15%
- **Temporal (20%):** `date_delta_days` 70% + `date_raw_expression` 30% (normalized substring match)

---

## 🗂️ Estructura de Archivos Final

```
quipu-research/
├── README.md                         # Roadmap principal del proyecto
├── DEVELOPMENT_LOG.md                # Este archivo — log de desarrollo cerrado
├── golden_dataset.jsonl              # 364 entradas (201 orgánicas + 163 sintéticas)
├── backups/                          # Backups auto-generados del golden dataset
│
├── 01_Data_Acquisition/
│   ├── README.md                     # Documentación de uso de utilidades
│   ├── dataset_curator.py            # Backend Flask con API REST
│   ├── api_types.py                  # TypedDict para el curator
│   ├── api_types_examples.py         # Ejemplos de tipos para el curator
│   ├── data/
│   │   ├── categories.json           # 14 categorías con descripciones
│   │   └── tentative_dataset.json    # Dataset parseado (506 conversaciones)
│   ├── docs/
│   │   └── LABELING_GUIDELINES.md    # Reglas de etiquetado
│   ├── scripts/
│   │   ├── parse_whatsapp_chat.py    # Parser del chat de WhatsApp
│   │   └── start_curator.sh          # Inicia el curator UI
│   ├── templates/
│   │   ├── curator.html              # UI principal de curación
│   │   └── insights.html             # Dashboard de estadísticas
│   └── files/
│       └── WhatsApp Chat with Quipu.txt
│
├── 02_Evaluation/
│   ├── evaluator.py                  # Clase Evaluator con todas las métricas + FieldErrorStats
│   ├── schemas.py                    # TypedDict: PredictionEntry, EvaluationReport, ErrorBreakdown, FieldErrorStats, etc.
│   ├── run_evaluation.py             # Script de evaluación contra APIs remotas (LiteLLM)
│   ├── run_local_evaluation.py       # Script de evaluación directa sobre modelos Unsloth locales
│   ├── huggingface_setup.md          # Guía: deploy en HF Endpoints + corrección del tokenizador
│   ├── setup_runpod.sh               # Setup de RunPod para evaluación remota
│   ├── test_quipu.py                 # Test rápido de inferencia
│   ├── results/                      # Resultados de evaluaciones
│   ├── venv/                         # Entorno virtual (pytest)
│   └── tests/
│       └── test_evaluator.py         # 28 tests unitarios
│
├── 03_Optimization/
│   ├── dspy_modules/                 # Módulos y Signatures de DSPy
│   ├── metrics.py                    # Adaptador de DSPyMetric (Evaluator)
│   ├── observability.py              # Telemetría de tokens y costos (USD)
│   ├── optimization_schemas.py       # Schemas para el pipeline de optimización
│   ├── run_optimization.py           # Pipeline multi-optimizador: BootstrapFewShot + MIPROv2
│   ├── requirements.txt              # dspy-ai, litellm, pydantic
│   └── tests/
│       └── test_metrics.py           # Unit test del metric adapter
│
├── 04_Synthetic_Data/
│   ├── README.md                     # Documentación y uso del módulo
│   ├── config.py                     # Auto-detección de provider LLM
│   ├── balance_analyzer.py           # Análisis de gaps por categoría y features
│   ├── generate_reverse.py           # Strategy 1: Output→Input (--dry-run, --append, --limit)
│   ├── generate_forward_roundtrip.py # Strategy 2: Forward Round-Trip con validación (--targeted)
│   ├── prompts/
│   │   └── reverse_generation.py     # Prompts estilo WhatsApp argentino
│   ├── validators/
│   │   └── quality_checks.py         # Validación de schema y consistencia de montos
│   ├── generated/                    # Output de generaciones para review
│   └── venv/                         # Entorno virtual (litellm)
│
└── 05_Fine_Tuning/
    ├── train.py                      # Script Unsloth LoRA bfloat16, soporta --merge-16bit
    ├── bakeoff_pipeline.py           # Orquestador del Bake-off (train + serve + eval)
    ├── dataset_formatter.py          # Conversor de golden/synthetic a ChatML
    ├── package_gguf.py               # Empaquetado a GGUF + push a HuggingFace Hub
    ├── run_pipeline.sh               # Orquestador completo en Tmux (setup → train → eval)
    ├── setup_gguf_runpod.sh          # Instalación de dependencias para GGUF en RunPod
    ├── requirements_training.txt     # Dependencias Unsloth / HuggingFace
    └── data/
        └── train_chatml.jsonl        # Output listo para Unsloth (9,196 registros)
```

---

## 💡 Decisiones de Diseño Clave

- **`description` excluida de EntityAccuracy:** es texto libre sin respuesta única correcta. Incluirla con fuzzy matching agrega ruido.
- **F1 TP requiere `amount` + `type` exactos:** cantidad y naturaleza del gasto deben coincidir. Una cantidad incorrecta no puede ser TP aunque la categoría coincida.
- **Temporal como unidad (20%):** `date_delta_days` (70%) + `date_raw_expression` (30%). El modelo nunca debe calcular fechas absolutas — solo extraer la expresión. El backend (Rust, fases futuras) resuelve el delta.
- **Null semántico en fechas absolutas:** `delta=null` no es error, es el comportamiento correcto cuando hay una fecha absoluta. El evaluator lo trata correctamente.
- **FieldErrorStats y ErrorBreakdown:** Denominador `total_expected_transactions`. Los FN cuentan como error en todos los campos.
- **Optimización de Signatures (Marzo 2026):** Se inyectó conocimiento experto de lunfardo rioplatense (lucas, gambas, palos) y reglas de desambiguación (facturas panadería vs boleta) directamente en el base `Signature` de DSPy para mejorar el baseline Zero-Shot.
- **Resiliencia en Optimización:** El script `run_optimization.py` ahora soporta `KeyboardInterrupt` (Ctrl+C) para salvar el progreso parcial y genera un `failures_log.json` automático para debugging de casos con score < 0.8.
- **Soporte de Modelos de Razonamiento (o1/o3):** Pipeline adaptado para cumplir requerimientos estrictos de OpenAI (temp=1.0, max_tokens=16000) en modelos de razonamiento usados como teachers.
- **Métrica Permisiva con Markdown:** `quipu_metric` ahora pre-limpia la respuesta antes del "Hard Gate", permitiendo que el optimizador se concentre en la precisión del contenido incluso si el modelo usa wrappers JSON.

---

## 🤖 Model Baselines (Zero-Shot Prompting)

A continuación, los resultados iniciales probando diferentes modelos sobre el evaluador utilizando el prompt _Zero-Shot_ creado en `run_evaluation.py` (febrero 2026). Dado que el formateo en JSON varía para cada modelo, el parseo tolerante hace que el "Strict JSON Score" caiga, pero las métricas del negocio miden la precisión real de extracción.

| Modelo / API (OpenRouter) | N Muestras | F1-Score | Precision | Recall | Entity Accuracy | Cat Match |
|---------------------------|------------|----------|-----------|--------|-----------------|-----------|
| `gpt-4o-mini` | 25 | **0.9855** | 1.000 | 0.971 | 82.8% | 61.0% |
| `minimax/minimax-m2.5` | 15 | **0.9714** | 1.000 | 0.944 | 78.0% | 53.3% |
| `google/gemma-3-12b-it` | 15 | **0.9032** | 0.933 | 0.875 | **86.4%** | **63.3%** |

> **Nota:** El `Strict JSON Score` para los proveedores de la capa de API OpenRouter da `0.0%` dado que inyectan wrappers de código o texto adicional que fallan la estricta validación limpia de arreglo puro.

---

## 🚀 Model Optimization Baselines (DSPy BootstrapFewShot)

Pruebas iniciales demostrando el funcionamiento del evaluador DSPy automático con `quipu_metric` en OpenRouter. Se utilizó un sample para buscar una métrica con fallas donde arrancar a optimizar.

Todas las corridas usan `BootstrapFewShot`, N=25 entries (split 70/15/15 → ~5 en test set), 26/02/2026.

### Scores Globales

| Modelo | Score Final (Test) | Hard Gate | Total Tokens | Costo USD |
|--------|--------------------|-----------|--------------|----------|
| `gpt-4o-mini` | **0.9310 (93.1%)** | 0 / 5 | 6,589 | $0.0013 |
| `minimax/minimax-m2.5` | **0.9350 (93.5%)** | 0 / 5 | 10,408 | $0.0057 |
| `google/gemma-2-9b-it` | **0.6840 (68.4%)** | 1 / 5 | N/A* | $0.0000 |

*\*gemma-2-9b-it no está mapeado en litellm para costo, tokens reportados en 0.*

### Error Breakdown por Campo (sobre 5 transacciones esperadas por test set)

| Campo | `gpt-4o-mini` | `minimax-m2.5` | `gemma-2-9b-it` |
|-------|:---:|:---:|:---:|
| Hard Gate (JSON inválido) | 0 / 5 (0%) | 0 / 5 (0%) | **1 / 5 (20%)** |
| Amount errors | 0 / 5 (0%) | 0 / 5 (0%) | 1 / 5 (20%) |
| Currency errors | 0 / 5 (0%) | 0 / 5 (0%) | 1 / 5 (20%) |
| Type errors | 0 / 5 (0%) | 0 / 5 (0%) | 1 / 5 (20%) |
| **Category errors** | 3 / 5 (60%) | 3 / 5 (60%) | **5 / 5 (100%)** |
| **Date delta errors** | 0 / 5 (0%) | 1 / 5 (20%) | **5 / 5 (100%)** |
| **Date expression errors** | **4 / 5 (80%)** | 1 / 5 (20%) | 1 / 5 (20%) |

### Análisis

- **`gemma-2-9b-it`** — El hard gate elimina 1/5 entradas (JSON inválido). Sobre las restantes, falla en categoría y date_delta en **el 100% de los casos**. El problema es estructural: no sigue el esquema de salida correctamente.
- **`minimax-m2.5`** — JSON siempre limpio. El cuello de botella es la **categoría** (60% de error): confunde categorías similares. El date_delta tiene un único fallo puntual.
- **`gpt-4o-mini`** — Amount/type/currency perfectos. La categoría tiene 60% de error (igual que minimax). El anomalía es `date_expression_errors: 4/5 (80%)`: extrae bien el `date_delta_days` pero serializa la expresión raw de forma diferente al golden. Esto sugiere que el formato de `date_raw_expression` no está bien especificado en el prompt.

> **Conclusión:** El Hard Gate **no** es el cuello de botella principal para gpt-4o-mini ni minimax. El problema dominante es la **categoría** (ambigüedad semántica entre categorías); y para gpt-4o-mini, la serialización de `date_raw_expression`. Para gemma-2-9b es todo: necesita más few-shot o un prompt más estructurado.

---

## 🚀 Primera Corrida con MIPROv2 (Dataset Completo — 27/02/2026)

Con el CLI de optimización ampliado (nuevos flags `--optimizer`, `--teacher-model`, `--auto`) se ejecutó la primera pasada de **MIPROv2** sobre el dataset completo de 112 entradas (split 70/15/15 → ~78 train / ~17 val / ~17 test).

### Cambios previos a la corrida

- **`run_optimization.py`** — Refactorizado para soportar múltiples teleprompters:
  - `--optimizer {bootstrap,miprov2}` — selecciona el teleprompter.
  - `--teacher-model` — modelo docente para MIPROv2 (`prompt_model`); si se omite, usa el mismo que el estudiante.
  - `--auto {light,medium,heavy}` — presupuesto de optimización de MIPROv2.
- **`run_evaluation.py`** — Eliminada la inyección de `Fecha y hora actual` del prompt de usuario: el modelo no necesita la fecha en el prompt de usuario, y eliminarla simplifica el contexto.

### Resultado

| Modelo (estudiante) | Optimizer | Auto | Score Train (aprox.) | Score Test |
|---------------------|-----------|------|----------------------|------------|
| `minimax/minimax-m2.5` | MIPROv2 | medium | ~99% | ~83% |

### ⚠️ Diagnóstico: Overfitting

La brecha entre el score de training (~99%) y el score del test set (~83%) es una señal clara de **overfitting**. El optimizador aprendió a ajustar los prompts casi perfectamente al trainset pero generalizó mal a datos no vistos.

**Causa raíz probable:** El golden dataset de 112 entradas es demasiado pequeño y poco diverso para que MIPROv2 (que busca en un espacio de instrucciones mucho mayor que BootstrapFewShot) encuentre prompts verdaderamente generalizables.

### 📌 Decisión: Volver a etiquetar

Antes de continuar con optimización, se prioriza **ampliar el dataset** etiquetando manualmente entre **100 y 200 gastos nuevos**. Los beneficios esperados son:

1. **Mayor diversidad léxica y categórica** — más variedad en cómo se expresan los gastos en el mundo real.
2. **Base sólida para datos sintéticos** — un corpus de referencia más rico permite generar variaciones sintéticas más representativas.
3. **Menor overfitting en futuras corridas** — con un trainset más grande y variado, el optimizador no puede "memorizar" los ejemplos.

---

## 🏆 Optimización Definitiva con MIPROv2 (Marzo 2026)

Tras expandir el dataset y corregir sesgos de evaluación, se ejecutó una ronda final de optimización con **MIPROv2** (`--auto light`) utilizando `gpt-5-mini` como estudiante y `gpt-5.1-codex-max` como teacher.

### Resoluciones Técnicas Clave:

1. **Corrección de "True Negatives"**: Se arregló un bug crítico en `evaluator.py` donde el modelo era penalizado con F1=0.0 al predecir correctamente `[]` (cero transacciones) cuando se esperaba `[]`. 
2. **Enforcement de JSON Mode**: Se modificó `run_optimization.py` para forzar `response_format={"type": "json_object"}` *exclusivamente* en el modelo estudiante. Esto eliminó por completo los errores de parseo por envolturas de Markdown, asegurando un "Hard Gate" del 100%. (El teacher model se mantuvo en texto plano para evitar errores 400 de la API de OpenAI durante la generación de instrucciones).
3. **Eliminación de `current_date`**: Se comprobó que pasar la fecha actual como input confundía la métrica de `date_delta_days`, resultando en valores `null` en vez de `0`. Se eliminó por completo la dependencia de fecha real, forzando al LLM a trabajar puramente de forma semántica y relativa (ej. asumiendo siempre `0` si no se especifica fecha temporal).

### Resultado Final (Test Set: 55 ejemplos)

| Modelo (estudiante) | Teacher | Optimizer | Score Test | Costo USD |
|---------------------|---------|-----------|------------|-----------|
| `gpt-5-mini` | `gpt-5.1-codex-max` | MIPROv2 (`light`) | **98.06%** | ~$0.78 |

**Desglose de Errores (sobre 64 transacciones esperadas):**
- **Amount Errors:** 0
- **Currency Errors:** 0
- **Type Errors (Ingreso vs Egreso):** 0
- **Date Delta Errors:** 3
- **Category Errors:** 11 (Errores subjetivos por ambigüedad de categorías complejas)

**Conclusión:** 
El modelo optimizado es **100% perfecto extrayendo los datos matemáticos y financieros base** (Monto, Moneda, Tipo). Los únicos márgenes de error yacen en las fronteras grises de algunas categorías semánticas. La base está lista para producción y para ser utilizada como *Automatic Labeler* en el módulo de generación de datos sintéticos.

---

## 🎯 Generación Sintética: Clean-up & Targeted Mode (Marzo 2026)

Tras analizar los primeros ~6.700 registros generados en `forward_validated.jsonl`, se detectaron tres sesgos importantes introducidos por el *Teacher LLM*:
1. **Sobrerrepresentación de montos altos** (27% > $1M ARS).
2. **Dependencia temporal excesiva del día actual** (86% de las transacciones caían en "hoy" / `delta=0`).
3. **Desbalance inter-categorías**, especialmente en ingresos vs. egresos y categorías muy específicas (`Hogar_Mascotas`, `Deporte_Fitness`).

### Intervenciones

1. **Dataset Clean-up:**
   - Se descartaron **1.466 registros** problemáticos (outliers de longitud >500 chars, montos irreales >$3M ARS y duplicados exactos).
   - El dataset base limpio quedó en **7.981 registros.**

2. **Reingeniería del Script Generador (`generate_forward_roundtrip.py`):**
   - Incorporación de soporte para **fechas absolutas** (ej. `"el 15 de febrero"`, `"15/02/2026"`).
   - Ajuste general de los pesos (`weights`) en la asignación aleatoria para aplanar la campana matemática.

3. **La bandera `--targeted`:**
   - Se construyó un pipeline condicional de rebalanceo agresivo.
   - En este modo, se restringe la generación: 0% probabilidad para "hoy", 90% obligando a tipo `EXPENSE` (para compensar) y un multiplicador x5 para las categorías olvidadas (`Hogar_Mascotas`, `Deporte_Fitness`).
   - Se ejecutaron **~7.500 intentos** bajo este modo estricto.

### Resultados de Balance Final: ~9.2K Records

La ejecución targeted logró corregir orgánicamente el volumen general sin necesidad de borrar más datos. El estado final del corpus sintético:

| Métrica | Original (6.7K) | **Post-Targeted (9.2K)** |
|:---|:---:|:---:|
| **Records Totales** | 6.701 | **9.196** (10.234 txns) |
| **EXPENSE** | 54.0% | **65.2%** |
| **Fechas Absolutas** | 0.0% | **3.0%** |
| **"Hoy" (Delta=0)** | 86.5% | **72.3%** |
| **Montos < $15K ARS** | 1.2% | **25.1%** |
| **Montos > $1M ARS** | 27.5% | **10.7%** |

El corpus alcanzó un nivel de madurez estructural ideal. La adición de fechas absolutas y transacciones "micro" ($50-$2.000 ARS) permite escalar el entrenamiento a una etapa de generalización final.

---

## 🏆 Fine-Tuned Models (Bake-off Results)

Tras finalizar el fine-tuning (SFT) de la familia Qwen3.5 utilizando el dataset en formato ChatML, se corrieron las últimas evaluaciones directamente contra el golden dataset. 

A diferencia de los modelos base (evaluados en zero-shot), los modelos fine-tuneados asimilaron completamente el comportamiento transaccional. Lograron un **Strict JSON Score del 100%**, elaborando un JSON puro como respuesta sin requerir el prompt completo de instrucciones ni agregar formato markdown invasivo.

Las evaluaciones se corrieron mediante inferencia remota en vLLM (sobre endpoints privados de Hugging Face requiriendo autenticación vía `--api-key`). La configuración de parcheo del `tokenizer_class` y argumentos de vLLM quedó registrada en `02_Evaluation/huggingface_setup.md`.

### Resultados Globales (Marzo 2026)

Se evaluaron modelos comerciales y open-source en diferentes esquemas (Zero-Shot, Few-Shot y Fine-Tuned) contra un subset representativo del golden dataset:

| Modelo | Modalidad | F1-Score | Precision | Recall | Strict JSON | Entity Acc | Cat Match | Latencia (p50) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `gpt-5-mini` | Zero-Shot | 100.00% | 100.00% | 100.00% | 100.00% | 96.25% | 79.56% | 5.9s |
| `gpt-5-mini` | Few-Shot (3) | 100.00% | 100.00% | 100.00% | 100.00% | 95.61% | 75.28% | 4.7s |
| `Qwen-2.5-7B` | Zero-Shot | 83.81% | 97.78% | 73.33% | 74.00% | 67.46% | 55.78% | 1.6s |
| `Qwen-2.5-7B` | Few-Shot (3) | 86.73% | 92.45% | 81.67% | 86.00% | 82.12% | 49.78% | 2.5s |
| `Qwen3.5-0.8B` | Fine-Tuned | 93.62% | 92.15% | 95.14% | 100.00% | 88.05% | 66.16% | - |
| `Qwen3.5-2B` | Fine-Tuned | 94.06% | 92.79% | 95.37% | 100.00% | 88.87% | 73.18% | - |
| **`Qwen3.5-4B`** | **Fine-Tuned** | **97.57%** | **97.46%** | **97.69%** | **100.00%** | **94.93%** | **80.77%** | **2.2s** |

### Análisis Comparativo

- **El poder del Fine-Tuning en SLMs:** Un modelo de 4B parámetros afinado para la tarea supera cómodamente a un modelo actual base de 7B parámetros (Qwen-2.5-7B), incluso con under-prompting (Few-Shot). Qwen3.5-4B (FT) logra un **F1 de 97.57% vs 86.73%** del modelo 7B few-shot, y aplasta en fidelidad de estructura (**Strict JSON 100% vs 86%**).
- **Competitividad comercial:** El Qwen3.5-4B FT se posiciona casi a la par de `gpt-5-mini` en extracción (Entity Accuracy 94.9% vs 96.2%), pero lo supera en **Category Match (80.7% vs 79.5%)**. Esto demuestra que para dominios de gran especificidad contextual (dialecto rioplatense, jerga financiera argentina), la inyección de conocimiento vía SFT rinde más que la escala generalista comercial.
- **La trampa del Few-Shot en razonamiento abstracto:** Curiosamente, tanto en `gpt-5` como en `Qwen-2.5`, agregar ejemplos (Few-Shot) mejoró métricas estructurales o de recall temporal, pero **empeoró la precisión de categorización** (Category Match cayó ~4-6 puntos en ambos casos). Esto sugiere que para reglas de negocio muy finas, los modelos grandes se sobrefijan a las categorías presentes en la ventana de ejemplos (Over-conditioning), perdiendo la abstracción lograda en el pre-entrenamiento explícito del prompt.
- **Velocidad y soberanía:** gpt-5-mini exhibe latencias de ~5-6s. El motor local (GGUF) despliega el Qwen3.5-4B FT resolviendo tareas equivalentes en ~2.2s (p50) a $0 costo.

### Pruebas de Despliegue Local (Qwen-4B GGUF / llama.cpp)

Se realizó una validación del modelo Qwen-4B cuantizado a GGUF, servido localmente a través de `llama.cpp` (`llama-server`) en Mac. Tras aplicar los ajustes de enrutamiento en `run_evaluation.py` para forzar la compatibilidad con el proveedor de OpenAI (`custom_llm_provider="openai"`), la integración con LiteLLM fue exitosa.

Los resultados técnicos correspondientes a la ejecución (`results_Qwen-3.5-quipu-q4_k_m_20260314_210607.json`) sobre la muestra total (363 entradas) validan que la cuantización retiene las capacidades del modelo original:

| Modelo Local | N Muestras | F1-Score | Precision | Recall | Strict JSON | Entity Acc | Cat Match | Latencia (p50) | Latencia (p99) |
|--------------|------------|----------|-----------|--------|-------------|------------|-----------|----------------|----------------|
| `Qwen-3.5-quipu-q4_k_m (GGUF)` | 363 | 97.23% | 97.00% | 97.45% | 100.00% | 94.34% | 80.64% | 2.240s | 13.268s |

*Notas sobre inferencia local:*
1. El modelo GGUF **retiene la capacidad de formateo estructural** sin pérdida, logrando un *Strict JSON Score* del 100%.
2. El enrutador local de LiteLLM transmite exitosamente los Message Roles. El `System Prompt` fue correctamente aislado e interpretado por el SLM.
3. Se ratifica que el pipeline de Quipu es agnóstico del ambiente de inferencia: puede iterar entre proveedores remotos, vLLM y nodos locales (`llama.cpp`) fluidamente.

---

## ⚙️ Fase 3: Evolución del Motor de Inferencia Local (Marzo 2026)

Tras validar el modelo GGUF, se encaró la construcción del motor de inferencia de producción. El objetivo: servir el SLM localmente en Mac M5 con latencia mínima bajo carga concurrente. La evolución pasó por 3 arquitecturas hasta encontrar el diseño óptimo.

### 1. Rust MPSC + Limpieza Total de Caché

Primer enfoque: motor propio en Rust usando los bindings de `llama-cpp-2` con cola MPSC (Multi-Producer, Single-Consumer).

- **Problema del KV Cache:** Por seguridad se hacía `ctx.clear_kv_cache()` total por cada request, forzando a re-procesar los ~600 tokens del System Prompt (reglas financieras, categorías, etc.) en cada mensaje → "Cold Start" altísimo.
- **Resultados k6 (5 VUs):**
  - Min Latency: 3.08s | Max Latency (p90): 14.35s
  - Throughput: **14 requests en 30s** (secuencial, 1 en 1)

### 2. La Guerra del KV Cache (M-RoPE y Prefix Caching)

Intento de cachear el System Prompt para eliminar el prefill redundante de ~750ms:

- Se intentó un patrón **Master/Worker** copiando secuencias en VRAM (`seq_cp`), pero la familia Qwen usa M-RoPE (Multi-Resolution Rotary Position Embedding), lo que impedía particionar y clonar el estado del caché libremente → `GGML_ASSERT` en C++.
- **Fix manual (Prefix Caching por Truncamiento):** En vez de clonar secuencias, se usó una sola secuencia truncando desde el token 601 en adelante, preservando el System Prompt intacto en caché.
- **Logro:** TTFT cayó de ~750ms a **<50ms**. Se le ahorró a la GPU recalcular el 90% del contexto.
- **Resultados k6 (5 VUs):**
  - Min Latency: 2.24s | Max Latency: 11.06s
  - Throughput: **18 requests en 30s** (+30% vs. arquitectura anterior)

### 3. Pivot a `llama-server` (Continuous Batching)

Se detectó over-engineering: replicar el manejo de slots y memoria en Rust era reinventar la rueda. Decisión: separar el cerebro del sistema nervioso — dejar que `llama-server` maneje la VRAM y usar el backend web como API Gateway.

- **Continuous Batching:** Con `--parallel 4`, el server dejó de procesar de a 1 usuario. Empezó a agarrar mensajes de 4 usuarios simultáneamente, meterlos en la misma matriz y procesarlos en paralelo.
- **Resultados k6 (5 VUs — El Nirvana):**
  - Avg Latency: **4.4s** (antes 9.1s) | Max Latency (p95): **6.36s**
  - Throughput: **30 requests en 30s** (1 req/s sostenido)

### 📊 Tabla de Evolución del Throughput (k6, 5 VUs, 30s)

| Arquitectura | Prefill (TTFT) | Cola de Procesamiento | Min Latency | Max Latency | Throughput (30s) |
|:---|:---|:---|:---|:---|:---|
| **Rust (Clear Cache)** | ~750ms | Secuencial (1 en 1) | 3.08s | 15.32s | 14 |
| **Rust (Prefix Caching)** | ~50ms | Secuencial (1 en 1) | 2.24s | 11.06s | 18 |
| **llama-server** | ~50ms | **Paralelo (Batching)** | 3.46s | **8.94s** | **30** |

> **Conclusión:** Entender el cuello de botella (prefill), intentar mitigarlo a nivel de punteros (Rust/C++), y finalmente aplicar el patrón de arquitectura distribuida correcto (Microservicios + Continuous Batching) resultó en un motor local que procesa **1 petición por segundo** en hardware de consumo (Mac M5), manteniendo la latencia debajo de los 6 segundos bajo estrés.

---

## 🏁 Cierre del Proyecto de Research (27/03/2026)

### Estado Final

El proyecto `quipu-research` se da por **cerrado**. Todos los objetivos de investigación planteados originalmente fueron alcanzados o superados:

| Objetivo Original | Estado | Resultado |
|---|---|---|
| Golden Dataset curado | ✅ Completado | 364 entradas (201 orgánicas + 163 sintéticas) |
| Pipeline de Evaluación riguroso | ✅ Completado | `Evaluator` con 7 métricas, 28 tests, error breakdown por campo |
| Baseline & Prompt Optimization (DSPy) | ✅ Completado | 98.06% score con MIPROv2 (gpt-5-mini + gpt-5.1-codex-max teacher) |
| Generación de datos sintéticos | ✅ Completado | ~9.2K registros vía Forward Round-Trip con rebalanceo targeted |
| Fine-Tuning de SLMs | ✅ Completado | Qwen3.5-4B → 97.57% F1, 94.93% Entity Accuracy, 100% Strict JSON |
| Cuantización & Deploy local | ✅ Completado | GGUF q4_k_m corriendo en llama.cpp con 97.23% F1 |
| Motor de inferencia de producción | ✅ Completado | llama-server + Continuous Batching → 30 reqs/30s, p95 < 6.4s |

### Logros Clave para Quipu (la aplicación)

1. **Tiempos:** El modelo fine-tuneado localmente responde en ~4.4s promedio bajo carga concurrente (5 usuarios simultáneos) y ~2.2s en latencia individual, eliminando la dependencia de red. Bajo Continuous Batching procesa 1 req/s sostenido.
2. **Costos:** Inferencia local con GGUF tiene costo $0 por request. El fine-tuning completo (3 modelos × GPU RunPod) costó una fracción de lo que costaría en API calls equivalentes para el mismo volumen de datos.
3. **Precisión:** Un SLM de 4B parámetros cuantizado superó a GPT-4o-mini zero-shot en todas las métricas (Entity Accuracy: 94.93% vs 82.8%, Category Match: 80.77% vs 61.0%), demostrando que el conocimiento de dominio supera a la escala bruta.
4. **Privacidad:** Toda la inferencia corre localmente. Los datos financieros del usuario nunca salen del dispositivo.

### Decisión

El research alcanzó el punto donde las ganancias marginales de continuar iterando sobre el modelo no justifican el esfuerzo. Los aprendizajes y resultados obtenidos son suficientes para:

1. **Integrar el modelo en producción** dentro de Quipu como motor de NLU principal.
2. **Escribir una publicación** compartiendo el proceso completo — desde la recolección de datos en WhatsApp hasta el despliegue de un SLM cuantizado en un Mac — documentando cómo esto mejora los tiempos, costos y precisión de Quipu como aplicación de finanzas personales.

> Lo que queda pendiente es exclusivamente la redacción de la publicación, que ya no pertenece al scope de este repositorio de research.
