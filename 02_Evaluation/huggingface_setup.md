# Guía de Despliegue: Hugging Face Endpoints y Corrección del Tokenizador

## 1. Modificación del archivo JSON (`tokenizer_config.json`)

Debido a un pequeño bug durante la fusión de los pesos con Unsloth, el tokenizador se exporta con una clase genérica que vLLM no reconoce. Es necesario corregirlo manualmente antes de desplegar.

**Pasos a seguir:**
1. Ingresá a tu repositorio del modelo fusionado en Hugging Face.
2. Dirigite a la pestaña **Files and versions**.
3. Buscá y abrí el archivo **`tokenizer_config.json`**.
4. Hacé clic en el ícono del lápiz (**Edit**) arriba a la derecha del código.
5. Buscá la línea que contiene este valor:
   ```json
   "tokenizer_class": "TokenizersBackend"
   ```
6. Reemplazala exactamente por la clase nativa de Qwen:
    ```json
    "tokenizer_class": "Qwen2TokenizerFast"
    ```
7. Scrolleá hasta el final de la página y hacé clic en Commit changes para guardar.

## 2. Configuración del Inference Endpoint (vLLM Custom)

Para evitar problemas de compatibilidad con la arquitectura multimodal de Qwen3.5, es mandatorio forzar la última versión del motor vLLM y decirle explícitamente dónde leer los archivos.

Al momento de crear el Endpoint (o editando uno pausado), dirigite a la sección Advanced Configuration e ingresá estrictamente estos valores:

* Container Type: Custom

* Docker Image URL: vllm/vllm-openai:latest

* Container Port: 8000

* Docker Command / Container Args:

```
/repository --served-model-name default --max-model-len 4096 --trust-remote-code
```

Desglose de los parámetros clave en el Docker Command:

* /repository: Obliga a vLLM a leer los pesos físicos locales que Hugging Face ya descargó en el contenedor (evita que intente descargar un modelo base de internet). Al estar como primer argumento posicional, cumple con la sintaxis estricta de las versiones nuevas de vLLM.

* --served-model-name default: Etiqueta el modelo internamente para que coincida con el "model": "default" que vas a usar en tus peticiones curl o llamadas a la API.

* --trust-remote-code: Permite que vLLM ejecute el código personalizado del tokenizador de Qwen que viene incluido en los archivos del modelo, evitando que el servidor crashee en el arranque por bloqueos de seguridad.

## 3. Autenticación y Evaluación

Dado que tu modelo en Hugging Face es privado, el endpoint requiere autenticación. Para ejecutar la evaluación local (`run_evaluation.py`) contra este endpoint, debes proporcionar tu **Hugging Face Token** a LiteLLM.

Podés hacerlo de dos maneras:

**Opción A: Pasando el parámetro `--api-key` (Recomendado)**
```bash
python run_evaluation.py \
    --model huggingface/default \
    --api-base https://<tu-endpoint-url>.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions \
    --api-key hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
    --no-prompt
```
*Nota: Asegurate de usar `huggingface/default` como nombre del modelo, ya que el endpoint de vLLM lo sirve bajo el nombre `default` (gracias al parámetro `--served-model-name default`).*

**Opción B: Usando variable de entorno**
```bash
export HUGGINGFACE_API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python run_evaluation.py \
    --model huggingface/default \
    --api-base https://<tu-endpoint-url>.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions \
    --no-prompt
```