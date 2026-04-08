import argparse
import logging
import os
import sys
from pathlib import Path

from unsloth import FastLanguageModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Empaqueta un modelo local o de HuggingFace a formato GGUF y lo sube a HuggingFace."
    )
    
    # Origen del modelo
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Ruta local al modelo (ej: outputs/qwen35-0.8b-quipu-merged) o ID de HuggingFace (ej: Qwen/Qwen3.5-0.8B)."
    )
    
    # Destino en HuggingFace
    parser.add_argument(
        "--hub-repo", 
        type=str, 
        required=True,
        help="Repositorio target en HuggingFace donde se subirá el GGUF (ej: tu-usuario/Qwen-Quipu-GGUF)."
    )
    
    # Opciones de cuantización target
    parser.add_argument(
        "--quantization", 
        type=str, 
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16", "f32"],
        help="Método de cuantización GGUF a utilizar. Recomendado para balance: q4_k_m."
    )
    
    # Opciones adicionales
    parser.add_argument(
        "--hf-token", 
        type=str, 
        default=None,
        help="Token de acceso para HuggingFace. Si no se provee, intentará usar la variable de entorno HF_TOKEN."
    )
    parser.add_argument(
        "--save-local-dir", 
        type=str, 
        default=None,
        help="Directorio opcional para guardar una copia local del archivo GGUF antes de subirlo."
    )
    parser.add_argument(
        "--max-seq-len", 
        type=int, 
        default=2048,
        help="Longitud máxima de secuencia soportada por el modelo a cargar."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_source = args.model
    repo_id = args.hub_repo
    quant = args.quantization
    token = args.hf_token or os.environ.get("HF_TOKEN")

    if not token:
        logger.warning(
            "No se ha provisto un token de HuggingFace ni se encontró la variable de entorno HF_TOKEN. "
            "La subida al Hub podría fallar si no estás autenticado localmente o si el repo es privado/nuevo."
        )

    logger.info("=" * 60)
    logger.info("Iniciando empaquetado a GGUF y subida a HuggingFace")
    logger.info("  Modelo origen:   %s", model_source)
    logger.info("  Repositorio HF:  %s", repo_id)
    logger.info("  Cuantización:    %s", quant)
    logger.info("=" * 60)

    # Verificamos si es una ruta local o un HF ID
    if os.path.isdir(model_source):
        logger.info("Detectado modelo local en directorio: %s", model_source)
    else:
        logger.info("Se intentará descargar el modelo desde el Hub de HuggingFace: %s", model_source)

    try:
        logger.info("Cargando modelo y tokenizador con Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_source,
            max_seq_length=args.max_seq_len,
            dtype=None,             # Detectar automáticamente bf16/fp16
            load_in_4bit=False,     # Cargamos el modelo normal, la cuantización se hace en la exportación GGUF
        )
    except Exception as e:
        logger.error("Error al cargar el modelo '%s': %s", model_source, e)
        sys.exit(1)

    # Si se especificó un directorio local, guardamos el GGUF localmente primero
    if args.save_local_dir:
        local_dir = Path(args.save_local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_repo_save = str(local_dir / repo_id.split("/")[-1])
        logger.info("Guardando versión GGUF (%s) localmente en: %s", quant, local_repo_save)
        try:
            model.save_pretrained_gguf(
                local_repo_save, 
                tokenizer, 
                quantization_method=quant
            )
            logger.info("Guardado local exitoso.")
        except Exception as e:
            logger.error("Error al guardar localmente: %s", e)

    # Subir a HuggingFace Hub
    logger.info("Empaquetando a GGUF y subiendo al Hub bajo el repo: %s", repo_id)
    try:
        model.push_to_hub_gguf(
            repo_id,
            tokenizer,
            quantization_method=quant,
            token=token,
        )
        logger.info("=" * 60)
        logger.info("¡Proceso completado exitosamente!")
        logger.info("Puedes usar tu modelo con llama.cpp usando el repositorio: %s", repo_id)
        logger.info("=" * 60)
    except Exception as e:
        logger.error("Error al subir el modelo a HuggingFace: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
