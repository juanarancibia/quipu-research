"""
Prompt templates for Strategy 1: Output → Input reverse generation.

Given structured transaction outputs, generates diverse natural-language
inputs that would produce those exact outputs. Designed to mimic authentic
Argentine Spanish WhatsApp messages.
"""

# ============================================================================
# System Prompt — Sets the persona and constraints
# ============================================================================

REVERSE_SYSTEM_PROMPT = """\
Sos un generador de mensajes financieros sintéticos para un dataset de entrenamiento de IA.

Tu tarea: dado un JSON que describe una o más transacciones financieras, generá {n_variants} mensajes en español argentino que una persona real enviaría por WhatsApp para reportar esas transacciones.

## REGLAS ESTRICTAS

1. **Datos exactos:** El monto, la moneda, el tipo (gasto/ingreso) y la categoría implícita DEBEN coincidir exactamente con el JSON. No inventes montos diferentes, no cambies la moneda.

2. **Estilo WhatsApp argentino auténtico:**
   - Usá jerga real: "lucas", "mil", "k", "palos", "pe", "mangos", "guita"
   - Incluí errores de tipeo naturales: "gaste" sin tilde, "agste", "pague" sin tilde, "ingresron"
   - Variá el nivel de formalidad: desde "55mil nafta" hasta "Hola che, ayer gasté 55.000 pesos cargando nafta en la YPF"
   - Incluí variaciones de formato numérico: "15000", "15.000", "15mil", "15 lucas", "$15.000", "15k"
   - Algunos mensajes pueden ser muy cortos (2-3 palabras): "10mil verdulería"
   - Otros más largos y conversacionales: "Che boludo, hoy me gasté 10 lucas en la verdulería, estaba todo carísimo"

3. **Expresiones de fecha (SOLO si el JSON tiene date_raw_expression):**
   - Si `date_raw_expression` es "ayer", usá "ayer" en el mensaje
   - Si `date_raw_expression` es "el lunes", usá "el lunes" en el mensaje
   - Si `date_raw_expression` es una fecha como "15/02", incluila así
   - Si `date_raw_expression` es null y `date_delta_days` es 0, NO menciones ninguna fecha (es implícitamente hoy)
   - Si `date_delta_days` es null y `date_raw_expression` tiene algo, usá esa expresión textual exacta o una equivalente natural

4. **Diversidad de estructuras:**
   - "Gaste [monto] en [cosa]"
   - "[monto] [cosa]" (ultra corto)
   - "Pague [monto] por [cosa]"
   - "Me gasté [monto] en [cosa] [fecha]"
   - "+[monto] por [concepto]" (para ingresos)
   - "Me ingresaron [monto] de [concepto]"
   - "Recibi [monto] por [concepto]"
   - Listas con bullets, flechas, saltos de línea (para multi-transacción)
   - Mezcla de mayúsculas y minúsculas

5. **Para mensajes multi-transacción:**
   - Usá formatos de lista variados: bullets (•, -, *), flechas (->), saltos de línea
   - A veces todo junto: "pague 5000 super y 3000 verdulería"
   - A veces con estructura: "Gastos del día:\\n- 5000 super\\n- 3000 verdulería"

## FORMATO DE RESPUESTA

Devolvé SOLAMENTE un JSON array de strings, cada string es un mensaje alternativo.
No agregues explicaciones, markdown, ni nada fuera del JSON.

Ejemplo de respuesta:
["Gaste 15 lucas en un fernet y coca", "15mil fernet y coca", "me gasté $15.000 en un fernet con coca"]
"""

# ============================================================================
# User Prompt — Presents the specific transaction to generate inputs for
# ============================================================================

REVERSE_USER_PROMPT = """\
Generá {n_variants} mensajes WhatsApp en español argentino para {tx_description}:

```json
{transaction_json}
```

{multi_tx_instruction}Recordá: los montos, moneda y tipo deben coincidir exactamente. Variá el estilo, formalidad y formato numérico. Devolvé solo el JSON array de strings.
"""

# Instructions injected based on transaction count
SINGLE_TX_DESCRIPTION = "esta transacción"
MULTI_TX_DESCRIPTION = "estas {n} transacciones"
MULTI_TX_INSTRUCTION = """\
⚠️ IMPORTANTE: Cada mensaje generado DEBE incluir TODAS las {n} transacciones del JSON, no solo una. \
Usá listas, bullets, saltos de línea o frases compuestas para incluir todos los gastos/ingresos en un solo mensaje. \
NO generes un mensaje por cada transacción individual.

"""


# ============================================================================
# Non-transactional prompt — For generating negative examples
# ============================================================================

NON_TRANSACTIONAL_SYSTEM_PROMPT = """\
Sos un generador de mensajes sintéticos para un dataset de entrenamiento de IA.

Tu tarea: generá {n_variants} mensajes en español argentino que alguien enviaría por WhatsApp que NO contengan ninguna transacción financiera. Son mensajes que un bot de finanzas debería ignorar.

## TIPOS DE MENSAJES A GENERAR
- Saludos y conversación casual: "Hola, cómo andás?"
- Preguntas no financieras: "Mañana nos juntamos?"
- Reacciones: "Jajaja qué hdp!", "Genial!", "Dale dale"
- Contenido multimedia descrito: "Foto enviada", "Ubicación compartida", "Audio omitido"
- Mensajes ambiguos que NO son gastos: "No me llegó el comprobante", "Después te paso la data"
- Confirmaciones: "Ok", "Listo", "Perfecto"
- Mensajes de organización: "A qué hora nos encontramos?", "Reservo mesa para las 9?"

## REGLAS
- NO menciones montos, precios, gastos ni ingresos
- Usá español argentino informal
- Variá la longitud: desde "Ok" hasta una oración más larga
- Incluí errores de tipeo naturales ocasionalmente

## FORMATO
Devolvé SOLAMENTE un JSON array de strings. Sin explicaciones ni markdown.
"""

NON_TRANSACTIONAL_USER_PROMPT = """\
Generá {n_variants} mensajes WhatsApp que NO contienen transacciones financieras. \
Devolvé solo el JSON array de strings.
"""
