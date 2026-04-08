# Quipu NLU - Data Labeling Guidelines
**Version:** 1.1 (Phase 0 - 2026)
**Objective:** Establecer un *Ground Truth* determinístico para entrenar y evaluar los modelos NLU de Quipu. Minimizar la varianza humana para reducir la entropía en el *output space*.

## 🧠 Reglas de Oro (Core Principles)

1. **Basado en el RAW INPUT:** Etiquetá basado estrictamente en la información que está en el mensaje crudo de WhatsApp. No asumas contexto que el modelo no tiene forma de saber.
2. **Determinismo:** Ante el mismo mensaje en dos días distintos, tenés que elegir la misma categoría. Cero "intuición", 100% reglas empíricas.
3. **Multi-Intent (List[Transaction]):** Si el mensaje contiene más de un gasto ("10 lucas súper y 2k propina"), SE DEBE separar en dos objetos JSON distintos dentro de la lista. No los fusiones.
4. **Fechas Relativas vs Absolutas:** Distinguir claramente entre fechas relativas (que el modelo puede calcular) y fechas absolutas (que requieren procesamiento externo en Rust).
5. **Conversation Date:** SIEMPRE capturar la fecha en que el usuario envió el mensaje (`conversation_date` en formato YYYY-MM-DD). Este es el punto de referencia para todos los `date_delta_days`.

---

## 📅 Reglas de Etiquetado de Fechas

### Regla 1: Sin fecha mencionada
**Input:** "1000 coca" o "Gasté 5000 en el super"

```json
{
  "date_delta_days": 0,
  "date_raw_expression": null
}
```

- **date_delta_days = 0:** Asumimos mismo día que la conversación
- **date_raw_expression = null:** No hay expresión temporal en el mensaje

---

### Regla 2: Fecha relativa explícita
**Input:** "Ayer compré pan por 500"

```json
{
  "date_delta_days": -1,
  "date_raw_expression": "Ayer"
}
```

**Input:** "Mañana pago el alquiler"

```json
{
  "date_delta_days": 1,
  "date_raw_expression": "Mañana"
}
```

**Input:** "Hace 3 días fui al super"

```json
{
  "date_delta_days": -3,
  "date_raw_expression": "Hace 3 días"
}
```

- **date_delta_days:** Número calculable desde conversation_date
- **date_raw_expression:** La expresión textual exacta del usuario

**Expresiones relativas comunes:**
- "ayer" → -1
- "hoy" → 0
- "mañana" → 1
- "anteayer" → -2
- "hace X días" → -X
- "en X días" → +X

---

### Regla 3: Fecha absoluta o compleja
**Input:** "El 4 de julio pagué la obra social"

```json
{
  "date_delta_days": null,
  "date_raw_expression": "El 4 de julio"
}
```

**Input:** "El lunes pasado gasté 5000"

```json
{
  "date_delta_days": null,
  "date_raw_expression": "El lunes pasado"
}
```

**Input:** "El 15/02 fue mi cumpleaños"

```json
{
  "date_delta_days": null,
  "date_raw_expression": "El 15/02"
}
```

- **date_delta_days = null:** El modelo no puede calcularlo, lo procesa Rust
- **date_raw_expression:** La expresión textual exacta del usuario

**¿Por qué null?** Fechas como "el 4 de julio" o "el lunes pasado" requieren:
- Conocer el calendario completo
- Inferir el año correcto
- Calcular días de la semana
- Esto lo hace Rust en el backend, no el modelo

---

## 📋 Tabla Resumen

| Caso | Input ejemplo | date_delta_days | date_raw_expression |
|------|---------------|-----------------|---------------------|
| Sin fecha | "1000 coca" | 0 | null |
| Fecha relativa simple | "Ayer 1000 coca" | -1 | "Ayer" |
| Fecha relativa numérica | "Hace 3 días..." | -3 | "Hace 3 días" |
| Fecha absoluta | "El 4 de julio..." | null | "El 4 de julio" |
| Día de semana | "El lunes pasado..." | null | "El lunes pasado" |
| Fecha formato DD/MM | "El 15/02..." | null | "El 15/02" |

---

## 🎯 Guía Práctica para Curadores

### Paso 1: Lee el input completo
Identifica si hay alguna expresión temporal

### Paso 2: Clasifica el tipo de fecha
- ¿No hay fecha? → **date_delta_days: 0, date_raw_expression: null**
- ¿Es relativa ("ayer", "mañana", "hace X días")? → **Calcula delta + copia expresión**
- ¿Es absoluta ("el 4 de julio", "el lunes")? → **date_delta_days: null + copia expresión**

### Paso 3: En la UI
- **Date Offset:** Deja vacío si es null, o ingresa el número si es relativo
- **Date Expression:** Deja vacío si es null, o copia la expresión exacta del usuario

---

## 🗂️ Resolución de Fronteras Ambiguas (Edge Cases)

### 1. La Regla de la "Hamburguesa" (Alimentos vs. Ocio)
* **Problema:** ¿Un helado o una cena con amigos es comida o salida?
* **Regla:** Se clasifica por **Naturaleza de la Adquisición**, no por el producto.
    * Si requiere preparación o es materia prima (Verdulería, Carnicería, Súper, Dietética) ➡️ `Supermercado_Despensa`.
    * Si está listo para consumir y servido por terceros (Restaurante, Delivery, Bar, Helado en la calle, Café) ➡️ `Comida_Comprada`.

### 2. La Regla del Transporte (Movilidad vs. Vacaciones)
* **Problema:** ¿Un vuelo es transporte?
* **Regla:** `Transporte` es estrictamente movilidad operativa diaria (Nafta, SUBE, Uber, Peaje, Estacionamiento). Si es un pasaje de avión para turismo, va a `Ocio_Entretenimiento` (o la categoría de vacaciones que corresponda en el futuro). 

### 3. La Regla de la Salud (Estética vs. Medicina)
* **Problema:** ¿La peluquería es salud?
* **Regla:** `Salud_Bienestar` engloba el cuerpo humano como máquina. Aplica tanto a mantenimiento médico (Hospital, Farmacia, Psicólogo) como estético (Peluquería, Masajes, Cuidado personal). 

### 4. La Regla del "No Sé" (Fallback)
* **Problema:** Mensajes muy crípticos (Ej: "5000 a tito").
* **Regla:** Si la intención transaccional es imposible de deducir del mensaje crudo, enviarlo directo a `Regalos_Otros`. No intentes adivinar ni forzar una categoría, porque estarías inyectando ruido estadístico en clusters limpios.

---

## 📋 Diccionario de Categorías Definitivo

| Target / Label | Tipo | Descripción estricta para el etiquetador |
| :--- | :--- | :--- |
| `Salario_Honorarios` | Ingreso | Ingreso principal recurrente (sueldo, honorarios fijos). |
| `Ventas_Negocios` | Ingreso | Ingresos esporádicos, ventas secundarias, emprendimientos. |
| `Inversiones_Finanzas` | Ingreso | Rulos financieros, compra/venta USD, reintegros, cobro de deudas. |
| `Supermercado_Despensa` | Gasto | Materia prima e insumos de hogar (Verdulería, Súper, Chino). |
| `Comida_Comprada` | Gasto | Comida elaborada, delivery, restaurantes, bares, cafeterías. |
| `Vivienda_Servicios` | Gasto | Gastos fijos estructurales: Alquiler, expensas, luz, agua, internet. |
| `Transporte` | Gasto | Movilidad diaria: Uber, Nafta, SUBE, Peajes, Estacionamiento. |
| `Salud_Bienestar` | Gasto | Médicos, farmacia, psicología, peluquería, cuidado corporal. |
| `Ocio_Entretenimiento` | Gasto | Cine, recitales, suscripciones (Netflix/Spotify), salidas NO gastronómicas. |
| `Deporte_Fitness` | Gasto | Cuota del club, gimnasio, alquiler de canchas, indumentaria deportiva, torneos. |
| `Educacion_Capacitacion` | Gasto | Postgrados, facultad, cursos, libros, plataformas de aprendizaje. |
| `Hogar_Mascotas` | Gasto | Arreglos de casa, personal de limpieza, veterinaria, alimento animal. |
| `Financiero_Tarjetas` | Gasto | Pago de resúmenes de tarjeta, comisiones bancarias, impuestos. |
| `Regalos_Otros` | Gasto | Donaciones, regalos a 3ros, gastos indescifrables por falta de contexto. |