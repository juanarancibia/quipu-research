from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "juanarancibia/Qwen3.5-0.8B-quipu"
print(f"📥 Descargando/Cargando {model_id} desde Hugging Face...")

# Cargamos el tokenizer y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Usamos device_map="auto" para que use tu GPU local si tenés, o la CPU directamente
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

print("✅ Modelo cargado. Preparando prompt...")

# El mismo prompt de tu error
messages = [
    {"role": "system", "content": "You are Quipu, an AI trained to categorize financial transactions. Respond ONLY with valid JSON."},
    {"role": "user", "content": "Hola como estas vos sabes que me fui a jugar un partido de fútbol y gaste 45000 pesos"}
]

# Preparamos el formato ChatML que aprendió en el Fine-Tuning
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("🧠 Quipu está pensando...\n")
outputs = model.generate(
    **inputs, 
    max_new_tokens=100, 
    temperature=0.0, # Queremos que sea determinista (cero creatividad)
    pad_token_id=tokenizer.eos_token_id
)

# Decodificamos solo la respuesta nueva
input_length = inputs['input_ids'].shape[1]
response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

print("="*40)
print("💰 RESPUESTA DE QUIPU:")
print("="*40)
print(response)
print("="*40)
