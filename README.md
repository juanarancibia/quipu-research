# E2E NLU Pipeline: The process of building a 4B SLM for argentinian slang

This repo shows how I built the NLU core for Quipu, a personal finance WhatsApp bot. This is not just another fine-tuning tutorial. It's an end-to-end process of discovering how to distill an Small Language Model (SLM) for an specific task, moving from propietary OpenAI APIs to a quantized locally-hosted model. 

In  this journey I had to go through all the steps of an ML system: Data Acquisition, Evaluation of frontier models, DSPy prompt optimization, Bidirectional Synthethic Data generation, Fine Tuning and Post-Traning Data Quantization (PTQ).

### What is Quipu?
Quipu is an AI powered personal finance assistant built on top of WhatsApp. Instead of forcing users to navigate complex apps or fill out spreadsheets, it allows them to simply send a text or voice note in natural language to automatically log expenses, categorize spending, and track their budget. 

**The core engineering challenge:** Extracting strict financial JSONs from chaotic, unstructured chat data, and doing it fast enough for a real-time messaging UX, accurate enough for accounting, and cheap enough to scale locally without drowning in costs on proprietary API calls.

## Baseline

Quipu is a bot that's already being used in production. Since I'm a daily user of the bot, I experienced the time delays and misunderstanding of context in person, for some cases it failed identifying the correct expense category.

For example, "Gaste 10 lucas en el chino" is a complex task for an out of the box LLM, it has to understand that "lucas" means thousands and that "chino" refers to a local grocery store. This is a textbook case for model specialization.

Before doing anything, I needed to know exactly how bad the problem was. So I built a robust local evaluation framework and ran it against Frontier Models (GPT-4o-mini, GPT-5-mini, Gemma-3-12B, MiniMax 2.5).

I had a clear goal of matching or beating GPT-4o-mini's F1-Score and Precision on transaction extraction, but running it locally, fast, and cheap. Considering this a great experiment to learn how to distill a model end-to-end, I started with the foundation, Data Acquisition.

## Data Acquisition

As we all know having a golden dataset that has a good quality is the most important part of every ML pipeline. Citing the most repeated phrase in machine learning: "Garbage in, garbage out". 

Considering I was an actual user of Quipu, I had the advantage of having real world cases of input/output pairs in my own WhatsApp chat with the bot. So, I exported my personal chat history and built a parser to transform all the raw messages into a JSON format of input/output pairs. 

Then, keeping in mind that the bot's historical outputs weren't always precise, I built a custom local UI to manually curate message by message. This hands on approach was necessary in order to have all the data correctly labeled and guarantee a golden dataset that was pure quality, establishing the ground truth for the rest of the pipeline.

This manual curation process wasn't just about labelling the data, it was crucial for defining the final domain schema correctly. By reviewing hundreds of messages with my own eyes, I could see exactly what information was relevant to extract and what edge cases broke the structure. It allowed me to iterate on the entity definitions (like amount, category, description, date, etc.) and establish a strict JSON contract that actually represented the real world use case. The final schema was defined as:

```python
class Transaction(TypedDict):
    """A single financial transaction (expense or income)."""
    type: Literal["expense", "income"]
    amount: float
    currency: str
    category: str
    description: str
    date_delta_days: Optional[int]  # null if date is absolute/complex, 0 if no date mentioned, -1/1/etc if relative
    date_raw_expression: Optional[str]  # null if no date mentioned, "ayer" if relative, "el 4 de julio" if absolute
```

In the end, I managed to curate ~240 messages that represented about ~300 transactions, then scaling it with reverse synthetic data generation to 364 messages that represented 433 transactions.

## Evaluation

Before any optimization or training I needed to define a metric that let me measure how well a model performed in the desired domain. Considering that we were measuring transactions, some attributes of it are more critical than others. For example, the description was subjective noise that would affect the score when it really isn't that crucial for this specific case.

With all this into account, I was able to define the following metrics:

* **F1 Score** - In a single message multiple transactions can be expressed, this metric shows if the model was able to extract the exact **number of transactions**, **type** and **amount**.
* **Strict JSON Score** - A simple metric to penalize when the model doesnt answer with an exact JSON output. If it outputs a valid JSON its 1, and 0 if not.
* **Category Match** - Asserts whether the extracted output matches the transaction category or not, this is an important property of a transaction.
* **Entity Accuracy** - This represents how well the extraction performed, applying specific domain weights:
   * Finance (50%): amount 35% + currency 15%
   * Classification (30%): type 15% + category 15%
   * Temporal (20%): date_delta_days 70% + date_raw_expression 30% (normalized substring match)

Now that I had the evaluation metrics defined I was able to run the first evaluation with Zero-Shot Prompting

| Model / API (OpenRouter) | N Samples | F1-Score | Entity Accuracy | Category Match |
|---------------------------|------------|----------|-----------------|-----------|
| `gpt-4o-mini` | 25 | **0.9855** | 82.8% | 61.0% |
| `minimax/minimax-m2.5` | 15 | **0.9714** | 78.0% | 53.3% |
| `google/gemma-3-12b-it` | 15 | **0.9032** | **86.4%** | **63.3%** |

> **Note:** The `Strict JSON Score` for API OpenRouter provider is `0.0%` since they inject code wrappers or additional text that fail the strict validation of a strict JSON.


## Prompt optimization

The baseline evaluation showed that GPT-4o-mini was very good, but not perfect. Since my goal was to use this model as a "Teacher" to generate synthetic data and train a smaller "Student" model, the Teacher had to be better. If the Teacher hallucinated or misidentified a category, the Student would learn that error.

Instead of manually tweaking the prompt by trial and error, I used DSPy. DSPy is a framework that algorithmically optimizes LM prompts and weights. You give it your pipeline, your evaluation metric (the one defined before), and a few examples, and it "compiles" the best possible prompt.

At first, I ran an initial optimization using a small split of the dataset. The analysis revealed that generating valid JSON wasn't the main issue. The real bottlenecks were semantic ambiguity in expense categories (60% of errors) and inconsistent serialization of relative dates, for example: "ayer".

#### Error Breakdown by Field (Test Set: 5 transactions)

| Campo | `gpt-4o-mini` | `minimax-m2.5` | `gemma-2-9b-it` |
|-------|:---:|:---:|:---:|
| Hard Gate (JSON inválido) | 0 / 5 (0%) | 0 / 5 (0%) | **1 / 5 (20%)** |
| Amount errors | 0 / 5 (0%) | 0 / 5 (0%) | 1 / 5 (20%) |
| Currency errors | 0 / 5 (0%) | 0 / 5 (0%) | 1 / 5 (20%) |
| Type errors | 0 / 5 (0%) | 0 / 5 (0%) | 1 / 5 (20%) |
| **Category errors** | 3 / 5 (60%) | 3 / 5 (60%) | **5 / 5 (100%)** |
| **Date delta errors** | 0 / 5 (0%) | 1 / 5 (20%) | **5 / 5 (100%)** |
| **Date expression errors** | **4 / 5 (80%)** | 1 / 5 (20%) | 1 / 5 (20%) |

Seeing this results, I escalated to MIPROv2 (a more aggressive optimizer) on my initial dataset of 112 entries. The result was a massive overfitting issue, ~99% score on the train set but only ~83% on the test set. The optimizer had memorized the small dataset but couldn't generalize. 
Instead of tweaking hyperparameters, I paused optimization, went back to the Data Acquisition phase, and manually labeled ~150 more transactions to increase lexical and categorical diversity.

Now that I had a richer dataset, I made three technical decisions changes before the final run:

* Enforced JSON Mode: Hardcoded response_format={"type": "json_object"} in the API calls to guarantee 100% strict JSON compliance.

* Removed Absolute Dates: I dropped the current_date from the prompt, forcing the LLM to work purely with relative temporal logic, eliminating date delta confusion.

* Teacher-Student Setup: Used gpt-5.1-codex-max as the Teacher model to compile the optimal instructions for gpt-5-mini, the Student.

The Result of all this was a final compiled program that achieved a 98.06% Test Score. Zero errors in Amount, Currency, or Type. The only remaining errors were category ambiguities.


#### Final result (Test Set: 55 transactions)

| Model (Student) | Teacher | Optimizer | Score Test | Cost in USD |
|---------------------|---------|-----------|------------|-----------|
| `gpt-5-mini` | `gpt-5.1-codex-max` | MIPROv2 (`light`) | **98.06%** | ~$0.78 |



## Synthetic Data generation
*How I scaled from 300 manual examples to 10,000.*

Initially, I tried "Forward Generation" prompting the model to hallucinate a fake WhatsApp message from scratch, and then extracting the structured JSON from it. However, because my DSPy optimized extraction prompt relied heavily on Chain of Thought (CoT) to ensure accuracy, generating and parsing data this way was painfully slow and extremely expensive. Generating 10k new transactions not only would take forever, it would be very expensive.

Considering this, I pivoted to a "Reverse Generation" approach, which proved to be infinitely more scalable and cost effective:

* **Algorithmic Baseline**: First, I algorithmically generated thousands of valid target JSONs, this is the desired structured outputs. Then, I used balance analyzers to ensure categories, amounts, and every property in a trasaction weren't heavily unbalanced.

* **Reverse Hallucination**: I prompted an LLM to look at the JSON and hallucinate a messy, slang filled WhatsApp message that would logically result in that exact data.

* **The Validation Filter**: To ensure quality, I passed that generated text back through a smaller, cheaper model using a basic prompt. The goal was to see if the extracted transaction matched the algorithmically generated original.

Even though I had to discard many messages due to extraction or generation errors in the validation phase, this Reverse method was so fast and cheap that the high discard rate didn't matter. It allowed me to scale my dataset efficiently while maintaining certainty over the quality of the generated syntethic data.

#### Final results on the generated data

| Métrica | **Synthetic Generated Data (9.2K)** |
|:---|:---:|
| **Records Totales** | **9.196** (10.234 txns) |
| **EXPENSE** | **65.2%** |
| **Absolut dates** | **3.0%** |
| **"Hoy" (Delta=0)** | **72.3%** |
| **Amount < $15K ARS** | **25.1%** |
| **Amount > $1M ARS** | **10.7%** |

##### Category distribution (Expenses — 6669 txns)
| Categoría | Count | % |
|-----------|-------|---|
| Vivienda_Servicios | 739 | 11.1% |
| Transporte | 661 | 9.9% |
| Supermercado_Despensa | 650 | 9.7% |
| Salud_Bienestar | 638 | 9.6% |
| Ocio_Entretenimiento | 617 | 9.3% |
| Deporte_Fitness | 606 | 9.1% |
| Financiero_Tarjetas | 604 | 9.1% |
| Comida_Comprada | 595 | 8.9% |
| Educacion_Capacitacion | 564 | 8.5% |
| Hogar_Mascotas | 549 | 8.2% |
| Regalos_Otros | 446 | 6.7% |

##### Category Distribution (Income — 3565 txns)
| Categoría | Count | % |
|-----------|-------|---|
| Salario_Honorarios | 1208 | 33.9% |
| Inversiones_Finanzas | 910 | 25.5% |
| Regalos_Otros | 786 | 22.0% |
| Ventas_Negocios | 661 | 18.5% |


## Distillation and Fine Tuning

Now that I had the necessary datasets, the synthetic dataset for training (~10k records) and the golden dataset for validation, I was able to fine tune different models and evaluate them to see which scored the highest and met my initial goal, running it locally, fast, and cheap.

To handle the training compute efficiently, I ysed a RunPod instance with a RTX 5090 cloud GPU. This provided the necessary VRAM and cores to run Supervised Fine Tuning without memory bottlenecks.

I selected the Qwen3.5 open weights models (specifically the 0.8B, 2B and 4B) and formatted my synthetic data into the ChatML format for the fine tuning process.

To prove the value of the distillation process, I ran a final evaluation comparing these fine tuned SLMs against frontier models (like gpt-5-mini) and larger base models (Qwen-2.5-7B) in Zero-Shot and Few-Shot configurations.

| Model | Modality | F1-Score | Precision | Recall | Strict JSON | Entity Acc | Cat Match | Latencia (p50) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `gpt-5-mini` | Zero-Shot | 100.00% | 100.00% | 100.00% | 100.00% | 96.25% | 79.56% | 5.9s |
| `gpt-5-mini` | Few-Shot (3) | 100.00% | 100.00% | 100.00% | 100.00% | 95.61% | 75.28% | 4.7s |
| `Qwen-2.5-7B` | Zero-Shot | 83.81% | 97.78% | 73.33% | 74.00% | 67.46% | 55.78% | 1.6s |
| `Qwen-2.5-7B` | Few-Shot (3) | 86.73% | 92.45% | 81.67% | 86.00% | 82.12% | 49.78% | 2.5s |
| `Qwen3.5-0.8B` | Fine-Tuned | 93.62% | 92.15% | 95.14% | 100.00% | 88.05% | 66.16% | 0.92s |
| `Qwen3.5-2B` | Fine-Tuned | 94.06% | 92.79% | 95.37% | 100.00% | 88.87% | 73.18% | 1.5s |
| **`Qwen3.5-4B`** | **Fine-Tuned** | **97.57%** | **97.46%** | **97.69%** | **100.00%** | **94.93%** | **80.77%** | **2.2s** |

Some insights from this evaluation:
1. **The Power of SFT in SLMs**: A 4B parameter model fine tuned for a specific task comfortably outperforms a modern 7B parameter base model, even when the latter uses few-shot prompting. The Qwen3.5-4B fine tuned achieved an F1 of 97.57% vs the 7B's 86.73%, and crushed it in structural fidelity (Strict JSON 100% vs 86%).
2. **The Few-Shot Trap**: Interestingly, for both gpt-5 and Qwen-2.5, adding few-shot examples improved structural metrics but made worse the category classification accuracy. This suggests that for highly specific business rules, large models over fixate on the specific categories present in the few-shot context window, losing their broader abstraction capabilities.
3. **Competitive Fine tuned Model**: The fine tuned Qwen3.5-4B performed almost the same as gpt-5-mini in extraction accuracy but actually beat it in Category Match (80.7% vs 79.5%). Injecting highly specific domain knowledge (Argentine slang and financial jargon) via SFT makes better results than the general models.


## Post Training Quantization (PTQ)

While the fine tuned model achieved incredible accuracy, running a 16 bit 4B model is still too heavy for edge deployment or cheap cloud hosting. I needed to reduce the VRAM consume significantly.

To achieve this, I applied Post Training Quantization (PTQ), specifically targeting the Q4_K_M GGUF format. Quantization essentially compresses the model's weights from 16 bits down to 4 bits. This drastically reduces the memory required and speeds up inference, but it usually comes at a cost, less precision in the final model. I defined a strict threshold, if the F1 Score dropped below 95%, the quantized model would be rejected.

| Local Model | N Samples | F1-Score | Precision | Recall | Strict JSON | Entity Acc | Cat Match | Latency (p50) | Latency (p99) |
|--------------|------------|----------|-----------|--------|-------------|------------|-----------|----------------|----------------|
| `Qwen-3.5-quipu-q4_k_m (GGUF)` | 363 | 97.23% | 97.00% | 97.45% | 100.00% | 94.34% | 80.64% | 2.240s | 13.268s |

The results were amazing:
* I only lost ~0.25% of F1 Score
* The model size decreased from 8.42GB to 2.71 GB, a ~67% disk space reduction.
* The GGUF model retained 100% Strict JSON output capabilities. No format degradation occurred despite the 4-bit compression.
* The local llama.cpp server successfully interpreted the System Prompt and Message Roles perfectly, validating that the Quipu backend can switch between commercial APIs and local endpoints seamlessly.

This results showed that I've successfully compressed a frontier level extraction capability into a few gigabytes of VRAM, running locally at $0 API cost.


## Inference Architecture

While the p50 latency (2.24s) was acceptable, the p99 latency of 13.26 seconds was a massive red flag. For a real time WhatsApp bot, a 13 second tail latency means the user has to wait a lot of time to get answered. I needed maximum concurrency and strict Time To First Token (TTFT) optimization.

Building the local inference engine for the Q4_K_M GGUF model became an architectural battle against memory bandwidth:

* **The Custom Rust Engine & M-RoPE**: Initially I built a custom Rust inference server using llama-cpp-2. To avoid re-evaluating the heavy System Prompt (~600 tokens), I tried to truncate the KV cache after each request. However, Qwen 3.5 uses M-RoPE (Multimodal Rotary Position Embeddings). Its KV cache is not sequential, meaning truncation corrupted the KV Cache and broke the model.
* **The Cache Cloning Workaround**: I managed to fix the TTFT by maintaining a frozen master sequence (seq_id=0) for the System Prompt and cloning its KV cache into an ephemeral user sequence (seq_id=1) for each request. This dropped the prefill time from 750ms to ~50ms.
* **Llama Server**: Despite fixing the cache, the Rust engine still processed users sequentially. Instead of reinventing complex memory slot management for continuous batching, I switch to something simpler, replacing the custom engine with the highly optimized `llama-server`.

📊 Final Stress Test Results

By pivoting to `llama-server`, the engine stopped queuing users one by one and started processing up to 4 simultaneous messages in parallel, sharing the KV Cache natively.

Here is the architectural evolution evaluated under a strict k6 Stress Test (5 Concurrent Users over 30 seconds):

| Architecture | Prefill (TTFT) | Processing Mode | Min Latency | Max Latency | Throughput (30s) |
|:---|:---|:---|:---|:---|:---|
| **Rust (Clear Cache)** | ~750ms | Sequential (1 en 1) | 3.08s | 15.32s | 14 |
| **Rust (Prefix Caching)** | ~50ms | Sequential (1 en 1) | 2.24s | 11.06s | 18 |
| **llama-server** | ~50ms | **Parallel (Batching)** | 3.46s | **8.94s** | **30** |


### Architectural Insights

Looking closely at the stress test results, a critical system design paradox emerges: **Why did the Minimum Latency increase (from 2.24s to 3.46s) when moving to the highly optimized `llama-server`?**

This is the textbook trade off of **Continuous Batching**. 
In the sequential Rust engine, the first user in the queue gets 100% of the GPU's memory bandwidth and Compute (SMs), resulting in a lightning-fast response (2.24s). However, user #5 suffers the accumulated wait time of all previous requests, increasing the tail latency to an unacceptable 11.06s.

By delegating the engine to `llama-server`, we introduce parallel batching. The GPU now processes multiple contexts simultaneously. The compute effort per forward pass is heavier, meaning user #1 loses their preferred sequential treatment (latency rises to 3.46s). In exchange, we crush the tail latency for the whole system (Max Latency drops from ~15s to 8.94s) and increase overall throughput by 114% (14 to 30 requests). 

I sacrificed the best case scenario to guarantee a predictable, scalable system. 

## Future Work
While dropping the max latency under heavy load to ~9 seconds is a massive architectural win, **9 seconds is still a poor UX for a conversational WhatsApp bot.** To push this SLM into real time territory, the next optimization cycles should focus on:

1. **Decoupling Prefill and Decode (Chunked Prefill):** Currently, the heavy compute phase of reading the few-shot prompt (Prefill) blocks the memory-bound token generation (Decode) for other users in the batch. Implementing Chunked Prefill will prevent long prompts from spiking the tail latency of concurrent requests.
2. **Speculative Decoding:** Since Quipu's output is a highly structured, predictable JSON, using a microscopic draft model (e.g., <0.5B params) to generate tokens and having our 4B model simply verify them in parallel could theoretically double our Tokens/sec (TPS) during the decode phase.
